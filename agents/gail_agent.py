from collections import namedtuple
import os.path as osp

import torch
import torch.nn.utils as U
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import autograd
from torch.autograd import Variable

from helpers import logger
from helpers.dataset import Dataset
from helpers.console_util import log_env_info, log_module_info
from helpers.distributed_util import average_gradients, sync_with_root
from agents.nets import GaussPolicy, Value, Discriminator, KYEDiscriminator
from agents.gae import gae


class GAILAgent(object):

    def __init__(self, env, device, hps, expert_dataset):
        self.env = env
        self.ob_space = self.env.observation_space
        self.ob_shape = self.ob_space.shape
        self.ac_space = self.env.action_space
        self.ac_shape = self.ac_space.shape

        log_env_info(logger, self.env)

        self.ob_dim = self.ob_shape[0]  # num dims
        self.ac_dim = self.ac_shape[0]  # num dims
        self.device = device
        self.hps = hps
        self.expert_dataset = expert_dataset
        assert self.hps.clip_norm >= 0
        if self.hps.clip_norm <= 0:
            logger.info("[WARN] clip_norm={} <= 0, hence disabled.".format(self.hps.clip_norm))

        # Parse the label smoothing types
        self.apply_ls_fake = self.parse_label_smoothing_type(self.hps.fake_ls_type)
        self.apply_ls_real = self.parse_label_smoothing_type(self.hps.real_ls_type)

        # Create nets
        self.policy = GaussPolicy(self.env, self.hps).to(self.device)
        sync_with_root(self.policy)
        if not self.hps.shared_value:
            self.value = Value(self.env, self.hps).to(self.device)
            sync_with_root(self.value)
        if self.hps.kye_d_regress:
            self.discriminator = KYEDiscriminator(self.env, self.hps).to(self.device)
        else:
            self.discriminator = Discriminator(self.env, self.hps).to(self.device)
        sync_with_root(self.discriminator)

        # Set up demonstrations dataset
        self.e_dataloader = DataLoader(self.expert_dataset,
                                       self.hps.batch_size,
                                       shuffle=True,
                                       drop_last=True)

        # Set up the optimizers
        self.p_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.hps.p_lr)
        if not self.hps.shared_value:
            self.v_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.hps.v_lr)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.hps.d_lr)

        # Set up the learning rate schedule
        def _lr(t):  # flake8: using a def instead of a lambda
            if self.hps.with_scheduler:
                return (1.0 - ((t - 1.0) / (self.hps.num_timesteps //
                                            self.hps.rollout_len)))
            else:
                return 1.0

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.p_optimizer, _lr)

        log_module_info(logger, 'policy', self.policy)
        if not self.hps.shared_value:
            log_module_info(logger, 'value', self.policy)
        log_module_info(logger, 'discriminator', self.discriminator)

    def parse_label_smoothing_type(self, ls_type):
        """Parse the `label_smoothing_type` hyperparameter"""
        if ls_type == 'none':

            def _apply_ls(labels):
                pass

        elif 'random-uniform' in ls_type:
            # Label smoothing, suggested in 'Improved Techniques for Training GANs',
            # Salimans 2016, https://arxiv.org/abs/1606.03498
            # The paper advises on the use of one-sided label smoothing, i.e.
            # only smooth out the positive (real) targets side.
            # Extra comment explanation: https://github.com/openai/improved-gan/blob/
            # 9ff96a7e9e5ac4346796985ddbb9af3239c6eed1/imagenet/build_model.py#L88-L121
            # Additional material: https://github.com/soumith/ganhacks/issues/10
            _, lb, ub = ls_type.split('_')

            def _apply_ls(labels):
                # Replace labels by uniform noise from the interval
                labels.uniform_(float(lb), float(ub))

        elif 'soft-labels' in ls_type:
            # Traditional soft labels, giving confidence to wrong classes uniformly (all)
            _, alpha = ls_type.split('_')

            def _apply_ls(labels):
                labels.data.copy_((labels * (1. - float(alpha))) + (float(alpha) / 2.))

        elif 'disturb-label' in ls_type:
            # DisturbLabel paper: disturb the label of each sample with probability alpha.
            # For each disturbed sample, the label is randomly drawn from a uniform distribution
            # over the whole label set, regarless of the true label.
            _, alpha = ls_type.split('_')

            def _apply_ls(labels):
                flip = (labels.clone().detach().data.uniform_() <= float(alpha)).float()
                labels.data.copy_(torch.abs(labels.data - flip.data))

        else:
            raise RuntimeError("unknown label smoothing type: '{}'".format(ls_type))
        return _apply_ls

    def predict(self, ob, sample_or_mode):
        # Create tensor from the state (`require_grad=False` by default)
        ob = torch.Tensor(ob[None]).to(self.device)
        # Predict an action
        ac = self.policy.sample(ob) if sample_or_mode else self.policy.mode(ob)
        # Also retrieve the log-probability associated with the picked action
        logp = self.policy.logp(ob, ac)
        # Place on cpu and collapse into one dimension
        if self.hps.shared_value:
            v = self.policy.value(ob).cpu().detach().numpy().flatten()
        else:
            v = self.value(ob).cpu().detach().numpy().flatten()
        ac = ac.cpu().detach().numpy().flatten()
        logp = logp.cpu().detach().numpy().flatten()
        return ac, v, logp

    def train(self, p_rollout, d_rollout, iters_so_far):
        """Train the agent"""

        # Augment `rollout` with GAE (Generalized Advantage Estimation), which among
        # other things adds the GAE estimate of the MC estimate of the return
        gae(p_rollout, self.hps.gamma, self.hps.gae_lambda, rew_key='syn_rews')

        # Standardize advantage function estimate
        p_rollout['advs'] = ((p_rollout['advs'] - p_rollout['advs'].mean()) /
                             (p_rollout['advs'].std() + 1e-8))

        # Create DataLoader objects to iterate over transitions in rollouts
        p_keys = ['obs0', 'obs1', 'acs', 'logps', 'vs', 'advs', 'td_lam_rets']
        p_dataset = Dataset({k: p_rollout[k] for k in p_keys})
        p_dataloader = DataLoader(p_dataset,
                                  self.hps.batch_size,
                                  shuffle=True,
                                  drop_last=True)
        d_keys = ['obs0', 'obs1', 'acs']
        d_dataset = Dataset({k: d_rollout[k] for k in d_keys})
        d_dataloader = DataLoader(d_dataset,
                                  self.hps.batch_size,
                                  shuffle=True,
                                  drop_last=True)

        for _ in range(self.hps.optim_epochs_per_iter):

            for chunk in p_dataloader:

                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Optimize policy.

                # Transfer to device
                state = chunk['obs0'].to(self.device)
                next_state = chunk['obs1'].to(self.device)
                action = chunk['acs'].to(self.device)
                logp_old = chunk['logps'].to(self.device)
                v_old = chunk['vs'].to(self.device)
                advantage = chunk['advs'].to(self.device)
                td_lam_return = chunk['td_lam_rets'].to(self.device)

                # Update running moments
                self.policy.rms_obs.update(state)
                if not self.hps.shared_value:
                    self.value.rms_obs.update(state)

                # Policy loss
                entropy_loss = -self.hps.p_ent_reg_scale * self.policy.entropy(state).mean()
                logp = self.policy.logp(state, action)
                ratio = torch.exp(logp - logp_old)
                surrogate_loss_a = -advantage * ratio
                surrogate_loss_b = -advantage * ratio.clamp(1.0 - self.hps.eps,
                                                            1.0 + self.hps.eps)
                clip_loss = torch.max(surrogate_loss_a, surrogate_loss_b).mean()
                kl_approx = 0.5 * (logp - logp_old).pow(2).mean()
                kl_max = 0.5 * (logp - logp_old).pow(2).max()
                clip_frac = (ratio - 1.0).abs().gt(self.hps.eps).float().mean()
                # Value loss
                if self.hps.shared_value:
                    v = self.policy.value(state)
                else:
                    v = self.value(state)
                clip_v = v_old + (v - v_old).clamp(-self.hps.eps, self.hps.eps)
                v_loss_a = (clip_v - td_lam_return).pow(2)
                v_loss_b = (v - td_lam_return).pow(2)
                v_loss = torch.max(v_loss_a, v_loss_b).mean()
                if self.hps.shared_value:
                    p_loss = clip_loss + entropy_loss + (self.hps.baseline_scale * v_loss)
                else:
                    p_loss = clip_loss + entropy_loss

                # Update parameters
                self.p_optimizer.zero_grad()
                p_loss.backward()
                average_gradients(self.policy, self.device)
                if self.hps.clip_norm > 0:
                    U.clip_grad_norm_(self.policy.parameters(), self.hps.clip_norm)
                self.p_optimizer.step()
                self.scheduler.step(iters_so_far)
                if not self.hps.shared_value:
                    self.v_optimizer.zero_grad()
                    v_loss.backward()
                    average_gradients(self.value, self.device)
                    self.v_optimizer.step()

            if self.hps.kye_p_binning or self.hps.kye_p_regress:

                for chunk in d_dataloader:

                    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Optimize policy for aux tasks.

                    # Transfer to device
                    state = chunk['obs0'].to(self.device)
                    _state = state[:, 0:-1] if self.hps.wrap_absorb else state
                    next_state = chunk['obs1'].to(self.device)
                    action = chunk['acs'].to(self.device)

                    if self.hps.kye_mixing:
                        # Get a minibatch of expert data
                        expert_batch = next(iter(self.e_dataloader))
                        # Unpack
                        inpt_eb_obs0 = expert_batch['obs0']
                        targ_eb_obs0 = expert_batch['obs0']
                        targ_eb_acs = expert_batch['acs']
                        targ_eb_obs1 = expert_batch['obs1']
                        if self.hps.wrap_absorb:
                            inpt_eb_obs0 = inpt_eb_obs0[:, 0:-1]

                    if self.hps.kye_p_binning:
                        aux_loss = F.cross_entropy(
                            input=self.policy.auxo(_state),
                            target=self.get_reward(state, action, next_state)[1]
                        )
                        if self.hps.kye_mixing:
                            aux_loss += F.cross_entropy(
                                input=self.policy.auxo(inpt_eb_obs0),
                                target=self.get_reward(targ_eb_obs0,
                                                       targ_eb_acs,
                                                       targ_eb_obs1)[1]
                            )
                    elif self.hps.kye_p_regress:
                        aux_loss = F.smooth_l1_loss(
                            input=self.policy.auxo(_state),
                            target=self.get_reward(state, action, next_state)[0]
                        )
                        if self.hps.kye_mixing:

                            aux_loss += F.smooth_l1_loss(
                                input=self.policy.auxo(inpt_eb_obs0),
                                target=self.get_reward(targ_eb_obs0,
                                                       targ_eb_acs,
                                                       targ_eb_obs1)[0]
                            )
                    aux_loss *= self.hps.kye_p_scale

                    # Update parameters
                    self.p_optimizer.zero_grad()
                    aux_loss.backward()
                    average_gradients(self.policy, self.device)
                    if self.hps.clip_norm > 0:
                        U.clip_grad_norm_(self.policy.parameters(), self.hps.clip_norm)
                    self.p_optimizer.step()
                    self.scheduler.step(iters_so_far)

        for _ in range(self.hps.d_update_ratio):

            for p_chunk, e_chunk in zip(d_dataloader, self.e_dataloader):

                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Optimize discriminator.

                if self.hps.wrap_absorb:
                    up_obs0 = torch.cat([p_chunk['obs0'],
                                         e_chunk['obs0']], dim=0)
                    non_absorbing_rows = []
                    for j, row in enumerate([up_obs0[i, :] for i in range(up_obs0.shape[0])]):
                        if torch.all(torch.eq(row, torch.cat([torch.zeros_like(row[0:-1]),
                                                              torch.Tensor([1.])], dim=-1))):
                            # logger.info("[INFO] removing absorbing row (#{})".format(j))
                            pass
                        else:
                            non_absorbing_rows.append(j)
                    up_obs0 = up_obs0[non_absorbing_rows, 0:-1]
                else:
                    up_obs0 = torch.cat([p_chunk['obs0'],
                                         e_chunk['obs0']], dim=0)
                self.discriminator.rms_obs.update(up_obs0)
                p_e_loss = self.update_disc(p_chunk, e_chunk)

        # Aggregate the elements to return
        losses = {'pol': clip_loss + entropy_loss,
                  'val': v_loss,
                  'dis': p_e_loss,
                  # Sub-losses
                  'clip_loss': clip_loss,
                  'entropy_loss': entropy_loss,
                  'kl_approx': kl_approx,
                  'kl_max': kl_max,
                  'clip_frac': clip_frac}
        losses = {k: v.cpu().data.numpy() for k, v in losses.items()}

        return losses, self.scheduler.get_last_lr()

    def update_disc(self, p_chunk, e_chunk):
        """Update the discriminator network"""
        # Create tensors from the inputs
        p_state = p_chunk['obs0'].to(self.device)
        p_action = p_chunk['acs'].to(self.device)
        e_state = e_chunk['obs0'].to(self.device)
        e_action = e_chunk['acs'].to(self.device)

        # Compute scores
        p_scores = self.discriminator.D(p_state, p_action)
        e_scores = self.discriminator.D(e_state, e_action)

        # Create entropy loss
        scores = torch.cat([p_scores, e_scores], dim=0)
        entropy = F.binary_cross_entropy_with_logits(input=scores, target=torch.sigmoid(scores))
        entropy_loss = -self.hps.d_ent_reg_scale * entropy

        # Create labels
        fake_labels = 0. * torch.ones_like(p_scores).to(self.device)
        real_labels = 1. * torch.ones_like(e_scores).to(self.device)

        # Parse and apply label smoothing
        self.apply_ls_fake(fake_labels)
        self.apply_ls_real(real_labels)

        if self.hps.use_purl:
            # Create positive-unlabeled binary classification (cross-entropy) losses
            beta = 0.0  # hard-coded, using standard value from the original paper
            p_e_loss = -self.hps.purl_eta * torch.log(1. - torch.sigmoid(e_scores) + 1e-8)
            p_e_loss += -torch.max(-beta * torch.ones_like(p_scores),
                                   (F.logsigmoid(e_scores) -
                                    (self.hps.purl_eta * F.logsigmoid(p_scores))))
        else:
            # Create positive-negative binary classification (cross-entropy) losses
            p_loss = F.binary_cross_entropy_with_logits(input=p_scores,
                                                        target=fake_labels,
                                                        reduction='none')
            e_loss = F.binary_cross_entropy_with_logits(input=e_scores,
                                                        target=real_labels,
                                                        reduction='none')
            p_e_loss = p_loss + e_loss
        # Averate out over the batch
        p_e_loss = p_e_loss.mean()

        # Aggregated loss
        d_loss = p_e_loss + entropy_loss

        if self.hps.grad_pen:
            # Create gradient penalty loss (coefficient from the original paper)
            grad_pen_in = [p_state, p_action, e_state, e_action]
            grad_pen = 10. * self.grad_pen(*grad_pen_in)
            d_loss += grad_pen

        if self.hps.kye_d_regress:
            # Create and add auxiliary loss
            _p_state = p_state[:, 0:-1] if self.hps.wrap_absorb else p_state
            aux_loss = F.smooth_l1_loss(
                input=self.discriminator.auxo(p_state, p_action),
                target=self.policy.sample(_p_state)
            )
            if self.hps.kye_mixing:
                aux_loss += F.smooth_l1_loss(
                    input=self.discriminator.auxo(e_state, e_action),
                    target=self.policy.sample(_p_state)
                )
            aux_loss *= self.hps.kye_d_scale
            d_loss += aux_loss

        # Update parameters
        self.d_optimizer.zero_grad()
        d_loss.backward()
        average_gradients(self.discriminator, self.device)
        self.d_optimizer.step()
        return p_e_loss

    def grad_pen(self, p_ob, p_ac, e_ob, e_ac):
        """Gradient penalty regularizer (motivation from Wasserstein GANs (Gulrajani),
        but empirically useful in JS-GANs (Lucic et al. 2017)) and later in (Karol et al. 2018).
        """
        # Assemble interpolated state-action pair
        if self.hps.wrap_absorb:
            ob_dim = self.ob_dim + 1
            ac_dim = self.ac_dim + 1
        else:
            ob_dim = self.ob_dim
            ac_dim = self.ac_dim
        ob_eps = torch.rand(ob_dim).to(p_ob.device)
        ac_eps = torch.rand(ac_dim).to(p_ob.device)
        ob_interp = ob_eps * p_ob + ((1. - ob_eps) * e_ob)
        ac_interp = ac_eps * p_ac + ((1. - ac_eps) * e_ac)
        # Set `requires_grad=True` to later have access to
        # gradients w.r.t. the inputs (not populated by default)
        ob_interp = Variable(ob_interp, requires_grad=True)
        ac_interp = Variable(ac_interp, requires_grad=True)
        # Create the operation of interest
        score = self.discriminator.D(ob_interp, ac_interp)
        # Get the gradient of this operation with respect to its inputs
        grads = autograd.grad(outputs=score,
                              inputs=[ob_interp, ac_interp],
                              only_inputs=True,
                              grad_outputs=torch.ones(score.size()).to(p_ob.device),
                              retain_graph=True,
                              create_graph=True,
                              allow_unused=self.hps.state_only)
        assert len(list(grads)) == 2, "length must be exactly 2"
        # Return the gradient penalty (try to induce 1-Lipschitzness)
        if self.hps.state_only:
            grads = grads[0]
        grads_concat = torch.cat(list(grads), dim=-1)
        return (grads_concat.norm(2, dim=-1) - 1.).pow(2).mean()

    def get_reward(self, curr_ob, ac, next_ob):
        # Define the obeservation to get the reward of
        ob = next_ob if self.hps.state_only else curr_ob
        # Craft surrogate reward
        assert sum([isinstance(x, torch.Tensor) for x in [ob, ac]]) in [0, 2]
        if not isinstance(ob, torch.Tensor):  # then ac is not neither
            ob = torch.Tensor(ob)
            ac = torch.Tensor(ac)
        # Transfer to cpu
        ob = ob.cpu()
        ac = ac.cpu()
        # Compure score
        score = self.discriminator.D(ob, ac).detach().view(-1, 1)
        sigscore = torch.sigmoid(score)  # squashed in [0, 1]
        # Counterpart of GAN's minimax (also called "saturating") loss
        # Numerics: 0 for non-expert-like states, goes to +inf for expert-like states
        # compatible with envs with traj cutoffs for bad (non-expert-like) behavior
        # e.g. walking simulations that get cut off when the robot falls over
        minimax_reward = -torch.log(1. - torch.sigmoid(score) + 1e-8)
        if self.hps.minimax_only:
            reward = minimax_reward
        else:
            # Counterpart of GAN's non-saturating loss
            # Recommended in the original GAN paper and later in (Fedus et al. 2017)
            # Numerics: 0 for expert-like states, goes to -inf for non-expert-like states
            # compatible with envs with traj cutoffs for good (expert-like) behavior
            # e.g. mountain car, which gets cut off when the car reaches the destination
            non_satur_reward = F.logsigmoid(score)
            # Return the sum the two previous reward functions (as in AIRL, Fu et al. 2018)
            # Numerics: might be better might be way worse
            reward = non_satur_reward + minimax_reward
        # Perform binning
        num_bins = 3  # arbitrarily
        binned = (torch.abs(sigscore - 1e-8) // (1 / num_bins)).long().squeeze(-1)
        for i in range(binned.size(0)):
            if binned.view(-1)[i] > 2. or binned.view(-1)[i] < 0.:
                # This should never happen, flag, but don't rupt
                logger.info("[WARN] binned.view(-1)[{}]={}.".format(i, binned.view(-1)[i]))
        return self.hps.syn_rew_scale * reward, binned

    def save(self, path, iters):
        SaveBundle = namedtuple('SaveBundle', ['model', 'optimizer', 'scheduler'])
        p_bundle = SaveBundle(
            model=self.policy.state_dict(),
            optimizer=self.p_optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
        )
        if not self.hps.shared_value:
            v_bundle = SaveBundle(
                model=self.value.state_dict(),
                optimizer=self.v_optimizer.state_dict(),
                scheduler=None,
            )
        d_bundle = SaveBundle(
            model=self.policy.state_dict(),
            optimizer=self.d_optimizer.state_dict(),
            scheduler=None
        )
        torch.save(p_bundle._asdict(), osp.join(path, "p_iter{}.pth".format(iters)))
        if not self.hps.shared_value:
            torch.save(v_bundle._asdict(), osp.join(path, "v_iter{}.pth".format(iters)))
        torch.save(d_bundle._asdict(), osp.join(path, "d_iter{}.pth".format(iters)))

    def load(self, path, iters):
        p_bundle = torch.load(osp.join(path, "p_iter{}.pth".format(iters)))
        self.policy.load_state_dict(p_bundle['model'])
        self.p_optimizer.load_state_dict(p_bundle['optimizer'])
        self.scheduler.load_state_dict(p_bundle['scheduler'])
        if not self.hps.shared_value:
            v_bundle = torch.load(osp.join(path, "v_iter{}.pth".format(iters)))
            self.value.load_state_dict(v_bundle['model'])
            self.v_optimizer.load_state_dict(v_bundle['optimizer'])
        d_bundle = torch.load(osp.join(path, "d_iter{}.pth".format(iters)))
        self.discriminator.load_state_dict(d_bundle['model'])
        self.d_optimizer.load_state_dict(d_bundle['optimizer'])
