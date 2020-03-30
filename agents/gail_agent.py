from collections import namedtuple, defaultdict
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
from agents.nets import GaussPolicy, Value, Discriminator
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
        self.discriminator = Discriminator(self.env, self.hps).to(self.device)
        sync_with_root(self.discriminator)

        # Set up demonstrations dataset
        self.e_batch_size = min(len(self.expert_dataset), self.hps.batch_size)
        self.e_dataloader = DataLoader(
            self.expert_dataset,
            self.e_batch_size,
            shuffle=True,
            drop_last=True,
        )
        assert len(self.e_dataloader) > 0

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
            log_module_info(logger, 'value', self.value)
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

    def remove_absorbing(self, x):
        non_absorbing_rows = []
        for j, row in enumerate([x[i, :] for i in range(x.shape[0])]):
            if torch.all(torch.eq(row, torch.cat([torch.zeros_like(row[0:-1]),
                                                  torch.Tensor([1.]).to(self.device)], dim=-1))):
                # logger.info("[INFO] removing absorbing row (#{})".format(j))
                pass
            else:
                non_absorbing_rows.append(j)
        return x[non_absorbing_rows, :], non_absorbing_rows

    def update_policy_value(self, rollout, iters_so_far):
        """Update the policy and value networks"""

        # Container for all the metrics
        metrics = defaultdict(list)

        # Augment `rollout` with GAE (Generalized Advantage Estimation), which among
        # other things adds the GAE estimate of the MC estimate of the return
        gae(rollout, self.hps.gamma, self.hps.gae_lambda, rew_key='syn_rews')

        # Standardize advantage function estimate
        rollout['advs'] = ((rollout['advs'] - rollout['advs'].mean()) /
                           (rollout['advs'].std() + 1e-8))

        # Create DataLoader objects to iterate over transitions in rollouts
        keys = ['obs0', 'obs1', 'acs', 'logps', 'vs', 'advs', 'td_lam_rets']
        if self.hps.wrap_absorb:
            keys.extend(['obs0_orig', 'obs1_orig', 'acs_orig'])
        dataset = Dataset({k: rollout[k] for k in keys})
        dataloader = DataLoader(
            dataset,
            self.hps.batch_size,
            shuffle=True,
            drop_last=False,  # no compatibility issue, only used for policy alone
        )

        for _ in range(self.hps.optim_epochs_per_iter):

            for batch in dataloader:  # go through the whole p_rollout
                logger.info("[INFO] updating policy")

                # Transfer to device
                if self.hps.wrap_absorb:
                    state = batch['obs0_orig'].to(self.device)
                    action = batch['acs_orig'].to(self.device)
                else:
                    state = batch['obs0'].to(self.device)
                    action = batch['acs'].to(self.device)
                logp_old = batch['logps'].to(self.device)
                v_old = batch['vs'].to(self.device)
                advantage = batch['advs'].to(self.device)
                td_lam_return = batch['td_lam_rets'].to(self.device)

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
                _clip_loss = torch.max(surrogate_loss_a, surrogate_loss_b)
                clip_loss = _clip_loss.mean()
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
                _v_loss = torch.max(v_loss_a, v_loss_b)
                v_loss = _v_loss.mean()
                if self.hps.shared_value:
                    p_loss = clip_loss + entropy_loss + (self.hps.baseline_scale * v_loss)
                    _p_loss = _clip_loss + (self.hps.baseline_scale * _v_loss)
                else:
                    p_loss = clip_loss + entropy_loss
                    _p_loss = _clip_loss

                # Log metrics
                metrics['entropy_loss'].append(entropy_loss)
                metrics['clip_loss'].append(clip_loss)
                metrics['kl_approx'].append(kl_approx)
                metrics['kl_max'].append(kl_max)
                metrics['clip_frac'].append(clip_frac)
                metrics['v_loss'].append(v_loss)
                metrics['p_loss'].append(p_loss)

                if self.hps.kye_p:

                    # Sub-optimal potential re-definitions here
                    # but easier to debut and with little over-head
                    if self.hps.wrap_absorb:
                        _state = batch['obs0_orig'].to(self.device)
                    else:
                        _state = batch['obs0'].to(self.device)
                    state = batch['obs0'].to(self.device)
                    next_state = batch['obs1'].to(self.device)
                    action = batch['acs'].to(self.device)
                    if self.hps.wrap_absorb:
                        _, indices = self.remove_absorbing(state)
                        _state = _state[indices, :]
                        state = state[indices, :]
                        next_state = next_state[indices, :]
                        action = action[indices, :]
                    _aux_loss = F.smooth_l1_loss(
                        input=self.policy.auxo(_state),
                        target=self.get_reward(state, action, next_state),
                        reduction='none',
                    )

                    # Init collections of gradients
                    grads_a_list = []
                    grads_b_list = []
                    for i in range(_state.size(0)):
                        # Compute the gradients of the shared weights for the main task
                        inputs = self.policy.perception_stack.encoder_1.fc_block.fc.weight
                        grads_a = autograd.grad(outputs=[_p_loss[i, ...]],
                                                inputs=[inputs],
                                                only_inputs=True,
                                                grad_outputs=[torch.ones_like(_p_loss[i, ...])],
                                                retain_graph=True,
                                                create_graph=True,
                                                allow_unused=True)
                        # Compute the gradients of the shared weights for the auxiliary task
                        grads_b = autograd.grad(outputs=[_aux_loss[i, ...]],
                                                inputs=[inputs],
                                                only_inputs=True,
                                                grad_outputs=[torch.ones_like(_aux_loss[i, ...])],
                                                retain_graph=True,
                                                create_graph=True,
                                                allow_unused=True)
                        grads_a_list.append(grads_a[0])
                        grads_b_list.append(grads_b[0])
                    grads_a = torch.stack(grads_a_list, dim=0).sum(dim=-1)
                    grads_b = torch.stack(grads_b_list, dim=0).sum(dim=-1)
                    cos_sims = F.cosine_similarity(grads_a, grads_b).unsqueeze(-1)
                    cos_sims = cos_sims.detach()  # safety measure
                    metrics['cos_sim'].append(cos_sims.mean())

                    # Assemble losses

                    assert _aux_loss.shape == cos_sims.shape, "shape mismatch"
                    _aux_loss *= torch.max(torch.zeros_like(cos_sims), cos_sims)
                    _aux_loss *= self.hps.kye_p_scale

                    aux_loss = _aux_loss.mean()
                    metrics['aux_loss'].append(aux_loss)

                    if self.hps.kye_mixing:
                        # Add mixing auxiliary loss
                        e_batch = next(iter(self.e_dataloader))  # get a minibatch of expert data
                        if self.hps.wrap_absorb:
                            _state_e = e_batch['obs0_orig']
                        else:
                            _state_e = e_batch['obs0']
                        state_e = e_batch['obs0']
                        next_state_e = e_batch['obs1']
                        action_e = e_batch['acs']
                        if self.hps.wrap_absorb:
                            _, indices = self.remove_absorbing(state_e)
                            _state_e = _state_e[indices, :]
                            state_e = state_e[indices, :]
                            next_state_e = next_state_e[indices, :]
                            action_e = action_e[indices, :]
                        aux_loss += F.smooth_l1_loss(
                            input=self.policy.auxo(_state_e),
                            target=self.get_reward(state_e, action_e, next_state_e),
                        )

                    p_loss += aux_loss

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

        metrics = {k: torch.stack(v).mean().cpu().data.numpy() for k, v in metrics.items()}
        return metrics, self.scheduler.get_last_lr()

    def update_discriminator(self, rollout, iters_so_far):
        """Update the discriminator network"""

        # Container for all the metrics
        metrics = defaultdict(list)

        # Create DataLoader object to iterate over transitions in rollouts
        d_keys = ['obs0']
        if self.hps.state_only:
            d_keys.append('obs1')
        else:
            d_keys.append('acs')

        d_dataset = Dataset({k: rollout[k] for k in d_keys})
        d_dataloader = DataLoader(
            d_dataset,
            self.e_batch_size,
            shuffle=True,
            drop_last=True,
        )

        for e_batch in self.e_dataloader:
            logger.info("[INFO] updating discriminator")

            # Get a minibatch of policy data
            d_batch = next(iter(d_dataloader))

            # Transfer to device
            p_input_a = d_batch['obs0'].to(self.device)
            e_input_a = e_batch['obs0'].to(self.device)
            if self.hps.state_only:
                p_input_b = d_batch['obs1'].to(self.device)
                e_input_b = e_batch['obs1'].to(self.device)
            else:
                p_input_b = d_batch['acs'].to(self.device)
                e_input_b = e_batch['acs'].to(self.device)

            # Update running moments
            _state = torch.cat([p_input_a, e_input_a], dim=0)
            if self.hps.wrap_absorb:
                _state = self.remove_absorbing(_state)[0][:, 0:-1]
            self.discriminator.rms_obs.update(_state)

            # Compute scores
            p_scores = self.discriminator.D(p_input_a, p_input_b)
            e_scores = self.discriminator.D(e_input_a, e_input_b)

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
                _p_e_loss = p_loss + e_loss
            # Averate out over the batch
            p_e_loss = _p_e_loss.mean()

            # Aggregated loss
            d_loss = p_e_loss + entropy_loss

            # Log metrics
            metrics['entropy_loss'].append(entropy_loss)
            metrics['p_e_loss'].append(p_e_loss)
            metrics['d_loss'].append(d_loss)

            if self.hps.grad_pen:
                # Create gradient penalty loss (coefficient from the original paper)
                grad_pen = 10. * self.grad_pen(p_input_a, p_input_b, e_input_a, e_input_b)
                d_loss += grad_pen
                # Log metrics
                metrics['grad_pen'].append(grad_pen)

            if self.hps.kye_d:

                # Create and add auxiliary loss
                if self.hps.wrap_absorb:
                    _, p_indices = self.remove_absorbing(p_input_a)
                    _p_input_a = p_input_a[p_indices, 0:-1]
                    p_input_a = p_input_a[p_indices, :]
                    p_input_b = p_input_b[p_indices, :]
                else:
                    _p_input_a = p_input_a
                _aux_loss = F.smooth_l1_loss(
                    input=self.discriminator.auxo(p_input_a, p_input_b),
                    target=self.policy.sample(_p_input_a),
                    reduction='none',
                )
                _aux_loss = _aux_loss.mean(dim=-1, keepdim=True)

                # Init collections of gradients
                grads_a_list = []
                grads_b_list = []
                for i in range(_p_input_a.size(0)):
                    # Compute the gradients of the shared weights for the main task
                    if self.hps.spectral_norm:
                        inputs = self.discriminator.d_encoder.fc_block.fc.weight_orig
                    else:
                        inputs = self.discriminator.d_encoder.fc_block.fc.weight
                    grads_a = autograd.grad(outputs=[_p_e_loss[i, ...]],  # without entropy loss
                                            inputs=[inputs],
                                            only_inputs=True,
                                            grad_outputs=[torch.ones_like(_p_e_loss[i, ...])],
                                            retain_graph=True,
                                            create_graph=True,
                                            allow_unused=True)
                    # Compute the gradients of the shared weights for the auxiliary task
                    grads_b = autograd.grad(outputs=[_aux_loss[i, ...]],
                                            inputs=[inputs],
                                            only_inputs=True,
                                            grad_outputs=[torch.ones_like(_aux_loss[i, ...])],
                                            retain_graph=True,
                                            create_graph=True,
                                            allow_unused=True)
                    grads_a_list.append(grads_a[0])
                    grads_b_list.append(grads_b[0])
                grads_a = torch.stack(grads_a_list, dim=0).sum(dim=-1)
                grads_b = torch.stack(grads_b_list, dim=0).sum(dim=-1)
                cos_sims = F.cosine_similarity(grads_a, grads_b).unsqueeze(-1)
                cos_sims = cos_sims.detach()  # safety measure
                metrics['cos_sim'].append(cos_sims.mean())

                # Assemble losses

                assert _aux_loss.shape == cos_sims.shape, "shape mismatch"
                _aux_loss *= torch.max(torch.zeros_like(cos_sims), cos_sims)
                _aux_loss *= self.hps.kye_d_scale

                aux_loss = _aux_loss.mean()
                metrics['aux_loss'].append(aux_loss)

                if self.hps.kye_mixing:
                    # Add mixing auxiliary loss
                    if self.hps.wrap_absorb:
                        _, e_indices = self.remove_absorbing(e_input_a)
                        _e_input_a = e_input_a[e_indices, 0:-1]
                        e_input_a = e_input_a[e_indices, :]
                        e_input_b = e_input_b[e_indices, :]
                    else:
                        _e_input_a = e_input_a
                    aux_loss += F.smooth_l1_loss(
                        input=self.discriminator.auxo(e_input_a, e_input_b),
                        target=self.policy.sample(_e_input_a),
                    )

                d_loss += aux_loss

            # Update parameters
            self.d_optimizer.zero_grad()
            d_loss.backward()
            average_gradients(self.discriminator, self.device)
            self.d_optimizer.step()

        metrics = {k: torch.stack(v).mean().cpu().data.numpy() for k, v in metrics.items()}
        return metrics

    def grad_pen(self, p_input_a, p_input_b, e_input_a, e_input_b):
        """Define the gradient penalty regularizer"""
        # Assemble interpolated inputs
        eps_a = torch.rand_like(p_input_a)  # default device is input device
        eps_b = torch.rand_like(p_input_b)  # default device is input device
        input_a_i = eps_a * p_input_a + ((1. - eps_a) * e_input_a)
        input_b_i = eps_b * p_input_b + ((1. - eps_b) * e_input_b)
        # Set `requires_grad=True` to later have access to
        # gradients w.r.t. the inputs (not populated by default)
        input_a_i = Variable(input_a_i, requires_grad=True)
        input_b_i = Variable(input_b_i, requires_grad=True)
        # Create the operation of interest
        score = self.discriminator.D(input_a_i, input_b_i)
        # Get the gradient of this operation with respect to its inputs
        grads = autograd.grad(outputs=score,
                              inputs=[input_a_i, input_b_i],
                              only_inputs=True,
                              grad_outputs=torch.ones(score.size()).to(self.device),
                              retain_graph=True,
                              create_graph=True,
                              allow_unused=self.hps.state_only)
        assert len(list(grads)) == 2, "length must be exactly 2"
        # Return the gradient penalty (try to induce 1-Lipschitzness)
        grads_concat = torch.cat(list(grads), dim=-1)
        return (grads_concat.norm(2, dim=-1) - 1.).pow(2).mean()

    def get_reward(self, state, action, next_state):
        # Define the discriminator inputs
        input_a = state
        if self.hps.state_only:
            input_b = next_state
        else:
            input_b = action
        # Craft surrogate reward
        assert sum([isinstance(x, torch.Tensor) for x in [input_a, input_b]]) in [0, 2]
        if not isinstance(input_a, torch.Tensor):  # then the other is not neither
            input_a = torch.Tensor(input_a)
            input_b = torch.Tensor(input_b)
        # Transfer to cpu
        input_a = input_a.cpu()
        input_b = input_b.cpu()
        # Compure score
        score = self.discriminator.D(input_a, input_b).detach().view(-1, 1)
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
        return self.hps.syn_rew_scale * reward

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
