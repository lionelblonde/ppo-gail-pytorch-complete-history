from collections import namedtuple
import os.path as osp

import numpy as np
import torch
import torch.nn.utils as U
import torch.nn.functional as F
from torch.utils.data import DataLoader

from helpers import logger
from helpers.dataset import Dataset
from helpers.console_util import log_module_info
from helpers.distributed_util import average_gradients, sync_with_root, RunMoms
from agents.nets import MLPGaussPolicy, Discriminator
from agents.gae import gae


class GAILAgent(object):

    def __init__(self, env, device, hps, expert_dataset):
        self.env = env
        self.ob_space = self.env.observation_space
        self.ob_shape = self.ob_space.shape
        self.ac_space = self.env.action_space
        self.ac_shape = self.ac_space.shape
        self.ob_dim = self.ob_shape[-1]  # num dims
        self.ac_dim = self.ac_shape[-1]  # num dims

        self.device = device
        self.hps = hps
        self.is_recurrent = False

        # Create nets
        self.policy = MLPGaussPolicy(self.env, self.hps).to(self.device)
        sync_with_root(self.policy)
        self.discriminator = Discriminator(self.env, self.hps).to(self.device)
        sync_with_root(self.discriminator)

        # Set up demonstrations dataset
        self.e_dataloader = DataLoader(expert_dataset, self.hps.batch_size, shuffle=True)

        # Set up the optimizers
        self.p_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.hps.p_lr)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.hps.d_lr)

        # Set up the learning rate schedule
        lr_lambda = lambda t: max(2e-5 / self.hps.p_lr, 1. - (float(t) / 1e6))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.p_optimizer, lr_lambda)

        log_module_info(logger, 'policy(+value)', self.policy)
        log_module_info(logger, 'discriminator', self.discriminator)

        if not self.hps.pixels:
            self.rms_obs = RunMoms(shape=self.ob_shape, use_mpi=True)

    def normalize_clip_ob(self, ob):
        # Normalize with running mean and running std
        ob = ((ob - self.rms_obs.mean) /
              (np.sqrt(self.rms_obs.var) + 1e-8))
        # Clip
        ob = np.clip(ob, -5.0, 5.0)
        return ob

    def predict(self, ob, sample_or_mode):
        # Create tensor from the state (`require_grad=False` by default)
        ob = ob[None] if self.hps.pixels else self.normalize_clip_ob(ob[None])
        ob = torch.FloatTensor(ob).to(self.device)

        # Predict an action
        ac = self.policy.sample(ob) if sample_or_mode else self.policy.mode(ob)
        # Also retrieve the log-probability associated with the picked action
        logp = self.policy.logp(ob, ac)
        # Place on cpu and collapse into one dimension
        v = self.policy.value(ob).cpu().detach().numpy().flatten()
        ac = ac.cpu().detach().numpy().flatten()
        logp = logp.cpu().detach().numpy().flatten()
        return ac, v, logp

    def train(self, rollout, timesteps_so_far):
        """Train the agent"""

        # Augment `rollout` with GAE (Generalized Advantage Estimation), which among
        # other things adds the GAE estimate of the MC estimate of the return
        gae(rollout, self.hps.gamma, self.hps.gae_lambda, rew_key="syn_rews")

        # Standardize advantage function estimate
        rollout['advs'] = ((rollout['advs'] - rollout['advs'].mean()) /
                           (rollout['advs'].std() + 1e-8))

        # Standardize and clip observations
        rollout['obs'] = (rollout['obs'] if self.hps.pixels
                          else self.normalize_clip_ob(rollout['obs']))

        # Create DataLoader object to iterate over transitions in rollouts
        dataset = Dataset({k: rollout[k] for k in ['obs',
                                                   'acs',
                                                   'vs',
                                                   'logps',
                                                   'advs',
                                                   'td_lam_rets']})
        dataloader = DataLoader(dataset, self.hps.batch_size, shuffle=True)

        for _ in range(self.hps.optim_epochs_per_iter):

            for chunk in dataloader:

                # Create tensors from the inputs
                state = torch.FloatTensor(chunk['obs']).to(self.device)
                action = torch.FloatTensor(chunk['acs']).to(self.device)
                logp_old = torch.FloatTensor(chunk['logps']).to(self.device)
                v_old = torch.FloatTensor(chunk['vs']).to(self.device)
                advantage = torch.FloatTensor(chunk['advs']).to(self.device)
                return_ = torch.FloatTensor(chunk['td_lam_rets']).to(self.device)

                # Policy loss
                entropy_loss = -self.hps.p_ent_reg_scale * self.policy.entropy(state).mean()
                logp = self.policy.logp(state, action)
                ratio = torch.exp(logp - logp_old)
                surrogate_loss_a = -advantage * ratio
                surrogate_loss_b = -advantage * ratio.clamp(1.0 - self.hps.eps, 1.0 + self.hps.eps)
                clip_loss = torch.max(surrogate_loss_a, surrogate_loss_b).mean()
                kl_approx = 0.5 * (logp - logp_old).pow(2).mean()
                kl_max = 0.5 * (logp - logp_old).pow(2).max()
                clip_frac = (ratio - 1.0).abs().gt(self.hps.eps).float().mean()
                # Value loss
                v = self.policy.value(state)
                clip_v = v_old + (v - v_old).clamp(-self.hps.eps, self.hps.eps)
                value_loss_a = (clip_v - return_).pow(2)
                value_loss_b = (v - return_).pow(2)
                value_loss = torch.max(value_loss_a, value_loss_b).mean()
                # Aggregated loss
                p_loss = clip_loss + entropy_loss + (0.1 * value_loss)

                # Update parameters
                self.p_optimizer.zero_grad()
                p_loss.backward()
                average_gradients(self.policy, self.device)
                gradnorm = U.clip_grad_norm_(self.policy.parameters(), self.hps.clip_norm)
                self.p_optimizer.step()
                self.scheduler.step(timesteps_so_far)

            for _ in range(self.hps.d_update_ratio):

                for chunk, e_chunk in zip(dataloader, self.e_dataloader):

                    # Create tensors from the inputs
                    state = torch.FloatTensor(chunk['obs']).to(self.device)
                    action = torch.FloatTensor(chunk['acs']).to(self.device)
                    # Get expert data and create tensors from the inputs
                    e_obs, e_acs = e_chunk['obs0'], e_chunk['acs']
                    e_state = torch.FloatTensor(e_obs).to(self.device)
                    e_action = torch.FloatTensor(e_acs).to(self.device)
                    # Compute scores
                    p_scores = self.discriminator(state, action)
                    e_scores = self.discriminator(e_state, e_action)
                    # Create entropy loss
                    scores = torch.cat([p_scores, e_scores], dim=0)
                    entropy = F.binary_cross_entropy_with_logits(input=scores,
                                                                 target=F.sigmoid(scores))
                    entropy_loss = -self.hps.d_ent_reg_scale * entropy
                    # Create labels
                    fake_labels = torch.zeros_like(p_scores).to(self.device)
                    real_labels = torch.ones_like(e_scores).to(self.device)
                    # Label smoothing, suggested in 'Improved Techniques for Training GANs',
                    # Salimans 2016, https://arxiv.org/abs/1606.03498
                    # The paper advises on the use of one-sided label smoothing, i.e.
                    # only smooth out the positive (real) targets side.
                    # Extra comment explanation: https://github.com/openai/improved-gan/blob/
                    # 9ff96a7e9e5ac4346796985ddbb9af3239c6eed1/imagenet/build_model.py#L88-L121
                    # Additional material: https://github.com/soumith/ganhacks/issues/10
                    real_labels.uniform_(0.7, 1.2)
                    # Create binary classification (cross-entropy) losses
                    p_loss = F.binary_cross_entropy_with_logits(input=p_scores, target=fake_labels)
                    e_loss = F.binary_cross_entropy_with_logits(input=e_scores, target=real_labels)
                    # Aggregated loss
                    d_loss = p_loss + e_loss + entropy_loss

                    # Update parameters
                    self.d_optimizer.zero_grad()
                    d_loss.backward()
                    U.clip_grad_norm_(self.discriminator.parameters(), self.hps.clip_norm)
                    average_gradients(self.discriminator, self.device)
                    self.d_optimizer.step()

        # Aggregate the elements to return
        losses = {'policy': clip_loss + entropy_loss,
                  'value': value_loss,
                  'discriminator': d_loss,
                  # Sub-losses
                  'clip_loss': clip_loss,
                  'entropy_loss': entropy_loss,
                  'kl_approx': kl_approx,
                  'kl_max': kl_max,
                  'clip_frac': clip_frac}
        losses = {k: v.clone().cpu().data.numpy() for k, v in losses.items()}

        return losses, gradnorm

    def save(self, path, iters):
        SaveBundle = namedtuple('SaveBundle', ['model', 'optimizer', 'scheduler'])
        p_bundle = SaveBundle(model=self.policy.state_dict(),
                              optimizer=self.p_optimizer.state_dict(),
                              scheduler=self.scheduler.state_dict())
        d_bundle = SaveBundle(model=self.policy.state_dict(),
                              optimizer=self.d_optimizer.state_dict(),
                              scheduler=None)
        torch.save(p_bundle._asdict(), osp.join(path, "p_iter{}.pth".format(iters)))
        torch.save(d_bundle._asdict(), osp.join(path, "d_iter{}.pth".format(iters)))

    def load(self, path, iters):
        p_bundle = torch.load(osp.join(path, "p_iter{}.pth".format(iters)))
        self.policy.load_state_dict(p_bundle['model'])
        self.p_optimizer.load_state_dict(p_bundle['optimizer'])
        self.scheduler.load_state_dict(p_bundle['scheduler'])
        d_bundle = torch.load(osp.join(path, "d_iter{}.pth".format(iters)))
        self.discriminator.load_state_dict(d_bundle['model'])
        self.d_optimizer.load_state_dict(d_bundle['optimizer'])
