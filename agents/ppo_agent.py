from collections import namedtuple
import os.path as osp

from gym import spaces

import numpy as np
import torch
import torch.nn.utils as U
from torch.utils.data import DataLoader

from helpers import logger
from helpers.dataset import Dataset
from helpers.console_util import log_module_info
from helpers.distributed_util import average_gradients, sync_with_root, RunMoms
from agents.nets import MLPGaussPolicy, MLPCatPolicy, LSTMGaussPolicy, LSTMCatPolicy
from agents.gae import gae


class PPOAgent(object):

    def __init__(self, env, device, hps):
        self.env = env
        self.ob_space = self.env.observation_space
        self.ob_shape = self.ob_space.shape
        self.ac_space = self.env.action_space
        self.ac_shape = self.ac_space.shape
        self.ob_dim = self.ob_shape[-1]  # num dims
        self.is_disc = isinstance(self.ac_space, spaces.Discrete)
        self.ac_dim = self.ac_space.n if self.is_disc else self.ac_shape[-1]

        self.device = device
        self.hps = hps
        self.is_recurrent = self.hps.hidden_state_size is not None

        # Create nets
        Policy = ((MLPCatPolicy if self.is_disc else MLPGaussPolicy)
                  if not self.is_recurrent
                  else (LSTMCatPolicy if self.is_disc else LSTMGaussPolicy))
        self.policy = Policy(self.env, self.hps).to(self.device)
        sync_with_root(self.policy)

        # Set up the optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.hps.p_lr)

        # Set up the learning rate schedule
        lr_lambda = lambda t: max(2e-5 / self.hps.p_lr, 1. - (float(t) / 1e6))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        log_module_info(logger, 'policy(+value)', self.policy)

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

    def predict_recurrent(self, ob, done, state, sample_or_mode):
        # Create tensor from the state (`require_grad=False` by default)
        ob = ob[None] if self.hps.pixels else self.normalize_clip_ob(ob[None])
        ob = torch.FloatTensor(ob).to(self.device)

        done = torch.FloatTensor([done]).to(self.device)
        old_state = torch.FloatTensor([state]).to(self.device)
        # Predict an action
        ac, state = (self.policy.sample(ob, done, old_state)
                     if sample_or_mode
                     else self.policy.mode(ob, done, old_state))
        # Also retrieve the log-probability associated with the picked action
        logp = self.policy.logp(ob, ac, done, old_state)
        # Place on cpu and collapse into one dimension
        v = self.policy.value(ob, done, old_state).cpu().detach().numpy().flatten()
        ac = ac.cpu().detach().numpy().flatten()
        logp = logp.cpu().detach().numpy().flatten()
        state = state.cpu().detach().numpy().flatten()
        return ac, v, logp, state

    def train(self, rollout, timesteps_so_far):
        """Train the agent"""

        # Augment `rollout` with GAE (Generalized Advantage Estimation), which among
        # other things adds the GAE estimate of the MC estimate of the return
        gae(rollout, self.hps.gamma, self.hps.gae_lambda, rew_key='env_rews')

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
                loss = clip_loss + entropy_loss + (0.1 * value_loss)

                # Update parameters
                self.optimizer.zero_grad()
                loss.backward()
                average_gradients(self.policy, self.device)
                gradnorm = U.clip_grad_norm_(self.policy.parameters(), self.hps.clip_norm)
                self.optimizer.step()
                self.scheduler.step(timesteps_so_far)

        # Aggregate the elements to return
        losses = {'policy': clip_loss + entropy_loss,
                  'value': value_loss,
                  # Sub-losses
                  'clip_loss': clip_loss,
                  'entropy_loss': entropy_loss,
                  'kl_approx': kl_approx,
                  'kl_max': kl_max,
                  'clip_frac': clip_frac}
        losses = {k: v.clone().cpu().data.numpy() for k, v in losses.items()}

        return losses, gradnorm

    def train_recurrent(self, rollout, timesteps_so_far):
        """Train the agent"""

        # Ensure we are using a recurrent model
        assert self.is_recurrent

        # Augment `rollout` with GAE (Generalized Advantage Estimation), which among
        # other things adds the GAE estimate of the MC estimate of the return
        gae(rollout, self.hps.gamma, self.hps.gae_lambda, rew_key='env_rews')

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
                                                   'td_lam_rets',
                                                   'states',
                                                   'dones']})
        dataloader = DataLoader(dataset, self.hps.batch_size, shuffle=False)

        for _ in range(self.hps.optim_epochs_per_iter):

            for chunk in dataloader:

                # Create tensors from the inputs
                state = torch.FloatTensor(chunk['obs']).to(self.device)
                action = torch.FloatTensor(chunk['acs']).to(self.device)
                logp_old = torch.FloatTensor(chunk['logps']).to(self.device)
                v_old = torch.FloatTensor(chunk['vs']).to(self.device)
                advantage = torch.FloatTensor(chunk['advs']).to(self.device)
                return_ = torch.FloatTensor(chunk['td_lam_rets']).to(self.device)
                mask = torch.FloatTensor(chunk['dones']).to(self.device)
                hidden_state = torch.FloatTensor(chunk['states']).to(self.device)

                h0 = hidden_state.clone().detach()

                # Policy loss
                entropy_loss = -self.policy.entropy(state, mask, h0).mean()
                entropy_loss *= self.hps.p_ent_reg_scale
                logp = self.policy.logp(state, action, mask, h0)
                ratio = torch.exp(logp - logp_old)
                surrogate_loss_a = -advantage * ratio
                surrogate_loss_b = -advantage * ratio.clamp(1.0 - self.hps.eps, 1.0 + self.hps.eps)
                clip_loss = torch.max(surrogate_loss_a, surrogate_loss_b).mean()
                kl_approx = 0.5 * (logp - logp_old).pow(2).mean()
                kl_max = 0.5 * (logp - logp_old).pow(2).max()
                clip_frac = (ratio - 1.0).abs().gt(self.hps.eps).float().mean()
                policy_loss = clip_loss + entropy_loss
                # Value loss
                v = self.policy.value(state, mask, h0)
                clip_v = v_old + (v - v_old).clamp(-self.hps.eps, self.hps.eps)
                value_loss_a = (clip_v - return_).pow(2)
                value_loss_b = (v - return_).pow(2)
                value_loss = torch.max(value_loss_a, value_loss_b).mean()
                # Aggregated loss
                loss = clip_loss + entropy_loss + (0.1 * value_loss)

                # Update parameters
                self.optimizer.zero_grad()
                loss.backward()
                average_gradients(self.policy, self.device)
                gradnorm = U.clip_grad_norm_(self.policy.parameters(), self.hps.clip_norm)
                self.optimizer.step()
                self.scheduler.step(timesteps_so_far)

        # Aggregate the elements to return
        losses = {'policy': clip_loss + entropy_loss,
                  'value': value_loss,
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
        bundle = SaveBundle(model=self.policy.state_dict(),
                            optimizer=self.optimizer.state_dict(),
                            scheduler=self.scheduler.state_dict())
        torch.save(bundle._asdict(), osp.join(path, "policy_iter{}.pth".format(iters)))

    def load(self, path, iters):
        bundle = torch.load(osp.join(path, "policy_iter{}.pth".format(iters)))
        self.policy.load_state_dict(bundle['model'])
        self.optimizer.load_state_dict(bundle['optimizer'])
        self.scheduler.load_state_dict(bundle['scheduler'])
