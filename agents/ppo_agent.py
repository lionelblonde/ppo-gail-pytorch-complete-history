from collections import namedtuple
import os.path as osp

from gym import spaces

import numpy as np
import torch
import torch.nn.utils as U
from torch.utils.data import DataLoader

from helpers import logger
from helpers.dataset import Dataset
from helpers.console_util import log_env_info, log_module_info
from helpers.distributed_util import average_gradients, sync_with_root, RunMoms
from agents.nets import GaussPolicy, CatPolicy
from agents.gae import gae


class PPOAgent(object):

    def __init__(self, env, device, hps):
        self.env = env

        self.ob_space = self.env.observation_space
        self.ob_shape = self.ob_space.shape
        self.ac_space = self.env.action_space
        self.ac_shape = self.ac_space.shape

        log_env_info(logger, self.env)

        self.ob_dim = self.ob_shape[-1]  # num dims
        self.is_discrete = isinstance(self.ac_space, spaces.Discrete)
        self.ac_dim = self.ac_space.n if self.is_discrete else self.ac_shape[-1]
        self.device = device
        self.hps = hps
        assert not self.hps.binned_aux_loss and not self.hps.squared_aux_loss
        if self.hps.clip_norm <= 0:
            logger.info("[WARN] clip_norm={} <= 0, hence disabled.".format(self.hps.clip_norm))

        # Create nets
        Policy = CatPolicy if self.is_discrete else GaussPolicy
        self.policy = Policy(self.env, self.hps).to(self.device)
        sync_with_root(self.policy)

        # Set up the optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.hps.p_lr)

        # Set up the learning rate schedule
        def _lr(t):  # flake8: using a def instead of a lambda
            if self.hps.with_scheduler:
                return (1.0 - ((t - 1.0) / (self.hps.num_timesteps //
                                            self.hps.rollout_len)))
            else:
                return 1.0

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, _lr)

        log_module_info(logger, 'policy(value)', self.policy)

        if self.hps.norm_obs:
            self.rms_obs = RunMoms(shape=self.ob_shape, use_mpi=True)

    def norm(self, ob):
        # Normalize with running mean and running std
        if torch.is_tensor(ob):
            ob = ((ob - torch.FloatTensor(self.rms_obs.mean)) /
                  (torch.FloatTensor(np.sqrt(self.rms_obs.var)) + 1e-12))
        else:
            ob = ((ob - self.rms_obs.mean) /
                  (np.sqrt(self.rms_obs.var) + 1e-12))
        return ob

    def clip(self, ob, lb=-5.0, ub=5.0):
        # Clip to remain within a certain range
        if torch.is_tensor(ob):
            ob = torch.clamp(ob, lb, ub)
        else:
            ob = np.clip(ob, lb, ub)
        return ob

    def predict(self, ob, sample_or_mode):
        # Create tensor from the state (`require_grad=False` by default)
        ob = ob[None]
        if self.hps.norm_obs:
            ob = self.norm(ob)
        if self.hps.clip_obs:
            ob = self.clip(ob)
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

    def train(self, rollout, iters_so_far):
        """Train the agent"""

        # Augment `rollout` with GAE (Generalized Advantage Estimation), which among
        # other things adds the GAE estimate of the MC estimate of the return
        gae(rollout, self.hps.gamma, self.hps.gae_lambda, rew_key='rews')

        # Standardize advantage function estimate
        rollout['advs'] = ((rollout['advs'] - rollout['advs'].mean()) /
                           (rollout['advs'].std() + 1e-12))

        # Standardize and clip observations
        if self.hps.norm_obs:
            rollout['obs0'] = self.norm(rollout['obs0'])
        if self.hps.clip_obs:
            rollout['obs0'] = self.clip(rollout['obs0'])

        # Create DataLoader object to iterate over transitions in rollouts
        dataset = Dataset({k: rollout[k] for k in ['obs0',
                                                   'acs',
                                                   'vs',
                                                   'logps',
                                                   'advs',
                                                   'td_lam_rets']})
        dataloader = DataLoader(dataset, self.hps.batch_size, shuffle=True)

        for _ in range(self.hps.optim_epochs_per_iter):

            for chunk in dataloader:

                # Create tensors from the inputs
                state = torch.FloatTensor(chunk['obs0']).to(self.device)
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
                surrogate_loss_b = -advantage * ratio.clamp(1.0 - self.hps.eps,
                                                            1.0 + self.hps.eps)
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
                loss = clip_loss + entropy_loss + (self.hps.baseline_scale * value_loss)

                # Update parameters
                self.optimizer.zero_grad()
                loss.backward()
                average_gradients(self.policy, self.device)
                if self.hps.clip_norm > 0:
                    U.clip_grad_norm_(self.policy.parameters(), self.hps.clip_norm)
                self.optimizer.step()
                self.scheduler.step(iters_so_far)

        # Aggregate the elements to return
        losses = {'pol': clip_loss + entropy_loss,
                  'val': value_loss,
                  # Sub-losses
                  'clip_loss': clip_loss,
                  'entropy_loss': entropy_loss,
                  'kl_approx': kl_approx,
                  'kl_max': kl_max,
                  'clip_frac': clip_frac}
        losses = {k: v.clone().cpu().data.numpy() for k, v in losses.items()}

        return losses, self.scheduler.get_last_lr()

    def save(self, path, iters):
        SaveBundle = namedtuple('SaveBundle', ['model', 'optimizer', 'scheduler'])
        bundle = SaveBundle(
            model=self.policy.state_dict(),
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
        )
        torch.save(bundle._asdict(), osp.join(path, "policy_iter{}.pth".format(iters)))

    def load(self, path, iters):
        bundle = torch.load(osp.join(path, "policy_iter{}.pth".format(iters)))
        self.optimizer.load_state_dict(bundle['optimizer'])
        self.scheduler.load_state_dict(bundle['scheduler'])
