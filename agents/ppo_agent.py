from collections import namedtuple, defaultdict
import os.path as osp

from gym import spaces

import torch
import torch.nn.utils as U
from torch.utils.data import DataLoader

from helpers import logger
from helpers.dataset import Dataset
from helpers.console_util import log_env_info, log_module_info
from helpers.distributed_util import average_gradients, sync_with_root
from agents.nets import GaussPolicy, Value, CatPolicy
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
        if self.hps.clip_norm <= 0:
            logger.info("[WARN] clip_norm={} <= 0, hence disabled.".format(self.hps.clip_norm))

        # Create nets
        Policy = CatPolicy if self.is_discrete else GaussPolicy
        self.policy = Policy(self.env, self.hps).to(self.device)
        sync_with_root(self.policy)
        if not self.hps.shared_value:
            self.value = Value(self.env, self.hps).to(self.device)
            sync_with_root(self.value)

        # Set up the optimizer
        self.p_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.hps.p_lr)
        if not self.hps.shared_value:
            self.v_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.hps.v_lr)

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

    def update_policy_value(self, rollout, iters_so_far):
        """Train the agent"""

        # Container for all the metrics
        metrics = defaultdict(list)

        # Augment `rollout` with GAE (Generalized Advantage Estimation), which among
        # other things adds the GAE estimate of the MC estimate of the return
        gae(rollout, self.hps.gamma, self.hps.gae_lambda, rew_key='env_rews')

        # Standardize advantage function estimate
        rollout['advs'] = ((rollout['advs'] - rollout['advs'].mean()) /
                           (rollout['advs'].std() + 1e-8))

        # Create DataLoader object to iterate over transitions in rollouts
        keys = ['obs0', 'acs', 'logps', 'vs', 'advs', 'td_lam_rets']
        dataset = Dataset({k: rollout[k] for k in keys})
        dataloader = DataLoader(
            dataset,
            self.hps.batch_size,
            shuffle=True,
            drop_last=False,  # no compatibility issue, only used for policy alone
        )

        for _ in range(self.hps.optim_epochs_per_iter):

            for chunk in dataloader:

                # Transfer to device
                state = chunk['obs0'].to(self.device)
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

                # Log metrics
                metrics['entropy_loss'].append(entropy_loss)
                metrics['clip_loss'].append(clip_loss)
                metrics['kl_approx'].append(kl_approx)
                metrics['kl_max'].append(kl_max)
                metrics['clip_frac'].append(clip_frac)
                metrics['v_loss'].append(v_loss)
                metrics['p_loss'].append(p_loss)

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
        torch.save(p_bundle._asdict(), osp.join(path, "p_iter{}.pth".format(iters)))
        if not self.hps.shared_value:
            torch.save(v_bundle._asdict(), osp.join(path, "v_iter{}.pth".format(iters)))

    def load(self, path, iters):
        p_bundle = torch.load(osp.join(path, "p_iter{}.pth".format(iters)))
        self.policy.load_state_dict(p_bundle['model'])
        self.p_optimizer.load_state_dict(p_bundle['optimizer'])
        self.scheduler.load_state_dict(p_bundle['scheduler'])
        if not self.hps.shared_value:
            v_bundle = torch.load(osp.join(path, "v_iter{}.pth".format(iters)))
            self.value.load_state_dict(v_bundle['model'])
            self.v_optimizer.load_state_dict(v_bundle['optimizer'])
