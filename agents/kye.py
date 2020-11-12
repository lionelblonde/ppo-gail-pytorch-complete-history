import math
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers import logger
from helpers.console_util import log_module_info
from agents.nets import init
from helpers.distributed_util import RunMoms


STANDARDIZED_OB_CLAMPS = [-5., 5.]


class PredNet(nn.Module):

    def __init__(self, env, hps, rms_obs):
        super(PredNet, self).__init__()
        self.hps = hps
        assert not self.hps.visual, "network not adapted to visual input (for now)"
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        if self.hps.wrap_absorb:
            ob_dim += 1
            ac_dim += 1
        self.leak = 0.1
        if self.hps.kye_batch_norm:
            # Define observation whitening
            self.rms_obs = rms_obs
        # Define the input dimension
        in_dim = ob_dim
        if self.hps.state_only:
            in_dim += ob_dim
        else:
            in_dim += ac_dim
        # Assemble the layers and output heads
        self.fc_stack = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(in_dim, 100)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(100, 100)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
        ]))
        self.head = nn.Linear(100, 1)
        # Perform initialization
        self.fc_stack.apply(init(weight_scale=math.sqrt(2) / math.sqrt(1 + self.leak**2)))
        self.head.apply(init(weight_scale=0.01))

    def forward(self, input_a, input_b):
        if self.hps.kye_batch_norm:
            # Apply normalization
            if self.hps.wrap_absorb:
                # Normalize state
                input_a_ = input_a.clone()[:, 0:-1]
                input_a_ = self.rms_obs.standardize(input_a_).clamp(*STANDARDIZED_OB_CLAMPS)
                input_a = torch.cat([input_a_, input_a[:, -1].unsqueeze(-1)], dim=-1)
                if self.hps.state_only:
                    # Normalize next state
                    input_b_ = input_b.clone()[:, 0:-1]
                    input_b_ = self.rms_obs.standardize(input_b_).clamp(*STANDARDIZED_OB_CLAMPS)
                    input_b = torch.cat([input_b_, input_b[:, -1].unsqueeze(-1)], dim=-1)
            else:
                # Normalize state
                input_a = self.rms_obs.standardize(input_a).clamp(*STANDARDIZED_OB_CLAMPS)
                if self.hps.state_only:
                    # Normalize next state
                    input_b = self.rms_obs.standardize(input_b).clamp(*STANDARDIZED_OB_CLAMPS)
        else:
            input_a = input_a.clamp(*STANDARDIZED_OB_CLAMPS)
            if self.hps.state_only:
                input_b = input_b.clamp(*STANDARDIZED_OB_CLAMPS)
        # Concatenate
        x = torch.cat([input_a, input_b], dim=-1)
        x = self.fc_stack(x)
        x = self.head(x)
        return x


class KnowYourEnemy(object):

    def __init__(self, env, device, hps, rms_obs):
        self.env = env
        self.device = device
        self.hps = hps
        self.rms_obs = rms_obs

        # Create nets
        self.pred_net = PredNet(self.env, self.hps, self.rms_obs).to(self.device)

        # Create optimizer
        self.optimizer = torch.optim.Adam(self.pred_net.parameters(), lr=self.hps.kye_lr)

        # Define reward normalizer
        self.rms_pred_losses = RunMoms(shape=(1,), use_mpi=False)

        log_module_info(logger, 'KYE Pred Network', self.pred_net)

    def remove_absorbing(self, x):
        non_absorbing_rows = []
        for j, row in enumerate([x[i, :] for i in range(x.shape[0])]):
            if torch.all(torch.eq(row, torch.cat([torch.zeros_like(row[0:-1]),
                                                  torch.Tensor([1.]).to(self.device)], dim=-1))):
                # logger.info("removing absorbing row (#{})".format(j))
                pass
            else:
                non_absorbing_rows.append(j)
        return x[non_absorbing_rows, :], non_absorbing_rows

    def update(self, dataloader, disc_score_func):
        """Update the opponent predictor network"""

        # Container for all the metrics
        metrics = defaultdict(list)

        for batch in dataloader:
            logger.info("updating kye predictor")

            # Transfer to device
            state = batch['obs0'].to(self.device)
            next_state = batch['obs1'].to(self.device)
            action = batch['acs'].to(self.device)
            if self.hps.wrap_absorb:
                _, indices = self.remove_absorbing(state)
                state = state[indices, :]
                next_state = next_state[indices, :]
                action = action[indices, :]

            # Define inputs
            input_a = state
            if self.hps.state_only:
                input_b = next_state
            else:
                input_b = action

            # Compute loss
            _loss = F.mse_loss(
                self.pred_net(input_a, input_b),
                disc_score_func(input_a, input_b),
                reduction='none',
            )
            loss = _loss.mean(dim=-1)
            mask = loss.clone().detach().data.uniform_().to(self.device)
            mask = (mask < self.hps.proportion_of_exp_per_kye_update).float()
            loss = (mask * loss).sum() / torch.max(torch.Tensor([1.]), mask.sum())
            metrics['loss'].append(loss)

            # Update running moments
            pred_losses = F.mse_loss(
                self.pred_net(input_a, input_b),
                disc_score_func(input_a, input_b),
                reduction='none',
            ).detach()
            self.rms_pred_losses.update(pred_losses)

            # Update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        metrics = {k: torch.stack(v).mean().cpu().data.numpy() for k, v in metrics.items()}
        return metrics

    def get_int_rew(self, input_a, input_b, disc_score_func):
        # Compute intrinsic reward
        pred_losses = F.mse_loss(
            self.pred_net(input_a, input_b),
            disc_score_func(input_a, input_b),
            reduction='none',
        ).detach()
        # Normalize intrinsic reward
        pred_losses = self.rms_pred_losses.divide_by_std(pred_losses)
        int_rews = F.softplus(-pred_losses)
        return int_rews
