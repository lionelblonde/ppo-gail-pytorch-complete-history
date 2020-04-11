from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from helpers import logger
from helpers.console_util import log_module_info
from agents.nets import PredNet, TargNet


class RandomNetworkDistillation(object):

    def __init__(self, env, device, hps, expert_dataset):
        self.env = env
        self.device = device
        self.hps = hps
        self.expert_dataset = expert_dataset

        # Create nets
        self.pred_net = PredNet(self.env, self.hps).to(self.device)
        self.targ_net = TargNet(self.env, self.hps).to(self.device)  # fixed, not trained

        # Set up demonstrations dataset
        self.e_batch_size = min(len(self.expert_dataset), self.hps.batch_size)
        self.e_dataloader = DataLoader(
            self.expert_dataset,
            self.e_batch_size,
            shuffle=True,
            drop_last=True,
        )
        assert len(self.e_dataloader) > 0

        self.optimizer = torch.optim.Adam(self.pred_net.parameters(), lr=5e-4)  # XXX

        log_module_info(logger, 'RND Pred Network', self.pred_net)
        log_module_info(logger, 'RND Targ Network', self.targ_net)

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

    def train(self):
        """Update the RND predictor network"""

        # Container for all the metrics
        metrics = defaultdict(list)

        for e_batch in self.e_dataloader:

            # Transfer to device
            e_input_a = e_batch['obs0'].to(self.device)
            if self.hps.state_only:
                e_input_b = e_batch['obs1'].to(self.device)
            else:
                e_input_b = e_batch['acs'].to(self.device)

            if self.hps.rnd_batch_norm:
                # Update running moments
                _e_input_a = e_input_a.clone()
                if self.hps.wrap_absorb:
                    _e_input_a = self.remove_absorbing(_e_input_a)[0][:, 0:-1]
                self.pred_net.rms_obs.update(_e_input_a)
                self.targ_net.rms_obs.update(_e_input_a)

            # Compute scores
            loss = F.smooth_l1_loss(
                self.pred_net(e_input_a, e_input_b),
                self.targ_net(e_input_a, e_input_b),
            )
            metrics['loss'].append(loss)

            # Update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        metrics = {k: torch.stack(v).mean().cpu().data.numpy() for k, v in metrics.items()}
        return metrics
