import math
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Core.

def nhwc_to_nchw(x):
    """Permute dimensions to go from NHWC to NCHW format"""
    return x.permute(0, 3, 1, 2)


def nchw_to_nhwc(x):
    """Permute dimensions to go from NCHW to NHWC format"""
    return x.permute(0, 2, 3, 1)


def conv_to_fc(x, in_width, in_height):
    assert(isinstance(x, nn.Sequential) or
           isinstance(x, nn.Module))
    specs = [[list(i) for i in [mod.kernel_size,
              mod.stride,
              mod.padding]]
             for mod in x.modules()
             if isinstance(mod, nn.Conv2d)]
    acc = [deepcopy(in_width),
           deepcopy(in_height)]
    for e in specs:
        for i, (k, s, p) in enumerate(zip(*e)):
            acc[i] = ((acc[i] - k + (2 * p)) // s) + 1
    return acc[0] * acc[1]


def init(nonlin=None, param=None,
         weight_scale=1., constant_bias=0.):
    """Perform orthogonal initialization"""

    def _init(m):

        if (isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)):
            scale = (nn.init.calculate_gain(nonlin, param)
                     if nonlin is not None
                     else weight_scale)
            nn.init.orthogonal_(m.weight, gain=scale)
            if m.bias is not None:
                nn.init.constant_(m.bias, constant_bias)
        elif (isinstance(m, nn.BatchNorm2d) or
              isinstance(m, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    return _init


class ResBlock(nn.Module):

    def __init__(self, chan):
        """Residual block from the IMPALA paper ("bottleneck block" from ResNet)
        (https://arxiv.org/abs/1802.01561)
        """
        super(ResBlock, self).__init__()
        # Assemble the impala residual block
        self.residual_block = nn.Sequential(OrderedDict([
            ('pre_conv2d_block_1', nn.Sequential(OrderedDict([  # preactivated
                ('nl', nn.ReLU()),
                ('conv2d', nn.Conv2d(chan, chan, kernel_size=3, stride=1, padding=1)),
            ]))),
            ('pre_conv2d_block_2', nn.Sequential(OrderedDict([  # preactivated
                ('nl', nn.ReLU()),
                ('conv2d', nn.Conv2d(chan, chan, kernel_size=3, stride=1, padding=1)),
            ]))),
        ]))
        self.skip_co = nn.Sequential()  # insert activation, downsampling, etc. if needed
        # Perform initialization
        self.residual_block.apply(init(nonlin='relu', param=None))

    def forward(self, x):
        return self.residual_block(x) + self.skip_co(x)


class ImpalaBlock(nn.Module):

    def __init__(self, in_chan, out_chan):
        """Meta-block from the IMPALA paper
        https://arxiv.org/abs/1802.01561
        """
        super(ImpalaBlock, self).__init__()
        self.impala_block = nn.Sequential(OrderedDict([
            ('conv2d', nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=0)),
            ('maxpool2d', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
            ('res_block_1', ResBlock(out_chan)),
            ('res_block_2', ResBlock(out_chan)),
        ]))
        # Initialize the 'conv2d' layer
        # The residual blocks have already been initialized
        self.impala_block.conv2d.apply(init(nonlin='relu', param=None))

    def forward(self, x):
        return self.impala_block(x)


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Distributional toolkits.

class NormalToolkit(object):

    @staticmethod
    def logp(x, mean, std):
        neglogp = (0.5 * ((x - mean) / std).pow(2).sum(dim=-1, keepdim=True) +
                   0.5 * math.log(2 * math.pi) +
                   std.log().sum(dim=-1, keepdim=True))
        return -neglogp

    @staticmethod
    def entropy(std):
        return (std.log() + 0.5 * math.log(2.0 * math.pi * math.e)).sum(dim=-1, keepdim=True)

    @staticmethod
    def sample(mean, std):
        # Reparametrization trick
        eps = torch.empty(mean.size()).normal_().to(mean.device)
        return mean + std * eps

    @staticmethod
    def mode(mean):
        return mean

    @staticmethod
    def kl(mean, std, other_mean, other_std):
        return (other_std.log() -
                std.log() +
                (std.pow(2) +
                 (mean - other_mean).pow(2)) / (2.0 * other_std.pow(2)) -
                0.5).sum(dim=-1, keepdim=True)


class CatToolkit(object):

    @staticmethod
    def logp(x, logits):
        x = x[None] if len(x.size()) == 1 else x
        eye = torch.eye(logits.size()[-1]).to(logits.device)
        one_hot_ac = F.embedding(input=x.long(),
                                 weight=eye).to(logits.device)
        # Softmax loss (or Softmax Cross-Entropy loss)
        neglogp = -(one_hot_ac[:, 0, :].detach() *
                    F.log_softmax(logits, dim=-1)).sum(dim=-1, keepdim=True)
        return -neglogp

    @staticmethod
    def entropy(logits):
        a0 = logits - torch.max(logits, dim=-1, keepdim=True)[0]
        ea0 = torch.exp(a0)
        z0 = torch.sum(ea0, dim=-1, keepdim=True)
        p0 = ea0 / z0
        entropy = (p0 * (torch.log(z0) - a0)).sum(dim=-1)
        return entropy

    @staticmethod
    def sample(logits):
        # Gumbel-Max trick (>< Gumbel-Softmax trick)
        u = torch.empty(logits.size()).uniform_().to(logits.device)
        return torch.argmax(logits - torch.log(-torch.log(u)), dim=-1)

    @staticmethod
    def mode(logits):
        probs = torch.sigmoid(logits)
        return torch.argmax(probs)

    @staticmethod
    def kl(logits, other_logits):
        a0 = logits - torch.max(logits, dim=-1, keepdim=True)[0]
        a1 = other_logits - torch.max(other_logits, dim=-1, keepdim=True)[0]
        ea0 = torch.exp(a0)
        ea1 = torch.exp(a1)
        z0 = torch.sum(ea0, dim=-1, keepdim=True)
        z1 = torch.sum(ea1, dim=-1, keepdim=True)
        p0 = ea0 / z0
        kl = (p0 * (a0 - torch.log(z0) - a1 + torch.log(z1))).sum(dim=-1)
        return kl


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Perception stacks.

class ShallowMLP(nn.Module):

    def __init__(self, env, hps, hidden_size):
        """MLP layer stack as usually used in Deep RL"""
        super(ShallowMLP, self).__init__()
        ob_dim = env.observation_space.shape[0]
        # Assemble fully-connected encoder
        self.encoder = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim, hidden_size)),
                ('ln', nn.LayerNorm(hidden_size)),
                ('nl', nn.ReLU()),
            ]))),
        ]))
        # Perform initialization
        self.encoder.apply(init(nonlin='relu', param=None))

    def forward(self, x):
        return self.encoder(x)


class TeenyTinyCNN(nn.Module):

    def __init__(self, env, hps):
        super(TeenyTinyCNN, self).__init__()
        in_width, in_height, in_chan = env.observation_space.shape
        # Assemble the convolutional encoder
        self.encoder = nn.Sequential(OrderedDict([
            ('conv2d_block_1', nn.Sequential(OrderedDict([
                ('conv2d', nn.Conv2d(in_chan, 16, kernel_size=4, stride=2, padding=0)),
                ('nl', nn.ReLU()),
            ]))),
            ('conv2d_block_2', nn.Sequential(OrderedDict([
                ('conv2d', nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)),
                ('nl', nn.ReLU()),
            ]))),
        ]))
        # Assemble the fully-connected decoder
        self.decoder = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(32 * conv_to_fc(self.encoder, in_width, in_height), 64)),
                ('ln', nn.LayerNorm(64)),
                # According to the paper "Parameter Space Noise for Exploration", layer
                # normalization should only be used for the fully-connected part of the network.
                ('nl', nn.ReLU()),
            ]))),
        ]))
        # Perform initialization
        self.encoder.apply(init(nonlin='relu', param=None))
        self.decoder.apply(init(nonlin='relu', param=None))

    def forward(self, x):
        # Normalize the observations
        x /= 255.
        # Swap from NHWC to NCHW
        x = nhwc_to_nchw(x)
        # Stack the convolutional layers
        x = self.encoder(x)
        # Flatten the feature maps into a vector
        x = x.view(x.size(0), -1)
        # Stack the fully-connected layers
        x = self.decoder(x)
        # Return the resulting embedding
        return x


class NatureCNN(nn.Module):

    def __init__(self, env, hps):
        """CNN layer stack inspired from DQN's CNN from the Nature paper:
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        """
        super(NatureCNN, self).__init__()
        in_width, in_height, in_chan = env.observation_space.shape
        # Assemble the convolutional encoder
        self.encoder = nn.Sequential(OrderedDict([
            ('conv2d_block_1', nn.Sequential(OrderedDict([
                ('conv2d', nn.Conv2d(in_chan, 16, kernel_size=8, stride=4, padding=0)),
                ('nl', nn.ReLU()),
            ]))),
            ('conv2d_block_2', nn.Sequential(OrderedDict([
                ('conv2d', nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0)),
                ('nl', nn.ReLU()),
            ]))),
            ('conv2d_block_3', nn.Sequential(OrderedDict([
                ('conv2d', nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)),
                ('nl', nn.ReLU()),
            ]))),
        ]))
        # Assemble the fully-connected decoder
        self.decoder = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(64 * conv_to_fc(self.encoder, in_width, in_height), 512)),
                ('ln', nn.LayerNorm(512)),
                # According to the paper "Parameter Space Noise for Exploration", layer
                # normalization should only be used for the fully-connected part of the network.
                ('nl', nn.ReLU()),
            ]))),
        ]))
        # Perform initialization
        self.encoder.apply(init(nonlin='relu', param=None))
        self.decoder.apply(init(nonlin='relu', param=None))

    def forward(self, x):
        # Normalize the observations
        x /= 255.
        # Swap from NHWC to NCHW
        x = nhwc_to_nchw(x)
        # Stack the convolutional layers
        x = self.encoder(x)
        # Flatten the feature maps into a vector
        x = x.view(x.size(0), -1)
        # Stack the fully-connected layers
        x = self.decoder(x)
        # Return the resulting embedding
        return x


class SmallImpalaCNN(nn.Module):

    def __init__(self, env, hps):
        """CNN layer stack inspired from IMPALA's "small" CNN (2 convolutional layers)
        https://arxiv.org/abs/1802.01561
        Note that we do not use the last LSTM of the described architecture.
        """
        super(SmallImpalaCNN, self).__init__()
        in_width, in_height, in_chan = env.observation_space.shape
        # Assemble the convolutional encoder
        self.encoder = nn.Sequential(OrderedDict([
            ('conv2d_block_1', nn.Sequential(OrderedDict([
                ('conv2d', nn.Conv2d(in_chan, 16, kernel_size=8, stride=4, padding=0)),
                ('nl', nn.ReLU()),
            ]))),
            ('conv2d_block_2', nn.Sequential(OrderedDict([
                ('conv2d', nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0)),
                ('nl', nn.ReLU()),
            ]))),
        ]))
        # Assemble the fully-connected decoder
        self.decoder = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(32 * conv_to_fc(self.encoder, in_width, in_height), 256)),
                ('ln', nn.LayerNorm(256)),
                # According to the paper "Parameter Space Noise for Exploration", layer
                # normalization should only be used for the fully-connected part of the network.
                ('nl', nn.ReLU()),
            ]))),
        ]))
        # Perform initialization
        self.encoder.apply(init(nonlin='relu', param=None))
        self.decoder.apply(init(nonlin='relu', param=None))

    def forward(self, x):
        # Normalize the observations
        x /= 255.
        # Swap from NHWC to NCHW
        x = nhwc_to_nchw(x)
        # Stack the convolutional layers
        x = self.encoder(x)
        # Flatten the feature maps into a vector
        x = x.view(x.size(0), -1)
        # Stack the fully-connected layers
        x = self.decoder(x)
        # Return the resulting embedding
        return x


class LargeImpalaCNN(nn.Module):

    def __init__(self, env, hps):
        """CNN layer stack inspired from IMPALA's "large" CNN (15 convolutional layers)
        https://arxiv.org/abs/1802.01561
        Note that we do not use the last LSTM of the described architecture.
        """
        super(LargeImpalaCNN, self).__init__()
        in_width, in_height, in_chan = env.observation_space.shape
        # Assemble the convolutional encoder
        self.encoder = nn.Sequential(OrderedDict([
            ('impala_block_1', ImpalaBlock(in_chan, 16)),
            ('impala_block_2', ImpalaBlock(16, 32)),
            ('impala_block_3', ImpalaBlock(32, 32)),
            ('nl', nn.ReLU()),
        ]))
        # Assemble the fully-connected decoder
        self.decoder = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(32 * conv_to_fc(self.encoder, in_width, in_height), 256)),
                ('ln', nn.LayerNorm(256)),
                # According to the paper "Parameter Space Noise for Exploration", layer
                # normalization should only be used for the fully-connected part of the network.
                ('nl', nn.ReLU()),
            ]))),
        ]))
        # Perform initialization
        # Encoder already initialized at this point.
        self.decoder.apply(init(nonlin='relu', param=None))

    def forward(self, x):
        # Normalize the observations
        x /= 255.
        # Swap from NHWC to NCHW
        x = nhwc_to_nchw(x)
        # Stack the convolutional layers
        x = self.encoder(x)
        # Flatten the feature maps into a vector
        x = x.view(x.size(0), -1)
        # Stack the fully-connected layers
        x = self.decoder(x)
        # Return the resulting embedding
        return x


def perception_stack_parser(x):
    if x == 'shallow_mlp':
        return (lambda u, v: ShallowMLP(u, v, hidden_size=100)), 100
    elif x == 'teeny_tiny_cnn':
        return (lambda u, v: TeenyTinyCNN(u, v)), 64
    elif x == 'nature_cnn':
        return (lambda u, v: NatureCNN(u, v)), 512
    elif x == 'small_impala_cnn':
        return (lambda u, v: SmallImpalaCNN(u, v)), 256
    elif x == 'large_impala_cnn':
        return (lambda u, v: LargeImpalaCNN(u, v)), 256
    else:
        raise NotImplementedError("invalid perception stack")


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Networks.

class GaussPolicy(nn.Module):

    def __init__(self, env, hps):
        super(GaussPolicy, self).__init__()
        ac_dim = env.action_space.shape[0]
        self.hps = hps
        # Define perception stack
        net_lambda, fc_in = perception_stack_parser(self.hps.perception_stack)
        self.perception_stack = net_lambda(env, self.hps)
        # Assemble the last layers and output heads
        self.p_decoder = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(fc_in, fc_in)),
                ('ln', nn.LayerNorm(fc_in)),
                ('nl', nn.ReLU()),
            ]))),
        ]))
        self.p_head = nn.Linear(fc_in, ac_dim)
        self.ac_logstd_head = nn.Parameter(torch.full((ac_dim,), math.log(0.6)))
        self.v_decoder = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(fc_in, fc_in)),
                ('ln', nn.LayerNorm(fc_in)),
                ('nl', nn.ReLU()),
            ]))),
        ]))
        self.v_head = nn.Linear(fc_in, 1)
        if self.hps.binned_aux_loss or self.hps.squared_aux_loss:
            self.r_decoder = nn.Sequential(OrderedDict([
                ('fc_block', nn.Sequential(OrderedDict([
                    ('fc', nn.Linear(fc_in, fc_in)),
                    ('ln', nn.LayerNorm(fc_in)),
                    ('nl', nn.ReLU()),
                ]))),
            ]))
            self.r_head = nn.Linear(fc_in, 3 if self.hps.binned_aux_loss else 1)  # bins
        # Perform initialization
        self.p_decoder.apply(init(nonlin='relu', param=None))
        self.p_head.apply(init(weight_scale=0.01, constant_bias=0.0))
        self.v_decoder.apply(init(nonlin='relu', param=None))
        self.v_head.apply(init(nonlin='linear', param=None))
        if self.hps.binned_aux_loss or self.hps.squared_aux_loss:
            self.r_decoder.apply(init(nonlin='relu', param=None))
            self.r_head.apply(init(nonlin='linear', param=None))

    def logp(self, ob, ac):
        out = self.forward(ob)
        return NormalToolkit.logp(ac, *out[0:2])  # mean, std

    def entropy(self, ob):
        out = self.forward(ob)
        return NormalToolkit.entropy(out[1])  # std

    def sample(self, ob):
        with torch.no_grad():
            out = self.forward(ob)
            ac = NormalToolkit.sample(*out[0:2])  # mean, std
        return ac

    def mode(self, ob):
        with torch.no_grad():
            out = self.forward(ob)
            ac = NormalToolkit.mode(out[0])  # mean
        return ac

    def kl(self, ob, other):
        assert isinstance(other, GaussPolicy)
        with torch.no_grad():
            out_a = self.forward(ob)
            out_b = other.forward(ob)
            kl = NormalToolkit.kl(*out_a[0:2],
                                  *out_b[0:2])  # mean, std
        return kl

    def value(self, ob):
        out = self.forward(ob)
        return out[2]  # value

    def ss_aux_loss(self, ob):
        if self.hps.binned_aux_loss or self.hps.squared_aux_loss:
            out = self.forward(ob)
            return out[3]  # aux
        else:
            raise ValueError("should not be called")

    def forward(self, ob):
        # Extract features
        x = self.perception_stack(ob)
        # Go through residual fully-connected layers specific for each head
        # inspired from OpenAI's RND repo, file located at
        # random-network-distillation/blob/master/policies/cnn_policy_param_matched.py
        ac_mean = self.p_head(self.p_decoder(x))
        ac_std = self.ac_logstd_head.expand_as(ac_mean).exp()
        value = self.v_head(self.v_decoder(x))
        out = [ac_mean, ac_std, value]
        if self.hps.binned_aux_loss or self.hps.squared_aux_loss:
            aux = self.r_head(self.r_decoder(x))
            if self.hps.binned_aux_loss:
                aux = F.log_softmax(aux, dim=1).exp()
            out.append(out)
        return out


class CatPolicy(nn.Module):

    def __init__(self, env, hps):
        super(CatPolicy, self).__init__()
        ac_dim = env.action_space.n
        # Define perception stack
        net_lambda, fc_in = perception_stack_parser(hps.perception_stack)
        self.perception_stack = net_lambda(env, hps)
        # Assemble the last layers and output heads
        self.p_decoder = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(fc_in, fc_in)),
                ('ln', nn.LayerNorm(fc_in)),
                ('nl', nn.ReLU()),
            ]))),
        ]))
        self.p_head = nn.Linear(fc_in, ac_dim)
        self.v_decoder = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(fc_in, fc_in)),
                ('ln', nn.LayerNorm(fc_in)),
                ('nl', nn.ReLU()),
            ]))),
        ]))
        self.v_head = nn.Linear(fc_in, 1)
        # Perform initialization
        self.p_decoder.apply(init(nonlin='relu', param=None))
        self.p_head.apply(init(weight_scale=0.01, constant_bias=0.0))
        self.v_decoder.apply(init(nonlin='relu', param=None))
        self.v_head.apply(init(nonlin='linear', param=None))

    def logp(self, ob, ac):
        ac_logits, _ = self.forward(ob)
        return CatToolkit.logp(ac, ac_logits)

    def entropy(self, ob):
        ac_logits, _ = self.forward(ob)
        return CatToolkit.entropy(ac_logits)

    def sample(self, ob):
        # Gumbel-Max trick (>< Gumbel-Softmax trick)
        with torch.no_grad():
            ac_logits, _ = self.forward(ob)
            ac = CatToolkit.sample(ac_logits)
        return ac

    def mode(self, ob):
        with torch.no_grad():
            ac_logits, _ = self.forward(ob)
            ac = CatToolkit.mode(ac_logits)
        return ac

    def kl(self, ob, other):
        assert isinstance(other, CatPolicy)
        with torch.no_grad():
            ac_logits, _ = self.forward(ob)
            ac_logits_, _ = other.forward(ob)
            kl = CatToolkit.kl(ac_logits, ac_logits_)
        return kl

    def value(self, ob):
        _, v = self.forward(ob)
        return v

    def forward(self, ob):
        # Extract features
        x = self.perception_stack(ob)
        # Go through residual fully-connected layers specific for each head
        # inspired from OpenAI's RND repo, file located at
        # random-network-distillation/blob/master/policies/cnn_policy_param_matched.py
        ac_logits = self.p_head(self.p_decoder(x))
        value = self.v_head(self.v_decoder(x))
        # Return the heads
        return ac_logits, value


class Discriminator(nn.Module):

    def __init__(self, env, hps):
        super(Discriminator, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        self.hps = hps
        self.leak = 0.1
        # Define the input dimension, depending on whether actions are used too.
        in_dim = ob_dim if self.hps.state_only else ob_dim + ac_dim
        self.score_trunk = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', U.spectral_norm(nn.Linear(in_dim, 100))),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', U.spectral_norm(nn.Linear(100, 100))),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
        ]))
        self.score_head = nn.Linear(100, 1)
        # Perform initialization
        self.score_trunk.apply(init(weight_scale=math.sqrt(2) / math.sqrt(1 + self.leak**2)))
        self.score_head.apply(init())

    def D(self, ob, ac):
        return self.forward(ob, ac)

    def forward(self, ob, ac):
        x = ob if self.hps.state_only else torch.cat([ob, ac], dim=-1)
        x = self.score_trunk(x)
        score = self.score_head(x)
        return score  # no sigmoid here
