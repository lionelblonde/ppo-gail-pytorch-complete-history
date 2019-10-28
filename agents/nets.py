import math

import torch
import torch.nn as nn
import torch.nn.modules.rnn as rnn
import torch.nn.functional as F
import torch.nn.utils as U


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Core.

def ortho_init(module, nonlinearity=None, weight_scale=1.0, constant_bias=0.0):
    """Applies orthogonal initialization for the parameters of a given module"""

    if nonlinearity is not None:
        gain = nn.init.calculate_gain(nonlinearity)
    else:
        gain = weight_scale

    if isinstance(module, (nn.RNNBase, rnn.RNNCellBase)):
        for name, param in module.named_parameters():
            if 'weight_' in name:
                nn.init.orthogonal_(param, gain=gain)
            elif 'bias_' in name:
                nn.init.constant_(param, constant_bias)
    else:  # other modules with single .weight and .bias
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, constant_bias)


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_state_size, hps):
        """LSTM implementation, Hochreiter & Schmidhuber 1997
        https://www.bioinf.jku.at/publications/older/2604.pdf
        Optional add-on: layer normalization
        """
        super(LSTM, self).__init__()
        self.hps = hps
        # Create layers for the two types of products there are in lstm's gate equations
        # namely W * x and W * h. We create only two and leave a separate piece of it for each gate
        self.xh = nn.Linear(input_size, 4 * hidden_state_size)
        self.hh = nn.Linear(hidden_state_size, 4 * hidden_state_size)
        ortho_init(self.xh, 'tanh', constant_bias=0.0)
        ortho_init(self.hh, 'tanh', constant_bias=0.0)

        # Define layernorm layers
        self.ln_x = nn.LayerNorm(4 * hidden_state_size) if hps.with_layernorm else lambda x: x
        self.ln_h = nn.LayerNorm(4 * hidden_state_size) if hps.with_layernorm else lambda x: x
        self.ln_c = nn.LayerNorm(hidden_state_size) if hps.with_layernorm else lambda x: x

    def forward(self, xs, masks, state):
        # Split the input state into hidden state and cell state
        h, c = state.chunk(2, 1)
        # Create a list to collect the hidden states (monitoring purposes)
        hs = []

        # Go over the sequence of inputs and masks from the current hidden state
        for t, (x, mask) in enumerate(zip(xs, masks)):
            # Apply masks (zero out if end of episode reached)
            h = h * (1 - mask)
            c = c * (1 - mask)
            # Create a matrix that contains the products present in all gate equations
            gates = self.ln_x(self.xh(x)) + self.ln_h(self.hh(h))
            # Split the matrix in four equal pieces so that each gate can learn its own piece
            f_g, i_g, c_g, o_g = gates.chunk(4, 1)
            # Apply the nonlinearities for each of the gates
            f_g = torch.sigmoid(f_g)
            i_g = torch.sigmoid(i_g)
            c_g = torch.tanh(c_g)
            o_g = torch.sigmoid(o_g)
            # Assemble the new lstm state, namely the hidden state and the cell state
            c = f_g * c + i_g * c_g
            h = o_g * torch.tanh(self.ln_c(c))

            # Add the hidden state to the list
            hs.append(h)

        # Return the list of consecutive hidden state and the last lstm state (hidden and cell)
        hc = torch.cat([h, c], dim=1)
        # Return the list of hidden state and the LSTM state (hidden + cell)
        return torch.stack(hs), hc


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


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Feature extractors.

class ShallowMLP(nn.Module):

    def __init__(self, env, hps):
        """MLP layer stack as usually used in Deep RL"""
        super(ShallowMLP, self).__init__()
        ob_dim = env.observation_space.shape[0]
        # Create fully-connected layers
        self.fc_1 = nn.Linear(ob_dim, 64)
        self.fc_2 = nn.Linear(64, 64)
        ortho_init(self.fc_1, nonlinearity='tanh', constant_bias=0.0)
        ortho_init(self.fc_2, nonlinearity='tanh', constant_bias=0.0)
        # Define layernorm layers
        self.ln_1 = nn.LayerNorm(64) if hps.with_layernorm else lambda x: x
        self.ln_2 = nn.LayerNorm(64) if hps.with_layernorm else lambda x: x

    def forward(self, ob):
        plop = ob
        # Stack fully-connected layers
        plop = torch.tanh(self.ln_1(self.fc_1(plop)))
        plop = torch.tanh(self.ln_2(self.fc_2(plop)))
        return plop


def parser_x(x):
    if x == 'shallow_mlp':
        return (lambda u, v: ShallowMLP(u, v)), 64
    elif x == 'none':
        return (lambda x: x), None
    else:
        raise NotImplementedError("invalid feature extractor")


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Networks.

class MLPGaussPolicy(nn.Module):

    def __init__(self, env, hps):
        super(MLPGaussPolicy, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]

        # Define feature extractor
        net_lambda, fc_in = parser_x(hps.feat_x_p)
        self.net_x = net_lambda(env, hps)

        # Define output heads
        self.ac_mean_head = nn.Linear(fc_in if fc_in is not None else ob_dim, ac_dim)
        ortho_init(self.ac_mean_head, weight_scale=0.01, constant_bias=0.0)
        self.ac_logstd_head = nn.Parameter(torch.full((ac_dim,), math.log(0.6)))
        self.value_head = nn.Linear(fc_in if fc_in is not None else ob_dim, 1)
        ortho_init(self.value_head, nonlinearity='linear', constant_bias=0.0)

    def logp(self, ob, ac):
        ac_mean, ac_std, _ = self.forward(ob)
        return NormalToolkit.logp(ac, ac_mean, ac_std)

    def entropy(self, ob):
        _, ac_std, _ = self.forward(ob)
        return NormalToolkit.entropy(ac_std)

    def sample(self, ob):
        with torch.no_grad():
            ac_mean, ac_std, _ = self.forward(ob)
            ac = NormalToolkit.sample(ac_mean, ac_std)
        return ac

    def mode(self, ob):
        with torch.no_grad():
            ac_mean, _, _ = self.forward(ob)
            ac = NormalToolkit.mode(ac_mean)
        return ac

    def kl(self, ob, other):
        assert isinstance(other, GaussPolicy)
        with torch.no_grad():
            ac_mean, ac_logstd, _ = self.forward(ob)
            ac_mean_, ac_logstd_, _ = other.forward(ob)
            kl = NormalToolkit.kl(ac_mean, ac_logstd,
                                  ac_mean_, ac_logstd_)
        return kl

    def value(self, ob):
        _, _, v = self.forward(ob)
        return v

    def forward(self, ob):
        plop = ob
        # Pipe through the feature extractor
        plop = self.net_x(plop)
        # Go through the output heads
        ac_mean = self.ac_mean_head(plop)
        ac_std = self.ac_logstd_head.expand_as(ac_mean).exp()
        value = self.value_head(plop)
        return ac_mean, ac_std, value


class LSTMGaussPolicy(nn.Module):

    def __init__(self, env, hps):
        super(LSTMGaussPolicy, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]

        # Define feature extractor
        net_lambda, fc_in = parser_x(hps.feat_x_p)
        self.net_x = net_lambda(env, hps)

        # Define reccurent layer
        self.lstm = LSTM(fc_in if fc_in is not None else ob_dim, hps.hidden_state_size, hps)

        # Define output heads
        self.ac_mean_head = nn.Linear(hps.hidden_state_size, ac_dim)
        ortho_init(self.ac_mean_head, weight_scale=0.01, constant_bias=0.0)
        self.ac_logstd_head = nn.Parameter(torch.full((ac_dim,), math.log(0.6)))
        self.value_head = nn.Linear(hps.hidden_state_size, 1)
        ortho_init(self.value_head, nonlinearity='linear', constant_bias=0.0)

    def logp(self, ob, ac, done, state):
        ac_mean, ac_std, _, _ = self.forward(ob, done, state)
        return NormalToolkit.logp(ac, ac_mean, ac_std)

    def entropy(self, ob, done, state):
        _, ac_std, _, _ = self.forward(ob, done, state)
        return NormalToolkit.entropy(ac_std)

    def sample(self, ob, done, state):
        # Reparameterization trick
        with torch.no_grad():
            ac_mean, ac_std, state, _ = self.forward(ob, done, state)
            ac = NormalToolkit.sample(ac_mean, ac_std)
        return ac, state

    def mode(self, ob, done, state):
        with torch.no_grad():
            ac_mean, _, state, _ = self.forward(ob, done, state)
            ac = NormalToolkit.mode(ac_mean)
        return ac, state

    def kl(self, ob, done, state, other):
        assert isinstance(other, LSTMGaussPolicy)
        with torch.no_grad():
            ac_mean, ac_logstd, _, _ = self.forward(ob, done, state)
            ac_mean_, ac_logstd_, _, _ = other.forward(ob, done, state)
            kl = NormalToolkit.kl(ac_mean, ac_logstd,
                                  ac_mean_, ac_logstd_)
        return kl

    def value(self, ob, done, state):
        _, _, _, v = self.forward(ob, done, state)
        return v

    def forward(self, ob, done, state):
        plop = ob
        # Pipe through the feature extractor
        plop = self.net_x(plop)

        # Add LSTM
        _, hc = self.lstm(plop, done, state)
        h, c = hc.chunk(2, 1)

        # Go through the output heads
        ac_mean = self.ac_mean_head(h)
        ac_std = self.ac_logstd_head.expand_as(ac_mean).exp()
        value = self.value_head(h)
        return ac_mean, ac_std, hc, value


class MLPCatPolicy(nn.Module):

    def __init__(self, env, hps):
        super(MLPCatPolicy, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.n

        # Define feature extractor
        net_lambda, fc_in = parser_x(hps.feat_x_p)
        self.net_x = net_lambda(env, hps)

        # Define output layer
        self.ac_logits_head = nn.Linear(fc_in if fc_in is not None else ob_dim, ac_dim)
        ortho_init(self.ac_logits_head, weight_scale=0.01, constant_bias=0.0)
        self.value_head = nn.Linear(fc_in if fc_in is not None else ob_dim, 1)
        ortho_init(self.value_head, nonlinearity='linear', constant_bias=0.0)

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
        plop = ob
        # Pipe through the feature extractor
        plop = self.net_x(plop)
        # Go through the output heads
        ac_logits = self.ac_logits_head(plop)
        value = self.value_head(plop)
        return ac_logits, value


class LSTMCatPolicy(nn.Module):

    def __init__(self, env, hps):
        super(LSTMCatPolicy, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.n

        # Define feature extractor
        net_lambda, fc_in = parser_x(hps.feat_x_p)
        self.net_x = net_lambda(env, hps)

        # Define reccurent layer
        self.lstm = LSTM(fc_in if fc_in is not None else ob_dim, hps.hidden_state_size, hps)

        # Define output layer
        self.ac_logits_head = nn.Linear(hps.hidden_state_size, ac_dim)
        ortho_init(self.ac_logits_head, weight_scale=0.01, constant_bias=0.0)
        self.value_head = nn.Linear(hps.hidden_state_size, 1)
        ortho_init(self.value_head, nonlinearity='linear', constant_bias=0.0)

    def logp(self, ob, ac, done, state):
        ac_logits, _, _ = self.forward(ob, done, state)
        return CatToolkit.logp(ac, ac_logits)

    def entropy(self, ob, done, state):
        ac_logits, _, _ = self.forward(ob, done, state)
        return CatToolkit.entropy(ac_logits)

    def sample(self, ob, done, state):
        # Reparameterization trick
        with torch.no_grad():
            ac_logits, state, _ = self.forward(ob, done, state)
            ac = CatToolkit.sample(ac_logits)
        return ac, state

    def mode(self, ob, done, state):
        with torch.no_grad():
            ac_logits, state, _ = self.forward(ob, done, state)
            ac = CatToolkit.mode(ac_logits)
        return ac, state

    def kl(self, ob, done, state, other):
        assert isinstance(other, LSTMGaussPolicy)
        with torch.no_grad():
            ac_logits, _, _ = self.forward(ob, done, state)
            ac_logits_, _, _ = other.forward(ob, done, state)
            kl = CatToolkit.kl(ac_logits, ac_logits_)
        return kl

    def value(self, ob, done, state):
        _, _, v = self.forward(ob, done, state)
        return v

    def forward(self, ob, done, state):
        plop = ob
        # Pipe through the feature extractor
        plop = self.net_x(plop)
        # Add LSTM layers
        _, hc = self.lstm(plop, done, state)
        h, c = hc.chunk(2, 1)
        # Go through the output heads
        ac_logits = self.ac_logits_head(h)
        value = self.value_head(h)
        return ac_logits, hc, value


class Discriminator(nn.Module):

    def __init__(self, env, hps):
        super(Discriminator, self).__init__()
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]
        self.hps = hps

        if self.hps.state_only:
            in_dim = self.ob_dim
        else:
            in_dim = self.ob_dim + self.ac_dim

        # Define hidden layers
        self.fc_1 = U.spectral_norm(nn.Linear(in_dim, 64))
        self.fc_2 = U.spectral_norm(nn.Linear(64, 64))
        ortho_init(self.fc_1, nonlinearity='leaky_relu', constant_bias=0.0)
        ortho_init(self.fc_2, nonlinearity='leaky_relu', constant_bias=0.0)

        # Define layernorm layers
        self.ln_1 = nn.LayerNorm(64) if self.hps.with_layernorm else lambda x: x
        self.ln_2 = nn.LayerNorm(64) if self.hps.with_layernorm else lambda x: x

        # Define score head
        self.score_head = nn.Linear(64, 1)
        ortho_init(self.score_head, nonlinearity='linear', constant_bias=0.0)

    def get_reward(self, ob, ac):
        """Craft surrogate reward"""
        ob = torch.FloatTensor(ob).cpu()
        ac = torch.FloatTensor(ac).cpu()

        # Counterpart of GAN's minimax (also called "saturating") loss
        # Numerics: 0 for non-expert-like states, goes to +inf for expert-like states
        # compatible with envs with traj cutoffs for bad (non-expert-like) behavior
        # e.g. walking simulations that get cut off when the robot falls over
        minimax_reward = -torch.log(1. - torch.sigmoid(self.D(ob, ac).detach()) + 1e-8)

        if self.hps.minimax_only:
            return minimax_reward
        else:
            # Counterpart of GAN's non-saturating loss
            # Recommended in the original GAN paper and later in (Fedus et al. 2017)
            # Numerics: 0 for expert-like states, goes to -inf for non-expert-like states
            # compatible with envs with traj cutoffs for good (expert-like) behavior
            # e.g. mountain car, which gets cut off when the car reaches the destination
            non_satur_reward = torch.log(torch.sigmoid(self.D(ob, ac).detach()))
            # Return the sum the two previous reward functions (as in AIRL, Fu et al. 2018)
            # Numerics: might be better might be way worse
            return non_satur_reward + minimax_reward

    def D(self, ob, ac):
        if self.hps.state_only:
            plop = ob
        else:
            plop = torch.cat([ob, ac], dim=-1)
        # Add hidden layers
        plop = F.leaky_relu(self.ln_1(self.fc_1(plop)))
        plop = F.leaky_relu(self.ln_2(self.fc_2(plop)))
        # Add output layer
        score = self.score_head(plop)
        return score

    def forward(self, ob, ac):
        score = self.D(ob, ac)
        return score
