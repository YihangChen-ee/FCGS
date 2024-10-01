import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np

class Entropy_gaussian(nn.Module):
    def __init__(self, Q=1):
        super(Entropy_gaussian, self).__init__()
        self.Q = Q
    def forward(self, x, mean, scale, Q=None, x_mean=None, return_lkl=False):
        if Q is None:
            Q = self.Q
        scale = torch.clamp(scale, min=1e-9)
        m1 = torch.distributions.normal.Normal(mean, scale)
        lower = m1.cdf(x - 0.5*Q)
        upper = m1.cdf(x + 0.5*Q)
        likelihood = torch.abs(upper - lower)
        if return_lkl:
            return likelihood
        else:
            likelihood = Low_bound.apply(likelihood)
            bits = -torch.log2(likelihood)
            return bits

class Entropy_gaussian_mix_prob_3(nn.Module):
    def __init__(self, Q=1):
        super(Entropy_gaussian_mix_prob_3, self).__init__()
        self.Q = Q
    def forward(self, x,
                mean1, mean2, mean3,
                scale1, scale2, scale3,
                probs1, probs2, probs3,
                Q=None, x_mean=None, return_lkl=False):
        if Q is None:
            Q = self.Q

        scale1 = torch.clamp(scale1, min=1e-9)
        scale2 = torch.clamp(scale2, min=1e-9)
        scale3 = torch.clamp(scale3, min=1e-9)

        m1 = torch.distributions.normal.Normal(mean1, scale1)
        m2 = torch.distributions.normal.Normal(mean2, scale2)
        m3 = torch.distributions.normal.Normal(mean3, scale3)

        likelihood1 = torch.abs(m1.cdf(x + 0.5*Q) - m1.cdf(x - 0.5*Q))
        likelihood2 = torch.abs(m2.cdf(x + 0.5*Q) - m2.cdf(x - 0.5*Q))
        likelihood3 = torch.abs(m3.cdf(x + 0.5*Q) - m3.cdf(x - 0.5*Q))

        likelihood = Low_bound.apply(probs1 * likelihood1 + probs2 * likelihood2 + probs3 * likelihood3)

        if return_lkl:
            return likelihood
        else:
            likelihood = Low_bound.apply(likelihood)
            bits = -torch.log2(likelihood)
            return bits

class Entropy_gaussian_mix_prob_4(nn.Module):
    def __init__(self, Q=1):
        super(Entropy_gaussian_mix_prob_4, self).__init__()
        self.Q = Q
    def forward(self, x,
                mean1, mean2, mean3, mean4,
                scale1, scale2, scale3, scale4,
                probs1, probs2, probs3, probs4,
                Q=None, x_mean=None, return_lkl=False):
        if Q is None:
            Q = self.Q

        scale1 = torch.clamp(scale1, min=1e-9)
        scale2 = torch.clamp(scale2, min=1e-9)
        scale3 = torch.clamp(scale3, min=1e-9)
        scale4 = torch.clamp(scale4, min=1e-9)

        m1 = torch.distributions.normal.Normal(mean1, scale1)
        m2 = torch.distributions.normal.Normal(mean2, scale2)
        m3 = torch.distributions.normal.Normal(mean3, scale3)
        m4 = torch.distributions.normal.Normal(mean4, scale4)

        likelihood1 = torch.abs(m1.cdf(x + 0.5*Q) - m1.cdf(x - 0.5*Q))
        likelihood2 = torch.abs(m2.cdf(x + 0.5*Q) - m2.cdf(x - 0.5*Q))
        likelihood3 = torch.abs(m3.cdf(x + 0.5*Q) - m3.cdf(x - 0.5*Q))
        likelihood4 = torch.abs(m4.cdf(x + 0.5*Q) - m4.cdf(x - 0.5*Q))

        likelihood = Low_bound.apply(probs1 * likelihood1 + probs2 * likelihood2 + probs3 * likelihood3 + probs4 * likelihood4)

        if return_lkl:
            return likelihood
        else:
            likelihood = Low_bound.apply(likelihood)
            bits = -torch.log2(likelihood)
            return bits

class Entropy_gaussian_mix_prob_2(nn.Module):
    def __init__(self, Q=1):
        super(Entropy_gaussian_mix_prob_2, self).__init__()
        self.Q = Q
    def forward(self, x,
                mean1, mean2,
                scale1, scale2,
                probs1, probs2,
                Q=None, x_mean=None, return_lkl=False):
        if Q is None:
            Q = self.Q

        scale1 = torch.clamp(scale1, min=1e-9)
        scale2 = torch.clamp(scale2, min=1e-9)

        m1 = torch.distributions.normal.Normal(mean1, scale1)
        m2 = torch.distributions.normal.Normal(mean2, scale2)

        likelihood1 = torch.abs(m1.cdf(x + 0.5*Q) - m1.cdf(x - 0.5*Q))
        likelihood2 = torch.abs(m2.cdf(x + 0.5*Q) - m2.cdf(x - 0.5*Q))

        likelihood = Low_bound.apply(probs1 * likelihood1 + probs2 * likelihood2)

        if return_lkl:
            return likelihood
        else:
            likelihood = Low_bound.apply(likelihood)
            bits = -torch.log2(likelihood)
            return bits

class Entropy_bernoulli(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, p):
        # p = torch.sigmoid(p)
        p = torch.clamp(p, min=1e-6, max=1 - 1e-6)
        pos_mask = (1 + x) / 2.0  # 1 -> 1, -1 -> 0
        neg_mask = (1 - x) / 2.0  # -1 -> 1, 1 -> 0
        pos_prob = p
        neg_prob = 1 - p
        param_bit = -torch.log2(pos_prob) * pos_mask + -torch.log2(neg_prob) * neg_mask
        return param_bit


class Entropy_factorized(nn.Module):
    def __init__(self, channel=32, init_scale=10, filters=(3, 3, 3), likelihood_bound=1e-6,
                 tail_mass=1e-9, optimize_integer_offset=True, Q=1):
        super(Entropy_factorized, self).__init__()
        self.filters = tuple(int(t) for t in filters)
        self.init_scale = float(init_scale)
        self.likelihood_bound = float(likelihood_bound)
        self.tail_mass = float(tail_mass)
        self.optimize_integer_offset = bool(optimize_integer_offset)
        self.Q = Q
        if not 0 < self.tail_mass < 1:
            raise ValueError(
                "`tail_mass` must be between 0 and 1")
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1.0 / (len(self.filters) + 1))
        self._matrices = nn.ParameterList([])
        self._bias = nn.ParameterList([])
        self._factor = nn.ParameterList([])
        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1.0 / scale / filters[i + 1]))
            self.matrix = nn.Parameter(torch.FloatTensor(
                channel, filters[i + 1], filters[i]))
            self.matrix.data.fill_(init)
            self._matrices.append(self.matrix)
            self.bias = nn.Parameter(
                torch.FloatTensor(channel, filters[i + 1], 1))
            noise = np.random.uniform(-0.5, 0.5, self.bias.size())
            noise = torch.FloatTensor(noise)
            self.bias.data.copy_(noise)
            self._bias.append(self.bias)
            if i < len(self.filters):
                self.factor = nn.Parameter(
                    torch.FloatTensor(channel, filters[i + 1], 1))
                self.factor.data.fill_(0.0)
                self._factor.append(self.factor)

    def _logits_cumulative(self, logits, stop_gradient):
        for i in range(len(self.filters) + 1):
            matrix = nnf.softplus(self._matrices[i])
            if stop_gradient:
                matrix = matrix.detach()
            # print('dqnwdnqwdqwdqwf:', matrix.shape, logits.shape)
            logits = torch.matmul(matrix, logits)
            bias = self._bias[i]
            if stop_gradient:
                bias = bias.detach()
            logits += bias
            if i < len(self._factor):
                factor = nnf.tanh(self._factor[i])
                if stop_gradient:
                    factor = factor.detach()
                logits += factor * nnf.tanh(logits)
        return logits

    def forward(self, x, Q=None, return_lkl=False):
        # x: [N, C], quantized
        if Q is None:
            Q = self.Q
        else:
            if isinstance(Q, torch.Tensor):
                Q = Q.permute(1, 0).contiguous()  # [C, N]
                Q = Q.view(Q.shape[0], 1, -1)  # [C, 1, N]
        x = x.permute(1, 0).contiguous()  # [C, N]
        x = x.view(x.shape[0], 1, -1)  # [C, 1, N]
        lower = self._logits_cumulative(x - 0.5*Q, stop_gradient=False)
        upper = self._logits_cumulative(x + 0.5*Q, stop_gradient=False)
        sign = -torch.sign(torch.add(lower, upper))
        sign = sign.detach()
        likelihood = torch.abs(nnf.sigmoid(sign * upper) - nnf.sigmoid(sign * lower))  # [C, 1, N]
        if return_lkl:
            likelihood = likelihood.view(likelihood.shape[0], -1).permute(1, 0).contiguous()
            return likelihood
        else:
            likelihood = Low_bound.apply(likelihood)  # [C, 1, N]
            bits = -torch.log2(likelihood)  # [C, 1, N]
            bits = bits.view(bits.shape[0], -1)  # [C, N]
            bits = bits.permute(1, 0).contiguous()  # [N, C]
            return bits

class Low_bound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        x = torch.clamp(x, min=1e-6)
        return x

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        grad1 = g.clone()
        grad1[x < 1e-6] = 0
        pass_through_if = np.logical_or(
            x.cpu().numpy() >= 1e-6, g.cpu().numpy() < 0.0)
        t = torch.Tensor(pass_through_if+0.0).cuda()
        return grad1 * t
