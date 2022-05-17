import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mog_eval(mog, data):
    B, N, d = data.shape
    pi, mu, Sigma = mog
    if len(pi.shape) == 1:
        pi = pi.unsqueeze(0).repeat(B, 1)
        mu = mu.unsqueeze(0).repeat(B, 1, 1)
        Sigma = Sigma.unsqueeze(0).repeat(B, 1, 1)
    p = pi.shape[-1]

    jll = -0.5 * (d * np.log(2 * np.pi) +
        Sigma.log().sum(-1).unsqueeze(1) +
        torch.bmm(data ** 2, 1. / Sigma.permute(0, 2, 1)) +
        ((mu ** 2) / Sigma).sum(-1).unsqueeze(1) +
        -2. * torch.bmm(data, (mu / Sigma).permute(0, 2, 1))
    ) + pi.log().unsqueeze(1)

    mll = jll.logsumexp(-1)
    cll = jll - mll.unsqueeze(-1)

    return jll, cll, mll


class DirNIWNet(nn.Module):
    def __init__(self, p, d):
        super(DirNIWNet, self).__init__()
        self.m = nn.Parameter(0.1 * torch.randn(p, d))
        self.V_ = nn.Parameter(np.log(np.exp(1) - 1) + 0.001 / np.sqrt(d) * torch.randn(p, d))
        self.p, self.d = p, d

    def forward(self):
        V = F.softplus(self.V_)
        return self.m, V

    def mode(self, prior=None):
        if prior is None:
            m, V = self.forward()
        else:
            m, V = prior
        pi = torch.ones(self.p).to(m) / self.p
        mu = m
        Sigma = V
        return pi, mu, Sigma

    def loglik(self, theta):
        raise NotImplementedError

    def map_m_step(self, data, weight, tau=1.0, prior=None):
        B, N, d = data.shape

        if prior is None:
            m, V = self.forward()
        else:
            m, V = prior

        wsum = weight.sum(1)
        wsum_reg = wsum + tau
        wxsum = torch.bmm(weight.permute(0, 2, 1), data)
        wxxsum = torch.bmm(weight.permute(0, 2, 1), data ** 2)
        pi = wsum_reg / wsum_reg.sum(1, keepdim=True)
        mu = (wxsum + m.unsqueeze(0) * tau) / wsum_reg.unsqueeze(-1)
        Sigma = (wxxsum + (V + m ** 2).unsqueeze(0) * tau) / wsum_reg.unsqueeze(-1) - mu ** 2

        return pi, mu, Sigma

    def map_em(self, data, mask=None, num_iters=3, tau=1.0, prior=None):
        B, N, d = data.shape

        if mask is None:
            mask = torch.ones(B, N).to(data)
        pi, mu, Sigma = self.mode(prior)
        pi = pi.unsqueeze(0).repeat(B, 1)
        mu = mu.unsqueeze(0).repeat(B, 1, 1)
        Sigma = Sigma.unsqueeze(0).repeat(B, 1, 1)

        for emiter in range(num_iters):
            _, qq, _ = mog_eval((pi, mu, Sigma), data)
            qq = qq.exp() * mask.unsqueeze(-1)
            pi, mu, Sigma = self.map_m_step(data, weight=qq, tau=tau, prior=prior)

        return pi, mu, Sigma
