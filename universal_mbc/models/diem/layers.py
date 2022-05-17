import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from universal_mbc.models.diem import networks


class DIEM(nn.Module):
    def __init__(self, d, H=2, p=5, L=3, tau=10.0, out='param_cat', distr_emb_args=None):
        super(DIEM, self).__init__()
        self.priors = nn.ModuleList([networks.DirNIWNet(p, d) for _ in range(H)])
        self.H = H
        self.L = L
        self.tau = tau
        self.out = out

        if out == 'param_cat':
            self.outdim = H * (1 + p + 2 * p * d)
        elif out == 'select_best':
            self.outdim = 1 * (p + 2 * p * d)
        elif out == 'select_best2':
            self.outdim = 1 * (H + p + 2 * p * d)
        else:
            raise NotImplementedError

    def forward(self, S, mask=None):
        B, N_max, d = S.shape

        if mask is None:
            mask = torch.ones(B, N_max).to(S)

        pis, mus, Sigmas, alphas = [], [], [], []
        for h in range(self.H):
            pi, mu, Sigma = self.priors[h].map_em(
                S, mask=mask, num_iters=self.L, tau=self.tau, prior=self.priors[h]()
            )
            _, _, alpha = networks.mog_eval((pi, mu, Sigma), S)
            alpha = (alpha * mask).sum(-1)
            pis.append(pi)
            mus.append(mu)
            Sigmas.append(Sigma)
            alphas.append(alpha)

        pis = torch.stack(pis, dim=2)
        mus = torch.stack(mus, dim=3)
        Sigmas = torch.stack(Sigmas, dim=3)
        alphas = torch.stack(alphas, dim=1)
        alphas = (alphas - alphas.logsumexp(1, keepdim=True)).exp()

        if self.out == 'param_cat':
            out = torch.cat([alphas, pis.reshape(B, -1), mus.reshape(B, -1), Sigmas.reshape(B, -1)], dim=1)
        elif self.out == 'select_best':
            _, idx = alphas.max(1)
            pi, mu, Sigma = pis[range(B), :, idx], mus[range(B), :, :, idx], Sigmas[range(B), :, :, idx]
            out = torch.cat([pi.reshape(B, -1), mu.reshape(B, -1), Sigma.reshape(B, -1)], dim=1)
        elif self.out == 'select_best2':
            _, idx = alphas.max(1)
            pi, mu, Sigma = pis[range(B), :, idx], mus[range(B), :, :, idx], Sigmas[range(B), :, :, idx]
            out = torch.cat([1.0 * F.one_hot(idx, self.H).to(alphas), pi.reshape(B, -1), mu.reshape(B, -1), Sigma.reshape(B, -1)], dim=1)
        else:
            raise NotImplementedError

        return out
