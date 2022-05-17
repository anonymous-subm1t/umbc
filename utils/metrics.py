"""
for containing statistics fucntions and anything else which can be re-used in multiple scenarios
"""


from typing import Tuple

import torch
from torch import nn
from torch.distributions import Normal

T = torch.Tensor

__all__ = ["ece", "ece_partial", "ece_partial_final", "reg_cal_err"]


def ece(y: T, logits: T, n_bins: int = 15, softmaxxed: bool = False) -> Tuple[float, T, T]:
    """vectorized version of expected calibration error (ECE) as outlined in https://arxiv.org/abs/1706.04599"""

    # intervals and boundaries for the bin probabilities
    interval = 1.0 / n_bins
    bound = torch.linspace(0, 1 - interval, n_bins, device=y.device)

    y_hat = logits
    if not softmaxxed:
        y_hat = logits.softmax(dim=1)

    y_hat_acc = y_hat.argmax(dim=1) == y
    y_hat_conf, _ = y_hat.max(dim=1)

    bins_acc = y_hat_acc.repeat(n_bins, 1)
    bins_conf = y_hat_conf.repeat(n_bins, 1)

    mask_conf = (bins_conf.T <= bound + interval).T * (bins_conf.T > bound).T  # type: ignore

    bins_acc = bins_acc * mask_conf
    bins_conf = bins_conf * mask_conf

    conf = bins_conf.sum(dim=1) / ((bins_conf != 0).sum(dim=1) + 1e-10)
    acc = bins_acc.sum(dim=1) / ((bins_conf != 0).sum(dim=1) + 1e-10)

    weights = (bins_conf != 0).sum(dim=1).float() / bins_conf.size(1)
    return torch.sum(weights * torch.abs(conf - acc)).item(), conf, acc


def ece_partial(y: T, logits: T, n_bins: int = 15, softmaxxed: bool = False) -> Tuple[T, T, T, int]:
    """
    vectorized version of expected calibration error (ECE) as outlined in https://arxiv.org/abs/1706.04599
    y: (n,)
    logits: (n, classes)
    """

    # intervals and boundaries for the bin probabilities
    interval = 1.0 / n_bins
    bound = torch.linspace(0, 1 - interval, n_bins)

    y_hat = logits.softmax(dim=1) if not softmaxxed else logits

    y_hat_acc = y_hat.argmax(dim=1) == y
    y_hat_conf, _ = y_hat.max(dim=1)

    bins_acc = y_hat_acc.repeat(n_bins, 1)
    bins_conf = y_hat_conf.repeat(n_bins, 1)

    mask_conf = (bins_conf.T <= bound + interval).T * (bins_conf.T > bound).T  # type: ignore

    bins_acc = bins_acc * mask_conf
    bins_conf = bins_conf * mask_conf

    conf = bins_conf.sum(dim=1)
    acc = bins_acc.sum(dim=1)

    n_in_bins = (bins_conf != 0).sum(dim=1).float()
    n = bins_conf.size(1)

    return conf, acc, n_in_bins, n


def ece_partial_final(conf: T, acc: T, n_in_bins: T, n: int) -> float:
    conf = conf / (n_in_bins + 1e-10)
    acc = acc / (n_in_bins + 1e-10)

    weights = n_in_bins / n
    return torch.sum(weights * torch.abs(acc - conf)).item()


def reg_cal_err(mu: T, sigma: T, y: T, bins: int = 15) -> torch.Tensor:
    """return the calibration error as defined in https://arxiv.org/abs/1807.00263"""
    yhat = Normal(mu, sigma)

    intervals = torch.linspace(1.0 / bins, 1.0, bins).to(y.device)  # (25)
    cdf_vals = yhat.cdf(y).repeat(bins, 1)  # (bins, yhat.size())

    out = (cdf_vals <= intervals.unsqueeze(1)).sum(dim=1) / float(cdf_vals.size(1))
    return torch.abs((out - intervals) ** 2).sum()  # type: ignore
