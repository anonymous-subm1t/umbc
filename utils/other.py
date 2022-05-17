import argparse
import hashlib
import itertools
import logging
import os
import random
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Tuple, Type

import numpy as np  # type: ignore
import torch
from matplotlib.colors import to_rgba  # type: ignore
from scipy import interpolate  # type: ignore
from torch import nn

__all__ = ["seed", "set_logger", "str2bool", "get_module_root", "random_features", "md5", "remove_dups"]


T = torch.Tensor


def md5(x: str) -> str:
    return hashlib.md5(x.encode()).hexdigest()


# https://github.com/google/edward2/blob/720d7985a889362d421053965b6c528390557b05/edward2/tensorflow/initializers.py#L759
# based off of the implementation here. keras orthogonal first performs a QR decomposition to get an orthogonal matrix and samples
# rows until there are enough to form the right size. The SNGP implementation uses the random norms which rescales col norms.
# https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/random_feature.py#L161
def random_features(rows: int, cols: int, stddev: float = 0.05, orthogonal: bool = False, random_norm: bool = True) -> T:
    if orthogonal:
        cols_sampled, c = 0, []
        while cols_sampled < cols:
            # qr only returns orthogonal Q's as an (N, N) square matrix
            c.append(stddev * torch.linalg.qr(torch.randn(rows, rows, requires_grad=False), mode="complete")[0])
            cols_sampled += rows

        w = torch.cat(c, dim=-1)[:, :cols]

        # if not random norms for the columns, scale each norm column by the expected norm of each column.
        # https://github.com/google/edward2/blob/720d7985a889362d421053965b6c528390557b05/edward2/tensorflow/initializers.py#L814
        if not random_norm:
            return w * np.sqrt(rows)  # type: ignore

        col_norms = (torch.randn(rows, cols) ** 2).sum(dim=0).sqrt()
        return w * col_norms  # type: ignore

    return stddev * torch.randn(rows, cols, requires_grad=False)


def get_module_root() -> str:
    # return the module root which is the grandparent of this file
    return str(Path(os.path.abspath(__file__)).parent.parent)


def seed(run: int) -> None:
    torch.manual_seed(run)
    random.seed(run)
    np.random.seed(run)



def set_logger(level: str) -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=level,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    return logging.getLogger()


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def remove_dups(a: List[Any]) -> List[Any]:
    """remove duplicate entries from a list-like object"""
    for i in range(len(a)):
        j = i + 1
        while j < len(a):
            if a[i] == a[j]:
                a = a[:j] + a[j + 1:]
                continue
            j += 1
    return a
