import os
from typing import Tuple

import numpy as np
import pandas as pd  # type: ignore
import torch
from torch.distributions import Categorical, Dirichlet
from torch.utils.data import Dataset  # type: ignore
from torchvision import datasets  # type: ignore
from torchvision.transforms import Compose  # type: ignore

T = torch.Tensor


class ImageNetClusters(Dataset):
    def __init__(self, x: T, y: T, clusters: int = 4, batch_size: int = 10, set_size: int = 256) -> None:
        """
        x and y in this case should be features extracted from imagenet with some other model.
        Returns a random of classes concatenated together which can then be passed to a clustering algorithm
        """
        self.y = y
        self.name = "ImageNetClusters"
        self.clusters = clusters
        self.batch_size = batch_size
        self.set_size = set_size

        self.classes = []
        for lbl in torch.unique(y):
            idx = (y == lbl).nonzero(as_tuple=True)[0]
            self.classes.append(x[idx])

    def __len__(self) -> int:
        return 1000 * self.batch_size  # one epoch will always be 1000 iterations regardless of batch size

    def __getitem__(self, i: int) -> Tuple[T, T]:
        return self.get_item()

    def sample_n_from_class(self, i: int, n: int) -> T:
        # sample N indices from tensor with the size of the class, replace is true because we might sample more than exist for
        # for a class and we want to ensure a random sample from that class.
        idx = torch.multinomial(torch.ones(self.classes[i].size(0)), n, replacement=True)
        return self.classes[i][idx]

    def get_item(self) -> Tuple[T, T]:
        classes = np.random.choice(len(self.classes), self.clusters, replace=False)

        pi = Dirichlet(torch.ones(self.clusters)).sample()
        labels = Categorical(probs=pi).sample(torch.Size([self.clusters * self.set_size]))
        N = (labels.unsqueeze(0) == torch.arange(self.clusters).unsqueeze(-1)).sum(dim=-1)  # (1, N) == (4, 1)

        # if n == 0 then we just need to have an empty tensor for this class
        x = torch.cat([self.sample_n_from_class(k, n) if n > 0 else torch.Tensor() for (k, n) in zip(classes, N)])
        y = torch.cat([torch.ones(self.set_size) * i for i, _ in enumerate(classes)])
        y = torch.cat((pi, y))

        return x, y
