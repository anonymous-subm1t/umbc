import os
import unittest
from argparse import Namespace
from functools import partial
from typing import Any, Dict, List

import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
import torch
from matplotlib import pyplot as plt  # type: ignore

from data.get import get_imagenet_clusters, get_imagenet_features
from data.modelnet import (ModelNet40, ModelNet40_2048, ModelNet40_2048_C,
                           ModelNet40_10000)
from data.toy_clustering import MixtureOfGaussians

DATA_ROOT = os.environ.get("DATA_ROOT", "")

class DatasetTests(unittest.TestCase):
    def test_dataset_imagenet_feature_clustering(self) -> None:
        clusters, batch = 4, 8
        train, _, val = get_imagenet_clusters(Namespace(clusters=clusters, batch_size=batch, num_workers=4))

        for ds in [train, val]:
            while True:
                for x, y in ds:
                    print(f"out: {x.size()=} {y.size()=}")

            self.assertTrue(all((u == v for (u, v) in zip(x.size(), [batch, 1024, 2048]))))
            self.assertEqual(y.size(0), batch)
            self.assertEqual(y.size(1), 1024 + clusters)

    def test_dataset_imagenet_features(self) -> None:
        args = Namespace(batch_size=32, num_workers=4)
        train, val, test = get_imagenet_features(args)
        for ds in train, val, test:
            for x, y in ds:
                break

            self.assertTrue(all([u == v for u, v in zip(x.size(), [32, 2048])]))
            self.assertEqual(y.size(0), 32)

    def test_dataset_modelnet(self) -> None:
        d_sets = [
            partial(ModelNet40, root="/d1/dataset/ModelNet40", train=True),
            partial(ModelNet40, root="/d1/dataset/ModelNet40", train=False),
            partial(ModelNet40_10000, "/d1/dataset/ModelNet40-10000/ModelNet40_cloud.h5", down_sample=2, train=True),
            partial(ModelNet40_10000, "/d1/dataset/ModelNet40-10000/ModelNet40_cloud.h5", down_sample=2, train=False, do_augmentation=False),
            partial(ModelNet40_2048, split="train", points=1000),
            partial(ModelNet40_2048, split="test", points=1000),

        ]
        expected_points = [5000, 5000, 5000, 5000, 1000, 1000]

        for ds, exp_pnt in zip(d_sets, expected_points):
            _ds = ds()  # type: ignore
            x, y = _ds[0]
            self.assertTrue(all((u == v for (u, v) in zip(x.size(), (exp_pnt, 3)))))

        d_sets = []
        for corr in [
            "background", "cutout", "density", "density_inc", "distortion",
            "distortion_rbf", "distortion_rbf_inv", "gaussian", "impulse", "lidar",
            "occlusion", "rotation", "shear", "uniform", "upsampling"
        ]:
            for severity in [1, 2, 3, 4, 5]:
                d_sets.append(partial(ModelNet40_2048_C, corr, severity))

        for ds in d_sets:
            _ds = ds()  # type: ignore
            x, y = _ds[0]

    def test_dataset_mixture_of_gaussians(self) -> None:
        B, K, D = 10, 4, 2
        for mvn_type in ["diag", "full"]:
            ds = MixtureOfGaussians(dim=D, mvn_type=mvn_type)

            fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(7 * 5, 6 * 2))
            axes = axes.flatten()

            X, labels, pi, mu, sigma = ds.sample(B, 500, K)

            # diag network will only give the diag so we need to extract the diagonal here.
            if mvn_type == "diag":
                sigma = torch.diagonal(sigma, dim1=-2, dim2=-1)
            else:
                sigma = sigma.view(B, K, -1)

            raw = torch.cat((pi.unsqueeze(-1), mu, sigma), dim=-1)
            pi, mu, sigma, logdet = ds.parse(raw)
            log_prob, _ = ds.log_prob(X, pi, mu, sigma, logdet)

            for (x, y, ax) in zip(X, labels, axes):
                data = pd.DataFrame({"x": [v[0].item() for v in x], "y": [v[1].item() for v in x], "class": [str(v.item()) for v in y]})
                sns.scatterplot(data=data, x="x", y="y", hue="class", ax=ax)

            path = os.path.join("data/examples/mixture-of-gaussians")
            os.makedirs(path, exist_ok=True)
            fig.tight_layout()
            fig.savefig(os.path.join(path, f"{mvn_type}-example.png"))
            fig.savefig(os.path.join(path, f"{mvn_type}-example.pdf"))
