import math
import os
from argparse import Namespace
from functools import partial
from typing import Any, List, Tuple

import numpy as np  # type: ignore
import torch
import torchvision  # type: ignore
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms  # type: ignore
from torchvision.datasets import CIFAR10, CIFAR100, MNIST  # type: ignore
from torchvision.transforms import Compose  # type: ignore
from torchvision.transforms.functional import InterpolationMode  # type: ignore

from data.imagenet import ImageNetClusters
from data.modelnet import (ModelNet40, ModelNet40_2048, ModelNet40_2048_C,
                           ModelNet40_10000)
from data.toy_clustering import MixtureOfGaussians

T = torch.Tensor


def get_mixture_of_gaussians(args: Namespace) -> Tuple[DataLoader, ...]:
    train = MixtureOfGaussians(dim=args.dim, mvn_type=args.mvn_type)
    loader = DataLoader(train, batch_size=1, shuffle=True)
    return loader, loader, loader


def get_imagenet_clusters(args: Namespace, mini: bool = False) -> Tuple[DataLoader, ...]:
    basedir = "/d1/dataset/umbc/universal_mbc"
    PATH = f"{basedir}/imagenet_resnet50_features/train/"

    set_size = 1024

    train_x = np.load(os.path.join(PATH, "x-ort.npy"))
    # train_x = np.load(os.path.join(PATH, "x.npy"))
    train_y = np.load(os.path.join(PATH, "y.npy"))

    train_perm = np.load(os.path.join(PATH, "train-perm.npy"))
    train_idx = np.zeros(train_y.shape[0])
    for k in train_perm:
        train_idx = np.logical_or(train_idx, (k == train_y))

    val_perm = np.load(os.path.join(PATH, "val-perm.npy"))
    val_idx = np.zeros(train_y.shape[0])
    for k in val_perm:
        val_idx = np.logical_or(val_idx, (k == train_y))

    train = ImageNetClusters(torch.from_numpy(train_x[train_idx]), torch.from_numpy(train_y[train_idx]), args.clusters, set_size=set_size)
    val = ImageNetClusters(torch.from_numpy(train_x[val_idx]), torch.from_numpy(train_y[val_idx]), args.clusters, set_size=set_size)

    train_ldr = DataLoader(train, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    val_ldr = DataLoader(val, shuffle=True, batch_size=args.batch_size, num_workers=args.batch_size, pin_memory=True)
    return train_ldr, train_ldr, val_ldr


def get_imagenet_features(args: Namespace) -> Tuple[DataLoader, ...]:
    basedir = "/d1/dataset/umbc/universal_mbc"
    train_x = np.load(f"{basedir}/imagenet_resnet50_features/train/x.npy")
    train_y = np.load(f"{basedir}/imagenet_resnet50_features/train/y.npy")

    val_x = np.load(f"{basedir}/imagenet_resnet50_features/val/x.npy")
    val_y = np.load(f"{basedir}/imagenet_resnet50_features/val/y.npy")

    train = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    val = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))

    train_ldr = DataLoader(val, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    val_ldr = DataLoader(train, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    return train_ldr, train_ldr, val_ldr


def get_modelnet(args: Namespace) -> Tuple[DataLoader, ...]:
    train = ModelNet40(root=args.data_root, train=True)
    test = ModelNet40(root=args.data_root, train=False)

    train_ldr = DataLoader(train, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    test_ldr = DataLoader(test, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    return train_ldr, train_ldr, test_ldr


def get_modelnet_fixed(args: Namespace) -> Tuple[DataLoader, ...]:
    ds = {100: 100, 1000: 10, 5000: 2, 10000: 1}[args.set_size]

    train = ModelNet40_10000(f"{args.data_root}/ModelNet40_cloud.h5", down_sample=ds, train=True, do_augmentation=True)
    test = ModelNet40_10000(f"{args.data_root}/ModelNet40_cloud.h5", down_sample=1, train=False, do_augmentation=False)

    train_ldr = DataLoader(train, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    test_ldr = DataLoader(test, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    return train_ldr, train_ldr, test_ldr


def get_modelnet_2048(args: Namespace) -> Tuple[DataLoader, ...]:
    train = ModelNet40_2048(root=args.data_root, split="train", points=args.set_size, standardize=True)
    test = ModelNet40_2048(root=args.data_root, split="test", points=2048, standardize=True)

    train_ldr = DataLoader(train, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    test_ldr = DataLoader(test, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    return train_ldr, train_ldr, test_ldr


def get_modelnet_2048_c(args: Namespace) -> Tuple[DataLoader, ...]:
    train = ModelNet40_2048_C(corruption="original", severity=0, points=args.set_size, standardize=True)
    test = ModelNet40_2048_C(corruption=args.corruption, severity=args.severity, points=args.set_size, standardize=True)

    train_ldr = DataLoader(train, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    test_ldr = DataLoader(test, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    return train_ldr, train_ldr, test_ldr


deref = {
    "modelnet": get_modelnet,
    "modelnet-fixed": get_modelnet_fixed,
    "modelnet-2048": get_modelnet_2048,
    "modelnet-2048-c": get_modelnet_2048_c,
    "imagenet-features": get_imagenet_features,
    "imagenet-clusters": get_imagenet_clusters,
    "toy-mixture-of-gaussians": get_mixture_of_gaussians,
}


def get_dataset(args: Namespace, **kwargs: Any) -> Tuple[DataLoader, ...]:
    if args.dataset not in deref.keys():
        raise NotImplementedError(f"dataset: {args.dataset} is not implemented")

    return deref[args.dataset](args, **kwargs)  # type: ignore


def get_dataset_by_name(args: Namespace, name: str = "", **kwargs: Any) -> Tuple[DataLoader, ...]:
    if name not in deref.keys():
        raise NotImplementedError(f"dataset: {name} is not implemented")

    return deref[name](args, **kwargs)  # type: ignore


def plot_samples(x: T, args: Namespace) -> None:
    path = os.path.join("data", "examples")
    os.makedirs(path, exist_ok=True)
    grid = torchvision.utils.make_grid(x, nrows=5)
    torchvision.utils.save_image(grid, os.path.join(path, f"{args.dataset}-example.png"))
