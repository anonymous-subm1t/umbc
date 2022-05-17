import os
from typing import Tuple

import h5py  # type: ignore
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.datasets import ModelNet  # type: ignore
from torch_geometric.transforms import SamplePoints  # type: ignore
from torchvision import transforms  # type: ignore

T = torch.Tensor
A = np.ndarray


class RotateZ(object):
    def __init__(self, min_rot=-0.1, max_rot=0.1, min_scale=0.8, max_scale=1.25):
        self.min_rot = min_rot
        self.max_rot = max_rot
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, x):
        x = x.unsqueeze(0).numpy()
        bs = 1
        theta = np.random.uniform(self.min_rot, self.max_rot, [1]) * np.pi
        theta = np.expand_dims(theta, 1)
        outz = np.expand_dims(x[:, :, 2], 2)
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        xx = np.expand_dims(x[:, :, 0], 2)
        yy = np.expand_dims(x[:, :, 1], 2)
        outx = cos_t * xx - sin_t * yy
        outy = sin_t * xx + cos_t * yy
        rotated = np.concatenate([outx, outy, outz], axis=2)
        min_scale, max_scale = 0.8, 1.25
        scale = np.random.rand(bs, 1, 3) * (max_scale - min_scale) + min_scale
        rotated_scale = rotated * scale
        return torch.from_numpy(rotated_scale).float().squeeze(0)


class Standardize(object):
    def __init__(self):
        pass

    def __call__(self, x):
        clipper = torch.mean(torch.abs(x.view(-1)), dim=-1, keepdims=True)
        z = torch.clip(x, -100 * clipper.item(), 100 * clipper.item())
        mean = torch.mean(z.view(-1))
        std = torch.std(z.view(-1))
        return (z - mean) / std


class ModelNet40(Dataset):
    def __init__(self, root, num_points=5000, train=True, transform=None):
        self.name = 'ModelNet40'
        self.root = root
        self.train = train
        self.transform = transform
        self.num_points = num_points

        if transform is None:
            tx = {True: [RotateZ(), Standardize()], False: [Standardize()]}[train]
            self.transform = transforms.Compose(tx)

        self.modelnet = ModelNet(root=root, name='40', train=train, transform=SamplePoints(num=num_points))
        self._len = len(self.modelnet)

    def __getitem__(self, index):
        data = self.modelnet[index]
        points, label = data['pos'], data['y']

        if self.transform is not None:
            points = self.transform(points)
        return points, label.item()

    def __len__(self):
        return len(self.modelnet)


def rotate_z2(theta: A, x: A) -> A:
    theta = np.expand_dims(theta, 1)
    outz = np.expand_dims(x[:, 2], 1)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    xx = np.expand_dims(x[:, 0], 1)
    yy = np.expand_dims(x[:, 1], 1)
    outx = cos_t * xx - sin_t * yy
    outy = sin_t * xx + cos_t * yy
    return np.concatenate([outx, outy, outz], axis=1)


def augment2(x: A):
    # rotation, scaling
    thetas = np.random.uniform(-0.1, 0.1, (1,)) * np.pi
    rotated = rotate_z2(thetas, x)
    scale = np.random.rand(1, 3) * 0.45 + 0.8
    return rotated * scale


def standardize2(x: A):
    clipper = np.mean(np.abs(x))
    z = np.clip(x, -100 * clipper, 100 * clipper)
    return (z - np.mean(z)) / np.std(z)


class ModelNet40_10000(Dataset):
    def __init__(self, fname, train=True, down_sample=10, do_standardize=True, do_augmentation=True):
        """this version is made to fit with the current API of our trainers"""
        self.name = 'ModelNet40-10000'
        self.fname = fname
        self.down_sample = down_sample
        self.train = train
        self.num_points = 10000 / down_sample

        with h5py.File(fname, 'r') as f:
            cloud, labels = {True: ('tr_cloud', 'tr_labels'), False: ('test_cloud', 'test_labels')}[train]
            self.data = np.array(f[cloud])
            self.label = np.array(f[labels])

        self.num_classes = np.max(self.label) + 1

        if not train and do_augmentation:
            raise ValueError("cannot do data augmentation on the test set, as do_augmentation=False for testing")

        self.prep1 = standardize2 if do_standardize else lambda x: x
        self.prep2 = (lambda x: augment2(self.prep1(x))) if do_augmentation else self.prep1

        # select the subset of points to use throughout beforehand
        self.perm = np.random.permutation(self.data.shape[1])[::self.down_sample]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.prep2(self.data[index, self.perm])).float(), self.label[index]


def rotate_z(theta: A, x: A) -> A:
    theta = np.expand_dims(theta, 1)
    outz = np.expand_dims(x[:, :, 2], 2)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    xx = np.expand_dims(x[:, :, 0], 2)
    yy = np.expand_dims(x[:, :, 1], 2)
    outx = cos_t * xx - sin_t * yy
    outy = sin_t * xx + cos_t * yy
    return np.concatenate([outx, outy, outz], axis=2)


def augment(x: A) -> A:
    bs = x.shape[0]
    # rotation, scaling
    thetas = np.random.uniform(-0.1, 0.1, [bs, 1]) * np.pi
    rotated = rotate_z(thetas, x)
    scale = np.random.rand(bs, 1, 3) * 0.45 + 0.8
    return rotated * scale


def standardize(x):
    clipper = np.mean(np.abs(x), (1, 2), keepdims=True)
    z = np.clip(x, -100 * clipper, 100 * clipper)
    mean = np.mean(z, (1, 2), keepdims=True)
    std = np.std(z, (1, 2), keepdims=True)
    return (z - mean) / std


class ModelNet40_2048(Dataset):
    def __init__(
        self,
        root: str = "/d1/dataset/ModelNet40-2048/data",
        split: str = "train",
        points: int = 1000,
        standardize: bool = True
    ) -> None:
        """
        adapted from the code here:
        https://github.com/jiachens/ModelNet40-C/blob/300055596e04ce055d81d802a95fb114f9e6a925/rs_cnn/data/ModelNet40Loader.py#L19
        """
        self.split = split
        self.points = points
        self.name = "ModelNet40-2048"

        # load and concatenate the training files here
        x, y = [], []
        files = [v for v in os.listdir(root) if (f"ply_data_{split}" in v and ".h5" in v)]
        for f in files:
            loaded = h5py.File(os.path.join(root, f), "r")
            x.append(loaded["data"][:])
            y.append(loaded["label"][:])

        self.x = np.concatenate(x, 0)
        self.y = np.concatenate(y, 0).squeeze().astype(np.int64)

        # NOTE: DeepSets only augmented the 5000 point experiment
        self.prep1 = standardize2 if standardize else lambda x: x
        self.prep2 = (lambda x: augment2(self.prep1(x))) if split == "train" else self.prep1

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, i: int) -> Tuple[T, T]:
        idx = np.random.permutation(self.x.shape[1])[:self.points]
        return torch.from_numpy(self.prep2(self.x[i, idx])).float(), self.y[i]


class ModelNet40_2048_C(Dataset):
    def __init__(
        self,
        corruption: str,
        severity: int,
        points: int = -1,  # -1 samples all of the points
        standardize: bool = True,
        root: str = "/d1/dataset/ModelNet40-2048-C/data",
    ):
        self.name = "ModelNet40-2048-C"
        self.corruption = corruption
        self.severity = severity
        self.points = points

        file = [v for v in os.listdir(root) if v == f"data_{corruption}_{severity}.npy"]
        if corruption == "original" and severity == 0:
            file = ["data_original.npy"]

        if len(file) != 1:
            raise ValueError(f"there should only be one filtered file: got: ({file})")

        self.x = np.load(os.path.join(root, file[0]))
        self.y = np.load(os.path.join(root, "label.npy")).squeeze().astype(np.int64)
        self.prep = standardize2 if standardize else lambda x: x

    def __getitem__(self, i: int) -> Tuple[T, T]:
        if self.points != -1:
            idx = np.random.permutation(self.x[i].shape[0])[:self.points]
            return torch.from_numpy(self.prep(self.x[i, idx])).float(), self.y[i]
        return torch.from_numpy(self.prep(self.x[i])).float(), self.y[i]

    def __len__(self):
        return self.x.shape[0]


class ModelNetFixedOriginalFromSetTransformer(object):
    def __init__(self, fname, batch_size=64, down_sample=10, do_standardize=True, do_augmentation=True):
        """This was the original dataloader from Set Transformer"""
        self.name = 'MODELNET40_FIXED'
        self.fname = fname
        self.batch_size = batch_size
        self.down_sample = down_sample

        with h5py.File(fname, 'r') as f:
            self._train_data = np.array(f['tr_cloud'])
            self._train_label = np.array(f['tr_labels'])
            self._test_data = np.array(f['test_cloud'])
            self._test_label = np.array(f['test_labels'])

        self.num_classes = np.max(self._train_label) + 1

        self.num_train_batches = len(self._train_data) // self.batch_size
        self.num_test_batches = len(self._test_data) // self.batch_size

        self.prep1 = standardize if do_standardize else lambda x: x
        self.prep2 = (lambda x: augment(self.prep1(x))) if do_augmentation else self.prep1

        assert len(self._train_data) > self.batch_size, \
            'Batch size larger than number of training examples'

        # select the subset of points to use throughout beforehand
        self.perm = np.random.permutation(self._train_data.shape[1])[::self.down_sample]
        print(self.perm)

    def train_data(self):
        rng_state = np.random.get_state()
        np.random.shuffle(self._train_data)
        np.random.set_state(rng_state)
        np.random.shuffle(self._train_label)
        return self.next_train_batch()

    def next_train_batch(self):
        start = 0
        end = self.batch_size
        N = len(self._train_data)
        perm = self.perm
        # batch_card = len(perm) * np.ones(self.batch_size, dtype=np.int32)
        while end < N:
            yield torch.from_numpy(self.prep2(self._train_data[start:end, perm])).float(), torch.from_numpy(self._train_label[start:end]).long()
            start = end
            end += self.batch_size

    def test_data(self):
        return self.next_test_batch()

    def next_test_batch(self):
        start = 0
        end = self.batch_size
        N = len(self._test_data)
        # batch_card = (self._train_data.shape[1] // self.down_sample) * np.ones(self.batch_size, dtype=np.int32)
        while end < N:
            yield torch.from_numpy(self.prep1(self._test_data[start:end, 1::self.down_sample])).float(), torch.from_numpy(self._test_label[start:end]).long()
            start = end
            end += self.batch_size
