import os
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models  # type: ignore
from torchvision import datasets, transforms
from tqdm import tqdm  # type: ignore

IMAGENET_PATH = "/st1/dataset/imagenet1k/raw-data/"
SAVEDIR = "imagenet_resnet50_features"

T = torch.Tensor


def setup_imagenet_data_loader() -> Tuple[DataLoader, ...]:
    n_worker = 16
    valdir = os.path.join(IMAGENET_PATH, 'val')
    traindir = os.path.join(IMAGENET_PATH, 'train')
    # transforms for a resnet

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # use the validation transforms for all of these since we do not care about regularization of the model, we just
    # want the embedded features of the pretrained model
    train_dataset = datasets.ImageFolder(traindir, val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=500, shuffle=False, num_workers=n_worker)

    val_dataset = datasets.ImageFolder(valdir, val_transforms)
    test_loader = DataLoader(val_dataset, batch_size=500, shuffle=False, num_workers=n_worker)

    return train_loader, test_loader


class ResNet50(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        self.fc = self.resnet.fc
        self.resnet.fc = nn.Identity()

        self.register_buffer("mean", torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, x: T) -> T:
        x = (x - self.mean) / self.std
        return self.resnet(x)  # type: ignore


def extract() -> None:
    device = torch.device("cuda:0")
    model = nn.DataParallel(ResNet50().to(device), device_ids=[0, 1, 2, 3, 4, 5])
    train_set, val_set = setup_imagenet_data_loader()

    model.eval()
    for ds, name in zip((train_set, val_set), ("train", "val")):
        path = os.path.join(SAVEDIR, name)
        if not os.path.exists(path):
            os.makedirs(path)

        xs, ys = [], []
        with torch.no_grad():
            for (x, y) in tqdm(ds, total=len(ds)):
                ft = model(x.to(device)).cpu()
                xs.append(ft)
                ys.append(y)

        xs, ys = torch.cat(xs).numpy(), torch.cat(ys).numpy()
        np.save(f"{SAVEDIR}/{name}/x.npy", xs)
        np.save(f"{SAVEDIR}/{name}/y.npy", ys)


if __name__ == "__main__":
    extract()
