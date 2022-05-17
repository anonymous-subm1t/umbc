import os

import numpy as np
from utils import random_features, get_module_root


def main() -> None:
    # make_random features, open the x file and make an ortho projection
    rf = random_features(rows=2048, cols=512, orthogonal=True, random_norm=True).numpy()

    root = get_module_root()
    path = f"{root}/universal_mbc/imagenet_resnet50_features/"
    train_x = np.load(os.path.join(path, "train/x.npy"))
    out = train_x @ rf
    np.save(os.path.join(path, "train/x-ort.npy"), out)

    del train_x
    del out

    val_x = np.load(os.path.join(path, "val/x.npy"))
    out = val_x @ rf
    np.save(os.path.join(path, "val/x-ort.npy"), out)


if __name__ == "__main__":
    main()
