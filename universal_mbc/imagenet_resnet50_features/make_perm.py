import os

import numpy as np
from utils import get_module_root

ROOT = get_module_root()
OUTPATH = f"{root}/universal_mbc/imagenet_resnet50_features/train/"


def main() -> None:
    perm = np.random.permutation(1000)
    train, val = perm[:800], perm[800:]

    np.save(os.path.join(OUTPATH, "train-perm.npy"), train)
    np.save(os.path.join(OUTPATH, "val-perm.npy"), val)


if __name__ == "__main__":
    main()
