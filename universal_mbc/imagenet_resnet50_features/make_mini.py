import numpy as np


def main() -> None:
    """make a mini version of the datasets for x and why which will load faster for debugging"""

    x, y = np.load("train/x.npy"), np.load("train/y.npy")
    idx = np.random.permutation(x.shape[0])[:50000]
    x, y = x[idx], y[idx]

    np.save("train/x-mini.npy", x)
    np.save("train/y-mini.npy", y)


if __name__ == "__main__":
    main()
