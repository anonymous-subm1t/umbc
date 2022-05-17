import argparse
import math

import numpy as np
import torch
from torch import nn

T = torch.Tensor


def main() -> None:
    for name in ["train", "val"]:
        x = torch.from_numpy(np.load("x_train.npy"))
        y = torch.from_numpy(np.load("y_train.npy"))

        centroids, covs = [], []
        for lbl in torch.unique(y):
            print(lbl)
            x_class = x[lbl == y]
            centroid = x_class.mean(dim=0, keepdim=True)
            cov = x_class - centroid
            cov = torch.mm(cov.T, cov) / (lbl == y).sum()

            centroids.append(centroid.squeeze(0))
            covs.append(cov)

        centroids = torch.stack(centroids)  # type: ignore
        covs = torch.stack(covs)  # type: ignore
        np.save(f"{name}/centroids.npy", centroids)
        np.save(f"{name}/covs.npy", covs)


class LogDetter(nn.Module):
    def test_logdet_accuracy(self) -> None:
        for i in range(10):
            x = torch.randn(100, 3)
            analytic = torch.logdet((x.T @ x) / 100)
            svd = self(x)
            assert torch.isclose(analytic, svd, rtol=0, atol=1e-4)
        print("all logdet tests passed")

    def forward(self, x: T) -> T:
        # Lemma: A = USV^T --> A^TA = VS^2V^T --> logdet A^TA = sum(log(V^2))
        logdet = torch.sum(2 * torch.log(torch.linalg.svdvals(x)))

        # the matrix is centered, but is not yet normalized over N (as in E[(x - u)(x - u)^T])
        # Lemma: det(cA) = det(cI) det(A) = c^n det(A) -> logdet(cA) = n log(c) + logdet(A)
        # logdet(x_class) = dim * log(N examples) + log(sum(SVD(x_class) ** 2))
        normalization = x.size(1) * -1 * np.log(x.size(0))
        return logdet + normalization


def precs() -> None:
    for name in ["train", "val"]:
        covs = torch.from_numpy(np.load(f"{name}/covs.npy"))
        precs = []
        for i, c in enumerate(covs):
            print(i)
            prec = torch.inverse(c)
            precs.append(prec)
            if torch.any(torch.isnan(prec)) or torch.any(torch.isinf(prec)):
                print("nan or inf")
        precs = torch.stack(precs).numpy()
        np.save(f"{name}/precs.npy", precs)


def do_logdets(gpu: int) -> None:
    tester = LogDetter()
    tester.test_logdet_accuracy()

    for name in ["train", "val"]:
        logdets = []  # type: ignore
        x = torch.from_numpy(np.load(f"{name}/x.npy"))
        y = torch.from_numpy(np.load(f"{name}/y.npy"))
        centroids = torch.from_numpy(np.load(f"{name}/centroids.npy"))

        logdetter = LogDetter().to(gpu)

        logdets = []
        for i, lbl in enumerate(torch.unique(y)):
            x_class = x[lbl == y] - centroids[lbl].unsqueeze(0)  # (B, D)
            logdet = logdetter(x_class.to(gpu)).cpu()  # type: ignore
            print(i, logdet)
            logdets.append(logdet.unsqueeze(0))

        logdets = torch.cat(logdets)  # type: ignore
        np.save(f"{name}/logdets.npy", logdets)


def log_likelihood() -> None:
    for name in ["train", "val"]:
        centroids = torch.from_numpy(np.load(f"{name}/centroids.npy")).to(0)
        # precs = torch.from_numpy(np.load(f"{name}/precs.npy"))
        # logdets = torch.from_numpy(np.load(f"{name}/logdets.npy"))
        x = torch.from_numpy(np.load(f"{name}/x.npy"))
        y = torch.from_numpy(np.load(f"{name}/y.npy"))
        precs = torch.eye(x.size(1)).unsqueeze(0).to(0)

        _acc, _ll, total = 0.0, 0.0, 0
        for inst, lbl in zip(x, y):
            x_diff = (inst.to(0).unsqueeze(0) - centroids)  # x = (1, D) centroids = (K, D) --> (K, D)

            m = torch.einsum("kd,kde->ke", x_diff, precs)  # (K, D), (K, D, D)
            m = (m * x_diff).sum(dim=-1)  # (K,)
            # logdeterminant of the identity is 0
            ll = -0.5 * np.log(2 * math.pi) - 5 * 0 - 0.5 * m  # (K,)
            _acc += (ll.argmax(dim=0) == lbl).sum().item()
            _ll += torch.logsumexp(ll, dim=-1).item()
            total += 1

        acc, ll = _acc / y.size(0), _ll / y.size(0)
        with open(f"{name}-results.txt", "w+") as f:
            f.write(f"accuracy: {acc} nll: {-ll}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="the GPU number to run on for logets")
    args = parser.parse_args()

    with torch.no_grad():
        # main()
        # precs()
        # do_logdets(args.gpu)
        log_likelihood()
