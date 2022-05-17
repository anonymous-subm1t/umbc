import unittest

import numpy as np  # type: ignore
import torch
from torch import nn

from utils import ece, ece_partial, ece_partial_final

T = torch.Tensor


class TestECE(unittest.TestCase):
    def test_partial_equals_regular_one_pass(self) -> None:
        logits = torch.rand(1000, 10)
        y = torch.randint(0, 10, (1000,))

        cal_error, _, _ = ece(y, logits)
        c, a, n_in_bins, n = ece_partial(y , logits)
        cal_error2 = ece_partial_final(c, a, n_in_bins, n)
        self.assertEqual(cal_error, cal_error2)

    def test_partial_equals_regular_partial_pass(self) -> None:
        # test equality for multple passes
        for i in range(10):
            logits = torch.rand(1000, 100)
            y = torch.randint(0, 100, (1000,))
            cal_error, _, _ = ece(y, logits)

            conf, acc, n_in_bins, n = torch.zeros(15), torch.zeros(15), torch.zeros(15), 0

            sections = torch.randperm(999)[:10].tolist()
            sections = [i for i in sections if i != 0]
            sections.sort()

            sections = [0] + sections + [1000]
            for i, sec in enumerate(sections[:-1]):
                y_ = y[sec : sections[i + 1]]
                logits_ = logits[sec : sections[i + 1]]
                c, a, nin, _n = ece_partial(y_, logits_)

                conf += c
                acc += a
                n_in_bins += nin
                n += _n

            cal_error2 = ece_partial_final(conf, acc, n_in_bins, n)
            self.assertAlmostEqual(cal_error, cal_error2, 3)
