# -*- coding: utf-8 -*-
"""Unit tests for the utility methods used throughout the project.

Date: June 2021

Authors: Alessandro Ingrosso <aingrosso@ictp.com>
         Sebastian Goldt <goldt.sebastian@gmail.com>
"""

import math
import unittest

import torch

import utils


class UtilsTest(unittest.TestCase):
    def test_correlation_length(self):
        # For the temperatures T= 1, 2, 3, the correlation lengths are given by
        Ts = torch.tensor([1, 2, 3])
        xi_true = torch.tensor([3.6719, 1.2954, 0.8813])

        # check the scalar case
        xi = utils.getCorrelationLength(2)
        msg = "Correlation length incorrect for scalar"
        self.assertTrue(xi - xi_true[1] < 1e-4, msg)

        # check the numpy array case
        xis = utils.getCorrelationLength(Ts.numpy())
        msg = "Correlation length incorrect for numpy array"
        self.assertTrue(torch.sum((xis - xi_true)**2) < 1e-4, msg)

        # check the Torch tensor
        xis = utils.getCorrelationLength(Ts)
        msg = "Correlation length incorrect for pytorch tensor"
        self.assertTrue(torch.sum((xis - xi_true)**2) < 1e-6, msg)


if __name__ == "__main__":
    unittest.main()
