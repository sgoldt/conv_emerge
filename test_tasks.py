#!/usr/bin/env python3
"""
Various tests for the tasks.

Author: Sebastian Goldt <goldt.sebastian@gmail.com>

July 2021
"""

import unittest

import torch

import tasks


class InputmodelTests(unittest.TestCase):
    def test_nlgp_mixture(self):
        """
        Check that the class-wise variance matrices of the inputs in (NL)GP mixtures
        agree for NLGP and GP.
        """
        D = 100
        input_names = ["nlgp", "nlgp"]
        xis = [0.1, 0.5]
        gain = 10
        pbc = True
        
        # Create the NL GP mixture
        distributions = tasks.build_nlgp_mixture(input_names, xis, D, pbc, gain)
        task = tasks.Mixture(distributions)

        # Sample
        xs, ys = task.sample(50000)

        # Compute class-wise variance matrices
        nlgp_emp_cov = [None] * 2
        for y in range(2):
            inputs = xs[ys == y]
            nlgp_emp_cov[y] = inputs.T @ inputs / inputs.shape[0]

        # Now let's repeat for GP mixture
        input_names = ["gp", "gp"]

        distributions = tasks.build_nlgp_mixture(input_names, xis, D, pbc, gain)
        task = tasks.Mixture(distributions)

        # Again, sample and compute class-wise covariance
        xs, ys = task.sample(50000)

        gp_emp_cov = [None] * 2
        for y in range(2):
            inputs = xs[ys == y]
            gp_emp_cov[y] = inputs.T @ inputs / inputs.shape[0]

        for y in range(2):
            diff = (torch.sum((nlgp_emp_cov[y] - gp_emp_cov[y]) ** 2)
                    / torch.sum(gp_emp_cov[y] ** 2))
            self.assertTrue(diff < 1e-2)

        # Now let's repeat for GP mixture WITH THE wrong XIS
        xis = [.01, .05]

        distributions = tasks.build_nlgp_mixture(input_names, xis, D, pbc, gain)
        task = tasks.Mixture(distributions)

        # Again, sample and compute class-wise covariance
        xs, ys = task.sample(50000)

        gp_emp_cov = [None] * 2
        for y in range(2):
            inputs = xs[ys == y]
            gp_emp_cov[y] = inputs.T @ inputs / inputs.shape[0]

        for y in range(2):
            diff = (torch.sum((nlgp_emp_cov[y] - gp_emp_cov[y]) ** 2)
                    / torch.sum(gp_emp_cov[y] ** 2))
            self.assertTrue(diff < 1e-2)


if __name__ == "__main__":
    unittest.main()
