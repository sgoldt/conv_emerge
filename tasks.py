# -*- coding: utf-8 -*-
"""Samplers for tasks.

This module implements samplers for various machine learning tasks, which define
distributions over input-output pairs (x, y).

Date: June 2021

Authors: Sebastian Goldt <goldt.sebastian@gmail.com>
"""

from abc import ABCMeta, abstractmethod
import torch

import inputs


class Task(metaclass=ABCMeta):
    """
    Abstract class for all the tasks used in these experiments.
    """

    @abstractmethod
    def input_dim(self):
        """
        Dimension of the vectors for 1D models, width of inputs for 2D models.
        """

    @abstractmethod
    def get_dataset(
        self, train=True, P=None, dtype=torch.float, device=torch.device("cpu"),
    ):
        """
        Retrieve stored dataset.

        Parameters:
        -----------
        train : True for training data set, else a test data set is loaded.
        P : number of samples
        """

    @abstractmethod
    def sample(self, P):
        """
        Samples P samples from the task.

        Returns:
        --------

        xs : (P, D)
             P inputs in D dimensions
        ys : (P)
             P labels
        """


class Mixture(Task):
    """
    Mixture of distributions, one for each label.
    """

    def __init__(self, distributions):
        """
        Parameters:
        -----------

        distributions: array
            a set of inputs, each of which will correspond to one label.
        """
        super().__init__()

        self.distributions = distributions
        self.num_classes = len(distributions)

        self.D = self.distributions[0].input_dim

    def input_dim(self):
        return self.D

    def get_dataset(
        self, train=True, P=None, dtype=torch.float, device=torch.device("cpu"),
    ):
        num_p = P // self.num_classes
        dataset_desc = "training" if train else "testing"
        print(f"will generate {dataset_desc} set with {num_p} patterns per class")
        X = torch.empty((0, self.D), dtype=dtype, device=device)
        y = torch.empty(0, dtype=dtype, device=device)
        for p, distribution in enumerate(self.distributions):
            Xtemp = distribution.get_dataset(
                train=train, P=num_p, dtype=dtype, device=device
            )
            X = torch.cat([X, Xtemp])
            y = torch.cat([y, p * torch.ones(len(Xtemp), dtype=dtype, device=device)])

        return X, y

    def sample(self, P=1):
        ys = torch.randint(self.num_classes, (P,))
        xs = torch.zeros(P, self.D)

        for m in range(self.num_classes):
            num_samples = torch.sum(ys == m).item()
            xs[ys == m] = self.distributions[m].sample(num_samples)

        return xs, ys

    def __str__(self):
        dist_names = [str(dist) for dist in self.distributions]
        name = "_".join(dist_names)
        return name


def build_nlgp_mixture(
    input_names, xis, D, torus, gain, dim=1, xi_pow_pi=True, perturbation=1e-3
):
    """
    Constructs a mixture of distributions (NLGP / GP) with the given xis and gain.
    """
    distributions = [None] * len(input_names)

    for idx, input_name in enumerate(input_names):
        # create the covariance for the given correlation length
        xi = xis[idx]

        covariance = inputs.trans_inv_var(
            D,
            torus=torus,
            p=2,
            xi=xi,
            perturbation=perturbation,
            dim=dim,
            xi_pow_pi=xi_pow_pi,
        )
        # create the non-linear GP
        nlgp = inputs.NLGP("erf", covariance, gain=gain)
        if input_name == "gp":
            # create a Gaussian process with the same covariance
            distributions[idx] = inputs.GaussianProcess(nlgp.covariance())
        elif input_name == "nlgp":
            distributions[idx] = nlgp
        else:
            raise ValueError("Did not recognise input name (gp | nlgp)")

    return distributions
