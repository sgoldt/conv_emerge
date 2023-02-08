# -*- coding: utf-8 -*-
"""Samplers for input distributions.

Date: June 2021

Authors: Sebastian Goldt <goldt.sebastian@gmail.com>
         Alessandro Ingrosso <aingrosso@ictp.
"""

from abc import ABCMeta, abstractmethod
import math
import os

import numpy as np
import scipy
from scipy.stats import multivariate_normal

import torch

import utils


def get_gabor(s, th, f, i, j):
    g = torch.exp(-0.5 * (i ** 2 + j ** 2) / s ** 2) * torch.cos(
        2 * np.pi * f * (i * np.cos(th) + j * np.sin(th))
    )
    return g / torch.sqrt((g ** 2).sum())


def trans_inv_var(
    D, torus=True, p=1, xi=1e-1, perturbation=1e-3, dim=1, xi_pow_pi=True
):
    """Returns a translation-invariant covariance matrix

        exp(−|i−j|^p / xi^{(1-xi_pow_pi) * p})

    which, for p=1, is the covariance of a 1D Ising model with correlation
    length xi corresponding to the given temperature.

    Parameters:
    -----------
    D : linear input size
    p : power of the distance in the exponent
    xi : correlation length
    perturbation : add this to diagonal elements of the matrix for numerical stability
    dim : input dimension
    """
    positions = torch.arange(D, dtype=torch.float)
    distances = positions[None] - positions[:, None]
    # the next line enforces the periodic boundary conditions
    if torus:
        distances = torch.minimum(torch.abs(distances), D - torch.abs(distances))

    xicov = xi ** p if xi_pow_pi else xi
    covariance = torch.exp(-torch.abs(distances) ** p / xicov)
    if dim == 2:
        # while torch now has a kron function, backwards compatibility imposes
        # that we use numpy's function
        covariance = torch.tensor(np.kron(covariance, covariance))
    # for numerical stability, add a little kick on the diagonal
    covariance += perturbation * torch.eye(*covariance.shape)

    return covariance


class InputModel(metaclass=ABCMeta):
    """
    Abstract class for all the data models used in these experiments.
    """

    _input_dim = None

    @property
    def input_dim(self):
        """
        Dimension of the vectors for 1D models, width of inputs for 2D models.
        """
        return self._input_dim

    @abstractmethod
    def sample(self, P):
        """
        Samples P samples from the data model.
        """


class GaussianProcess(InputModel):
    """
    Gaussian process with given mean and covariance matrix.

    """

    def __init__(self, covariance, mean=None):
        """
        Parameters:
        -----------
        covariance, mean : (D, D), (D)
            mean and covariance of the Gaussian process
        """
        super().__init__()

        self._input_dim = covariance.shape[0]
        self._mean = np.zeros(self._input_dim) if mean is None else mean.numpy()
        self._covariance = covariance.numpy()

    def sample(self, P=1):
        xs = torch.tensor(
            multivariate_normal.rvs(mean=self._mean, cov=self._covariance, size=P),
            dtype=torch.float32,
        )

        return xs

    def get_dataset(
        self, train=True, P=None, dtype=torch.float, device=torch.device("cpu")
    ):
        if P is None:
            raise ValueError("This class generates dataset online. Pleaser provide P.")
        return self.sample(P)

    def covariance(self):
        return self._covariance


class NLGP(GaussianProcess):
    """Non-linear Gaussian process

        z = f(gain x / sqrt(2)),

    where x is a Gaussian process with given mean and covariance,
    f is a non-linear function and gain is a scalar gain factor.

    The class is designed such that the scalar mean and covariance of the inputs is
    unchanged compared to the original Gaussian process.

    """

    def __init__(self, f, covariance, mean=None, gain=1):
        """
        Parameters:
        -----------
        f : erf
            a string indicating the nonlinearity used
        D (int):
            input dimension
        mean, covariance : (D), (D, D)
            mean and covariance of the Gaussian process
        g : double
            gain factor
        """
        super().__init__(covariance, mean)
        # by calling the constructor of the parent class, you initialise the Gaussian
        # process underlying this nonlinear GP.

        self.f = f
        self.gain = gain

        if self.f == "erf":
            self._act = utils.get_act(f)
            self._gain_pref = self.gain ** 2 / (1 + self.gain ** 2)
            self._gamma = 2 / math.pi * math.asin(self._gain_pref)
        else:
            raise NotImplementedError(f + " nonlinearity not yet implemented.")

    def sample(self, P=1):
        xs = super().sample(P)
        xs = self.gain * xs
        xs = 1 / math.sqrt(self._gamma) * self._act(xs)

        return xs

    def get_dataset(
        self, train=True, P=None, dtype=torch.float, device=torch.device("cpu")
    ):
        if P is None:
            raise ValueError("This class generates dataset online. Pleaser provide P.")
        return self.sample(P)

    def covariance(self):
        """
        Returns the analytical form of the covariance of this process,
        or None if it is not (yet) available.
        """
        covariance = None
        if self.f == "erf":
            covariance = (
                2
                / math.pi
                / self._gamma
                * torch.asin(self._gain_pref * torch.tensor(super().covariance()))
            )
        else:
            raise NotImplementedError("Have not implemented this non-linearity yet.")

        return covariance


class Ising(InputModel):
    """Ising model at the given temperature."""

    def __init__(
        self,
        dim=1,
        N=50,
        T=1,
        load_dir="data/ISING",
        num_steps_eq=1000,
        sampling_rate=10,
        seed=1,
    ):

        super().__init__()

        self.dim = dim
        self._input_dim = N

        if dim == 1:
            self.size_state = (N,)
            self.energy = self.energy_1d
            self.mcmove = self.mcmove_1d
        else:
            self.size_state = (N, N)
            self.energy = self.energy_2d
            self.mcmove = self.mcmove_2d

        self.N = N
        self.T = T
        self.beta = 1 / T
        self.load_dir = load_dir
        # online options
        self.seed = seed
        self.num_steps_eq = num_steps_eq
        self.sampling_rate = sampling_rate
        self.burnt = False

        # The covariance matrix is computed in its getter
        self._covariance = None

    def get_dataset(
        self,
        train=True,
        no_duplicates=True,
        P=None,
        dtype=torch.float,
        device=torch.device("cpu"),
    ):

        dataset_type = "train" if train else "_test"
        suffix = "" if train else "_test"
        filename = f"{self.load_dir}/ising_D{self.dim}_N{self.N}_T{self.T}"
        filename += "_nodupl" if no_duplicates else ""
        filename += f"{suffix}.npz"
        print(f"...will read {dataset_type} data from {filename}")
        container = np.load(filename)
        data = [container[key] for key in container]
        X, _, _, _ = data
        if P is not None:
            if P > len(X):
                raise ValueError("Not enough data in the stored dataset")
            X = X[:P]
        return torch.tensor(X, dtype=dtype, device=device)

    def sample(self, P=1):
        if not self.burnt:
            self.init_and_burn()
            print(f"chain T {self.T} burnt")
            self.burnt = True
        i_rec = 0
        states = torch.zeros((P, self.N))
        for it in range(1, P * self.sampling_rate + 1):
            self.mcmove()
            if it % self.sampling_rate == 0:
                states[i_rec] = self.state
                i_rec += 1
        return states

    def covariance(self):
        """
        Returns the covariance matrix of this Ising model
        """
        if self._covariance is None:
            # compute the covariance matrix
            positions = torch.arange(self.N, dtype=torch.float)
            distances = positions[None] - positions[:, None]
            # the next line enforces the periodic boundary conditions
            distances = torch.minimum(
                torch.abs(distances), self.N - torch.abs(distances)
            )

            xi = utils.getCorrelationLength(self.T)
            self._covariance = torch.exp(-torch.abs(distances) / xi)

        return self._covariance

    # markov chain methods
    def init_and_burn(self):
        self.state = 2 * torch.randint(2, size=self.size_state) - 1
        self.energy()
        for _ in range(self.num_steps_eq):
            self.mcmove()

    def mcmove_1d(self):
        i = np.random.randint(self.N)
        s_i = self.state[i]
        s_nb = self.state[(i + 1) % self.N] + self.state[(i - 1) % self.N]
        ΔE = 2 * s_i * s_nb
        if ΔE < 0 or np.random.rand() < np.exp(-self.beta * ΔE):
            self.state[i] = -s_i
            self.E += ΔE

    def energy_1d(self):
        self.E = 0.0
        for i in range(self.N):
            s_nb = self.state[(i + 1) % self.N] + self.state[(i - 1) % self.N]
            self.E -= s_nb * self.state[i]

    def mcmove_2d(self):
        i, j = np.random.randint(0, self.N, size=2)
        s_i = self.state[i, j]
        s_nb = (
            self.state[(i + 1) % self.N, j]
            + self.state[i, (j + 1) % self.N]
            + self.state[(i - 1) % self.N, j]
            + self.state[i, (j - 1) % self.N]
        )
        ΔE = 2 * s_i * s_nb
        if ΔE < 0 or np.random.rand() < np.exp(-self.beta * ΔE):
            self.state[i, j] = -s_i
            self.E += ΔE

    def energy_2d(self):
        self.E = 0.0
        for i in range(self.N):
            for j in range(self.N):
                s_nb = (
                    self.state[(i + 1) % self.N, j]
                    + self.state[i, (j + 1) % self.N]
                    + self.state[(i - 1) % self.N, j]
                    + self.state[i, (j - 1) % self.N]
                )
                self.E -= s_nb * self.state[i, j]


class Phi4(InputModel):
    def __init__(
        self,
        dim=1,
        D=100,
        lambd=2,
        musq=-6,
        zscore=False,
        normalize=False,
        equilibrate=200,
        sampling_rate=100,
        buffer_size=1000,
        save_dir="phi4configs",
        suffix="_train_online",
        load_dir="data/PHI4",
    ):

        # general options
        self.dim = dim
        self.D = D
        self.lambd = lambd
        self.musq = musq
        self.zscore = zscore
        self.normalize = normalize
        # offline options
        self.load_dir = load_dir
        # online options
        self.equilibrate = equilibrate
        self.sampling_rate = sampling_rate
        self.buffer_size = buffer_size
        self.suffix = suffix
        self.save_dir = save_dir
        self.buffer = np.empty((0, self.D))
        self.tot_iter = self.buffer_size * self.sampling_rate

    def input_dim(self):
        return self.D

    def gen_dataset(self, P=100):
        command = f"phi4/phi4_1d.exe {self.musq} {self.lambd} {self.D} {P * self.sampling_rate} {self.sampling_rate} {self.equilibrate} {self.save_dir} {self.suffix}"
        os.system(command)

    def get_dataset(
        self, train=True, P=None, dtype=torch.float, device=torch.device("cpu"),
    ):
        dataset_type = "train" if train else "test"
        filename = f"{self.load_dir}/configs_N{self.D}_m{self.musq}_l{self.lambd}_{dataset_type}.txt"
        print(f"...will read {dataset_type} data from {filename}")
        X = np.loadtxt(filename)
        if P is not None:
            if P > len(X):
                raise ValueError("Not enough data in the stored dataset")
            X = X[:P]
        X = scipy.stats.zscore(X, axis=1) if self.zscore else X
        if self.normalize:
            normX = np.sqrt((X ** 2).sum(-1))[:, None]
            X /= normX
        return torch.tensor(X, dtype=dtype, device=device)

    def sample(self, P=1, dtype=torch.float, device=torch.device("cpu")):
        # load buffer
        if len(self.buffer) == 0:
            # run C++ code and store results in the buffer
            # print("...loading phi4 buffer")
            command = f"phi4/phi4_1d.exe {self.musq} {self.lambd} {self.D} {self.tot_iter} {self.sampling_rate} {self.equilibrate} {self.save_dir} {self.suffix}"
            # print(command)
            os.system(command)
            self.buffer = np.loadtxt(
                f"{self.save_dir}/configs_N{self.D}_m{self.musq}_l{self.lambd}{self.suffix}.txt"
            )

        # retrieve config and pop buffer
        X = self.buffer[-1][None]
        X = scipy.stats.zscore(X, axis=1) if self.zscore else X
        self.buffer = self.buffer[:-1]
        return torch.tensor(X, dtype=dtype, device=device)
