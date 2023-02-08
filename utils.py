# -*- coding: utf-8 -*-
"""Helper methods used throughout the project.

Date: June 2021

Authors: Alessandro Ingrosso <aingrosso@ictp.com>
         Sebastian Goldt <goldt.sebastian@gmail.com>
"""

from collections import OrderedDict

import math

import numpy as np
from numpy.fft import fft, ifft
from scipy.interpolate import UnivariateSpline, CubicSpline

import torch
import torch.nn.functional as F

sq2 = math.sqrt(2)


############ GENERAL FUNCTIONS ############

activations = {
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "erf": lambda x: torch.erf(x / sq2),
    "erf+": lambda x: 0.5 * (1 + torch.erf(x / sq2)),
    "relu": F.relu,
    "linear": lambda x: x,
}


def get_act(act_type):
    return activations[act_type]


def kurtosis(weights, dim=-1, excess=True):
    """
    Parameters:
    -----------
    weights (...) : tensor
    dim: dimension over which the kurtosis should be measured, default=-1
    excess : if True, compute the excess kurtosis. Default is True.
    """
    D = weights.shape[dim]
    # The D prefactor comes from interpreting the sums as empirical means of the fourth-order moment
    # divided by the (empirical est of the second-order moment)^2
    kurt = D * weights.pow(4).sum(dim=dim) / weights.pow(2).sum(dim=dim)**2
    if excess:
        kurt -= 3
    return kurtosis


class BatchGenerator:
    def __init__(self, D, samplers, batch_size):
        self.D = D
        self.samplers = samplers
        self.batch_size = batch_size
        self.M = len(samplers)

    def sample(self, P=1):
        ys = torch.randint(self.M, (self.batch_size,))
        xs = torch.zeros((self.batch_size, self.D))
        for iy, y in enumerate(ys):
            xs[iy] = self.samplers[y].sample(P=P)[0]
        return xs, ys


def get_dict(net, to_numpy=False):
    if to_numpy:
        return OrderedDict(
            {k: v.detach().clone().to("cpu").numpy() for k, v in net.state_dict().items()}
        )
    else:
        return OrderedDict(
            {k: v.detach().clone().to("cpu") for k, v in net.state_dict().items()}
        )



def get1hot(ys, num_classes):
    """
    Transform an array with class labels into an array with one-hot encodings of
    these classes.
    """
    ys1hot = ys.unsqueeze(-1) == torch.arange(num_classes).reshape(1, num_classes)
    return ys1hot.float()


def getCorrelationLength(Ts):
    """
    Returns the correlation length of an Ising model at the given temperature.

    The correlation length xi is defined such that the covariance between spins is
    given by

        E x_i x_j = exp(−|i−j|/xi).

    Parameters:
    -----------
    Ts : the temperature(s), either as scalar, numpy array or pyTorch tensor.
    """
    # Turn Ts into a tensor
    if type(Ts) in [float, int]:
        Ts = torch.tensor([Ts])
    elif isinstance(Ts, np.ndarray):
        Ts = torch.from_numpy(Ts)
    elif isinstance(Ts, list):
        Ts = torch.tensor(Ts)

    return -1 / torch.log(torch.tanh(1.0 / Ts))


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def is_singular(A):
    return np.linalg.matrix_rank(A) < len(A)


def periodic_corr(x, y):
    """Periodic correlation, implemented using the FFT.

    x and y must be real sequences with the same length.
    """
    return ifft(fft(x) * fft(y).conj()).real


def chebfft(v, x):
    N = len(v) - 1
    if N == 0:
        return 0
    ii = np.arange(0, N)
    iir = np.arange(1 - N, 0)
    iii = np.array(ii, dtype=int)
    V = np.hstack((v, v[N - 1 : 0 : -1]))
    U = np.real(fft(V))
    W = np.real(ifft(1j * np.hstack((ii, [0.0], iir)) * U))
    w = np.zeros(N + 1)
    w[1:N] = -W[1:N] / np.sqrt(1 - x[1:N] ** 2)
    w[0] = sum(iii ** 2 * U[iii]) / N + 0.5 * N * U[N]
    w[N] = (
        sum((-1) ** (iii + 1) * ii ** 2 * U[iii]) / N + 0.5 * (-1) ** (N + 1) * N * U[N]
    )
    return w


def log(msg, logfile, print_to_out=True):
    """
    Print log message to  stdout and the given logfile.
    """
    logfile.write(msg + "\n")

    if print_to_out:
        print(msg)


def roll_batch_(data, vec=None):
    if vec is not None:
        vec = np.array(vec)
        shifts = np.repeat(vec[None], len(data), axis=0)
    else:
        shifts = np.random.randint(data.shape[-1], size=[len(data), 2])
    for i in range(len(data)):
        data[i] = torch.roll(data[i], tuple(shifts[i]), (0, 1))


def roll_batch(data, vec=None):
    data_rolled = data.clone()
    roll_batch_(data_rolled, vec)
    return data_rolled


def roll_dataset(X, y, dim = 1):

    delta_transl = 1
    max_transl = X.shape[-1]
    transl_x = range(0, max_transl, delta_transl)
    if dim == 2:
        transl_y = range(0, max_transl, delta_transl)
    else:
        transl_y = [0]

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=len(X),
                                         shuffle=False,
                                         num_workers=0,
                                         pin_memory=True)

    X_rolled, y_rolled = [], []
    for batch_idx, (data, target) in enumerate(loader):
        for ix, tx in enumerate(transl_x):
            for iy, ty in enumerate(transl_y):
                if dim == 2:
                    batch_rolled = torch.roll(data.squeeze(), (tx,ty), dims=(-2,-1))
                else:
                    batch_rolled = torch.roll(data.squeeze(), tx, dims=(-1))
                X_rolled.append(batch_rolled)
                y_rolled.append(target)

    X_rolled = torch.cat(X_rolled)
    y_rolled = torch.cat(y_rolled)

    perm = np.random.permutation(len(X_rolled))
    X_rolled = X_rolled[perm]
    y_rolled = y_rolled[perm]

    return X_rolled, y_rolled


def IPR(w):
    return ((w**2).sum(-1))**2 / (w**4).sum(-1)


def get_width_spline(l, D):
    spline = UnivariateSpline(np.arange(D), l - np.max(l)/2, s=0)
    r1, r2 = spline.roots()
    return np.abs(r2 - r1)


def compute_distance(x, y, D):
    return np.minimum(np.abs(x - y), D - np.abs(x - y))


def circular_mean_std(l, D):
    ps = np.exp(1j * 2 * np.pi * np.arange(D)/D)
    m1 = (l * ps).sum()
    anglem = np.angle(m1)
#     R = np.abs(m1)
#     return anglem / (2 * np.pi) * D, np.sqrt(np.log(1/R**2))
    return anglem / (2 * np.pi) * D, None
