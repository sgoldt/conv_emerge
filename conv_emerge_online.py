#!/usr/bin/env python3
"""
Online SGD on a binary classification with mixtures of various distributions.

Date: April 2021

Author: Sebastian Goldt <goldt.sebastian@gmail.com>
"""

import argparse
import datetime
import math
import pickle
from timeit import default_timer as timer
import sys

import torch
import torch.nn.functional as F
from torch import optim

import network
import tasks
import utils

NUM_TESTSAMPLES = 20000


def binarise(x):
    """
    Binarises the given input vector
    """
    x[x > 0] = 1
    x[x < 0] = -1
    return x


def preprocess(xs, classes, num_classes, args):
    """
    Does some basic preprocessing of inputs and labels

    Parameters:
    -----------
    xs : (P, D)
        P input vectors in D dimensions
    classes:
        P class labels as categorical variables with values 0, 1, ..., C-1
    num_classes : number of classes for this task
    args: argparse arguments given to the programme.
    """
    # add a channel to inputs if network is convolutional
    if args.model == "convolutional":
        C_in = 1  # one input channel for our "grayscale" images
        if args.dim == 1:
            xs = xs.reshape(xs.shape[0], C_in, xs.shape[1])
        else:  # dim == 2
            # args.D gives the linear dimension along the width or height
            xs = xs.reshape(xs.shape[0], C_in, args.D, args.D)
    # Pre-processing of targets of test data
    if args.binarise:
        xs = binarise(xs)
    elif args.loss == "ce":
        # train with class labels
        targets = classes
    elif args.loss == "mse":
        # train with one-hot encodings
        targets = utils.get1hot(classes, num_classes)

        if args.g in ["erf", "tanh"]:
            # encode the labels as -1, 1 if we train with erf
            targets = 2 * targets - 1
    else:
        raise ValueError("Did not understand the loss function.")

    return xs, targets


def main():
    # read command line arguments
    parser = argparse.ArgumentParser()
    inputs_help = "inputs: nlgp | gp"
    parser.add_argument("--inputs", nargs="+", default=["gp"], help=inputs_help)
    parser.add_argument(
        "--xis", type=float, default=[0.1, 0.5], nargs="+", help="correlation lengths"
    )
    gain_help = "gain of the NLGP"
    parser.add_argument("--gain", default=1, type=float, help=gain_help)
    model_help = "model: 2layer | conv | lazy | mf"
    parser.add_argument("--model", default="mf", help=model_help)
    g_help = "student activation function: tanh | linear | relu | erf"
    parser.add_argument("-g", default="erf", help=g_help)
    dim_help = "input dimension: one for vector, two for images. The input will have D**dim pixels"
    parser.add_argument("--dim", type=int, default=1, help=dim_help)
    D_help = "linear input dimension. The input will have D**dim pixels."
    parser.add_argument("-D", "--D", type=int, default=400, help=D_help)
    K_help = "# of student nodes / channels of the student"
    parser.add_argument("-K", "--K", type=int, default=8, help=K_help)
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=5,
        help="kernel size (for convolutional student)",
    )
    parser.add_argument("--bs", type=int, default=1, help="mini-batch size")
    bias_help = "train the bias. If an (optional) number b is given, the biases are not trained, but instead fixed at half the biases will be fixed at ±b."
    parser.add_argument("--bias", type=float, nargs="?", default="0", help=bias_help)
    parser.add_argument(
        "--pbc", help="periodic boundary condition for inputs", action="store_true"
    )
    parser.add_argument(
        "--extensivexis", help="multiply the given xi by D", action="store_true"
    )
    parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
    parser.add_argument(
        "--tied", help="For auto-encoder: tie the weights", action="store_true"
    )
    parser.add_argument("--l2", type=float, default=0, help="l2 reg")
    parser.add_argument("--l1", type=float, default=0, help="l1 reg")
    parser.add_argument("--loss", default="mse", help="loss function: ce | mse")
    parser.add_argument(
        "--device", "-d", default="cpu", help="run the program on: 'cuda' or 'cpu'."
    )
    parser.add_argument("--time", type=int, default=1000, help="max time = steps / D")
    parser.add_argument("--binarise", help="binarise the inputs", action="store_true")
    parser.add_argument("-q", "--quiet", help="be quiet", action="store_true")
    parser.add_argument("-s", "--seed", type=int, default=0, help="rng seed. Default=0")
    parser.add_argument("--dummy", help="dummy argument", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print("Using device:", device)

    # The learning task will be a mixture. Put together the distributions
    num_distributions = max(len(args.xis), len(args.inputs))

    if len(args.xis) != len(args.inputs):
        if len(args.xis) == 1:
            args.xis = [args.xis[0]] * num_distributions
        elif len(args.inputs) == 1:
            args.inputs = [args.inputs[0]] * num_distributions
        else:
            msg = """
                You need to give me either the same number of xis, inputs;
                or one input, different xis;
                or different inputs, one xi
                """
            raise ValueError(msg)

    # create the variance for the given correlation length
    xis = torch.tensor(args.xis)
    if args.extensivexis:
        xis *= args.D

    distributions = tasks.build_nlgp_mixture(
        args.inputs, xis, args.D, args.pbc, args.gain, dim=args.dim, xi_pow_pi=True,
    )
    task = tasks.Mixture(distributions)

    # create a nicely formatted description of this task for filenames
    task_desc = [str(val) for pair in zip(args.inputs, xis.numpy()) for val in pair]
    task_desc = "_".join(task_desc)
    if "nlgp" in args.inputs:
        task_desc += "_gain%g" % args.gain
    if args.pbc:
        task_desc += "_pbc"
    task_desc += "_dim%d_D%d" % (args.dim, args.D)

    # Work out what to do with the bias
    train_bias = args.bias is None
    bias_value = args.bias  # either None if just --bias is given or a numerical value

    # student network and loss
    g = utils.get_act(args.g)
    model_desc = args.model
    input_dim = args.D ** args.dim

    student = None
    student_desc = None
    if args.model == "convolutional":
        student = network.ConvTwoLayer(
            g,
            args.D,  # width = height
            args.K,  # num_channels
            2,  # num_classes
            args.dim,
            kernel_size=args.kernel_size,
            train_bias=train_bias,
        )
        student_desc = args.model + ("_channels%d_ks%d" % (args.K, args.kernel_size))
        if train_bias:
            student_desc += "_bias"
    else:
        student = network.TwoLayer(
            g,
            input_dim,
            args.K,
            task.num_classes,
            std0=1e-1,
            train_bias=train_bias,
            model=args.model,
        )
        # Change the second layer weights in case we only train the first-layer weights
        if args.model in ["conv", "lazy", "mf"]:
            student.fc2[:, 1] *= -1
        bias_desc = None
        # fix the bias
        if train_bias:
            # Initialise with large random biases to break the initial symmetry of the
            # network and allow it to break away from the equator.  Here, we choose ± 1,
            # because it will allow us to neatly separate neurons according to which class
            # they code for.
            with torch.no_grad():
                torch.nn.init.constant_(student.bias, 1)
                student.bias[int(args.K / 2) :] *= -1
            bias_desc = "bias"
        if not train_bias and bias_value > 0:
            torch.nn.init.constant_(student.bias, args.bias)
            student.bias[int(args.K / 2) :] *= -1
            bias_desc = "bias%g" % args.bias

        # put it all together for the student description
        student_desc = "%s_%s_K%d" % (args.model, args.g, student.K)
        if bias_desc is not None:
            student_desc += "_" + bias_desc

    student.to(device)

    # Tune the learning rate depending on the scaling of output layers
    lr = args.lr
    params = None
    if args.model == "convolutional":
        params = student.parameters()
    else:
        if args.model == "mf":
            lr *= args.K
        elif args.model == "lazy":
            lr *= math.sqrt(args.K)

        # Collect the trainable student parameters for the optimiser
        params = []
        params += [{"params": student.fc1, "lr": lr}]
        if train_bias:
            # correct learning rate for the bias
            params += [{"params": student.bias, "lr": lr / input_dim}]
        if args.model == "2layer":
            # Ensure the learning of the second layer rate scales correctly
            params += [{"params": student.fc2, "lr": lr / input_dim}]

    # Optimiser
    optimiser = optim.SGD(params, lr=lr)

    # Set up the loss function
    loss_fn = {"ce": F.cross_entropy, "mse": F.mse_loss}[args.loss]

    # Sample a test test
    test_xs, test_classes = task.sample(NUM_TESTSAMPLES)
    test_xs, test_targets = preprocess(test_xs, test_classes, task.num_classes, args)

    if torch.any(torch.isnan(test_xs)):
        print("Could not sample from the mixture, got some nans. Will exit now.")
        sys.exit()

    fname_root = "online_%s_%s_%s_lr%g_bs%d_l2%g_l1%g_seed%d" % (
        task_desc,
        args.loss,
        student_desc,
        lr,
        args.bs,
        args.l2,
        args.l1,
        args.seed,
    )
    log_fname = fname_root + ".log"
    logfile = open(log_fname, "w", buffering=1)
    welcome = "# Online learning on classification\n"
    args_desc = ["%s: %s" % (str(k), str(v)) for k, v in args.__dict__.items()]
    welcome += "# " + "; ".join(args_desc)
    # header for the column titles
    welcome += "\n# time, test loss, test accuracy, avg grads: "
    for name, _ in student.named_parameters():
        welcome += "%s, " % name
    utils.log(welcome, logfile)

    # when to print?
    end = torch.log10(torch.tensor([1.0 * args.time])).item()
    times_to_print = list(torch.logspace(-1, end, steps=200))

    # storing the weights: time -> weight matrix
    weights = {"time": []}
    for name, _ in student.named_parameters():
        weights[name] = []

    start = timer()
    time = 0
    dtime = 1 / input_dim

    # keep track of the gradients between two log messages to compute their average
    # This is the method of tracking gradients with the least computational overhead
    last_gradients = []

    while len(times_to_print) > 0:
        if time >= times_to_print[0].item() or time == 0:
            student.eval()
            with torch.no_grad():
                # compute the generalisation error w.r.t. the noiseless teacher
                preds = student(test_xs)
                eg = loss_fn(preds, test_targets)

                msg = "%g, %g, " % (time, eg)

                # get class for each input
                prediction = preds.max(1)[1]
                # number of correct predictions
                num_c = prediction.eq(test_classes.view_as(prediction)).sum().item()
                accuracy = 100.0 * num_c / NUM_TESTSAMPLES
                msg += "%g, " % (accuracy)

                # average gradient norm
                if time > 0:
                    grads_mean = torch.tensor(last_gradients).mean(dim=0)
                    msg += ("{:g}, " * len(grads_mean)).format(*grads_mean)
                    last_gradients = []
                else:
                    # add zeros to the first line of the output so all lines
                    # have same number of columns
                    for param in student.parameters():
                        if param.requires_grad:
                            msg += "nan, "

                quiet_msg = msg[:-2]

                # collect time, weights
                weights["time"] += [time]
                for name, param in student.named_parameters():
                    weights[name] += [param.clone().detach()]

                # Store weights
                weights_fname = fname_root + "_weights.pickle"
                with open(weights_fname, "wb") as f:
                    pickle.dump(weights, f, pickle.HIGHEST_PROTOCOL)

                msg = msg[:-2]
                utils.log(msg, logfile, print_to_out=not args.quiet)
                if args.quiet:
                    # instead, if we're in quiet mode, only print abbreviated msg
                    print(quiet_msg)

                times_to_print.pop(0)

        # TRAINING
        student.train()
        # sample mini-batch
        xs, targets = task.sample(args.bs)
        xs, targets = preprocess(xs, targets, task.num_classes, args)

        # forward pass
        preds = student(xs)

        # loss
        loss = loss_fn(preds, targets)

        # backward pass
        student.zero_grad()
        loss.backward()
        optimiser.step()

        # manually perform l2 / l1 regularisation, if desired
        # do it manually to control the prefactor of the weight decay term
        with torch.no_grad():
            for param in student.parameters():
                # regularise only the trainable parameters
                if not param.requires_grad:
                    continue

                if args.l2 > 0:  # l2 weight decay
                    param.data -= args.l2 / input_dim * param.data
                elif args.l1 > 0:  # l1 penalty
                    param.data -= args.l1 / input_dim * torch.sign(param.data)

        # log the gradients
        last_gradients += [
            [
                torch.norm(param.grad).item()
                for param in student.parameters()
                if param.requires_grad
            ]
        ]

        time += dtime

    end = timer()
    time_elapsed = datetime.timedelta(seconds=int(end - start))
    utils.log("# Computation took %s (hh:mm:ss)" % time_elapsed, logfile)

    print("Bye-bye")


if __name__ == "__main__":
    main()
