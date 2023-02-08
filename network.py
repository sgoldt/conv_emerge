import math
import torch
import torch.nn as nn
import torch.nn.functional as F

pi = math.pi
sqpi = math.sqrt(pi)
sq2pi = math.sqrt(2.0 * pi)
sq2 = math.sqrt(2.0)
sq8 = 2.0 * sq2


############ GENERAL FUNCTIONS ############

activations = {
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "erf": lambda x: torch.erf(x / sq2),
    "erfxmx3": lambda x: 2 / sqpi * (x / sq2 - x ** 3 / (3 * sq8)),
    "erf+": lambda x: 0.5 * (1 + torch.erf(x / sq2)),
    "x3": lambda x: x ** 3 / 3.0,
    "xmx3": lambda x: x - x ** 3 / 3.0,
    "relu": F.relu,
    "linear": lambda x: x,
}


def get_act(act_type):
    return activations[act_type]


class TwoLayer(nn.Module):
    """Simple fully-connected two-layer neural network of the form

    ϕ(x) = ∑ₖᴷ vᵏ g(wᵏ x / √D + bᵏ)

    where D is the dimension of the inputs. Note that even though we train on a
    classification task, we don't put a softmax at the end of the network. In
    this way, we have the choice between using the mean-squared error or some
    other error, such as cross-entropy.

    This network can be operated in four different *modes*:

        2layer : train both layers, starting from Gaussian initial weights
        conv : committee machine, vᵏ = 1
        lazy : committee machine, vᵏ = 1 / √K
        mf : committee machine, vᵏ = 1 / K

    """

    _models = ["2layer", "conv", "lazy", "mf"]

    def __init__(
        self, g, D, K, num_classes, std0=1e-1, train_bias=False, model="2layer"
    ):
        """
        Parameters:
        -----------

        g : a callable activation function
        D : input dimension
        K : number of hidden nodes
        num_classes : number of output heads
        std0 : standard deviation of initial weights
        train_bias : True if first layer has a trainable bias, default False.
        model : Choice of mode for this network: 2layer (default) | conv | lazy | mf
        """
        super().__init__()

        if model not in self._models:
            msg = "Did not recognise the mode of this network. Must be one of "
            msg += " | ".join(self._models)
            raise ValueError(msg)

        (self.g, self.D, self.K, self.num_classes) = (g, D, K, num_classes)

        # First-layer weights
        self.fc1 = nn.Parameter(torch.zeros(D, K), requires_grad=True)
        # 2nd layer. Initialise with ones; if trainable, will be re-initialised below
        self.fc2 = nn.Parameter(torch.ones(K, num_classes))
        self.fc2.requires_grad = model == "2layer"

        self.norm = {"2layer": 1, "conv": 1, "lazy": math.sqrt(K), "mf": K}[model]

        self.bias = nn.Parameter(torch.zeros(K), requires_grad=train_bias)

        # initialise trainable parameters with Gaussians
        for p in self.parameters():
            if p.requires_grad:
                nn.init.normal_(p, std=std0)

    def forward(self, x):
        """
        Compute the forward pass of the given inputs through the network.
        """
        x = x @ self.fc1 / math.sqrt(self.D) + self.bias
        x = self.g(x)
        x = x @ self.fc2
        x /= self.norm
        return x

    def freeze(self):
        """
        Turn off autograd for all network parameters.
        """
        for param in self.parameters():
            param.requires_grad = False


class ConvTwoLayer(nn.Module):
    """Minimal two-layer convolutional neural network.

    This network can be operated in four different *modes*:

        2layer : train both layers, starting from Gaussian initial weights
        conv : committee machine, vᵏ = 1
        lazy : committee machine, vᵏ = 1 / √K
        mf : committee machine, vᵏ = 1 / K

    """

    _models = ["2layer", "conv", "lazy", "mf"]

    def __init__(
        self,
        g,
        D,
        num_channels,
        num_classes,
        dim=1,
        std0=1e-1,
        train_bias=False,
        model="2layer",
        kernel_size=50,
        pooling = True,
    ):
        """
        Parameters:
        -----------

        g : a callable activation function
        D : width = height of the 'image'
        num_channels : number of channels
        num_classes : number of output heads
        std0 : standard deviation of initial weights
        train_bias : True if first layer has a trainable bias, default False.
        model : Choice of mode for this network: 2layer (default) | conv | lazy | mf
        """
        super().__init__()

        if model not in self._models:
            msg = "Did not recognise the mode of this network. Must be one of "
            msg += " | ".join(self._models)
            raise ValueError(msg)

        (self.g, self.D, self.num_classes) = (g, D, num_classes)

        # First-layer weights
        layer_type = nn.Conv1d if dim == 1 else nn.Conv2d
        self.firstlayer = layer_type(
            1, num_channels, kernel_size=kernel_size, bias=train_bias
        )
        # and the pooling layer
        self.pooling_fn = None
        if pooling:
            self.pooling_fn = F.max_pool1d if dim == 1 else F.max_pool2d

        # formula for width = height of the output channels taken from pyTorch docs at
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        H_out = 1
        H_out += (
            self.D
            + 2 * self.firstlayer.padding[0]
            - self.firstlayer.dilation[0] * (self.firstlayer.kernel_size[0] - 1)
            - 1
        ) / self.firstlayer.stride[0]
        # account for # of output channels with dim 2
        self.num_hidden = int(num_channels * H_out ** 2)
        if pooling:
            self.num_hidden = int(self.num_hidden / (2 ** 2))  # account for max_pool2d
        self.K = self.num_hidden  # for interoperability code calling fc networks, too

        # 2nd layer. Initialise with ones; we divide by K in forward
        self.fc2 = torch.ones(self.num_hidden, num_classes)
        self.fc2[:, 1] *= -1

    def forward(self, x):
        """
        Compute the forward pass of the given inputs through the network.
        """
        x = self.g(self.firstlayer(x))
        if self.pooling_fn is not None:
            x = self.pooling_fn(x, 2)
        x = torch.flatten(x, 1)
        x = x @ self.fc2
        x /= self.num_hidden
        return x

    def freeze(self):
        """
        Turn off autograd for all network parameters.
        """
        for param in self.parameters():
            param.requires_grad = False


class Convolutional(nn.Module):
    def __init__(
        self,
        dim=1,
        N=100,
        num_banks=1,
        kernel_size=5,
        stride=1,
        circular=True,
        num_out=1,
        soft=False,
        init_bias_zero=False,
        g=torch.tanh,
        dtype=torch.float,
        device=torch.device("cpu"),
    ):
        super(Convolutional, self).__init__()
        self.dim = dim
        self.N = N
        self.num_banks = num_banks
        self.kernel_size = kernel_size
        self.stride = stride
        self.circular = circular
        self.num_out = num_out
        self.soft = soft
        self.g = g
        self.pad = (kernel_size - 1) // 2
        lin_inp_size = (N - kernel_size + 2 * self.pad) // stride + 1

        if self.dim == 1:
            Conv = nn.Conv1d
            self.lin_inp_size = lin_inp_size
            pads = self.pad
        else:
            Conv = nn.Conv2d
            self.lin_inp_size = lin_inp_size ** 2
            pads = (self.pad, self.pad)

        padding_mode = "circular" if self.circular else "zeros"
        self.conv = Conv(
            in_channels=1,
            out_channels=num_banks,
            kernel_size=kernel_size,
            stride=stride,
            padding=pads,
            padding_mode=padding_mode,
        ).to(device)
        # NOTE - AI: I implemented this initialization AFTER the experiments which matched
        # soft conv with soft committee in NLGP: get rid of this in case
        if init_bias_zero:
            self.conv.bias.data.zero_()

        self.nl = self.lin_inp_size * num_banks
        self.fc = nn.Linear(self.nl, num_out, bias=False).to(device)
        if soft:
            self.fc.weight.data[:] = 1.0 / math.sqrt(self.nl)
        # NOTE - AI: I implemented this initialization AFTER the experiments which matched
        # soft conv with soft committee in NLGP: get rid of this in case
        else:
            self.fc.weight.data = torch.randn_like(self.fc.weight) / math.sqrt(self.nl)

    def forward(self, x):
        if self.dim == 1:
            x = self.conv(x.view(-1, 1, x.shape[-1]))
        else:
            x = self.conv(x.view(-1, 1, x.shape[-2], x.shape[-1]))
        x = self.g(x)
        x = x.view(-1, self.nl)
        x = self.fc(x)
        return x

    def set_trainable(self, params):
        # reset all grads
        with torch.no_grad():
            for param in self.parameters():
                param.requires_grad = False

        # set trainable weights
        self.conv.weight.requires_grad_()
        to_train = [{"params": self.conv.weight, "lr": params["lr_l1"]}]

        if params["train_bias"]:
            self.conv.bias.requires_grad_()
            to_train += [{"params": self.conv.bias, "lr": params["lr_bias"]}]

        if not params["soft"]:
            self.fc.weight.requires_grad_()
            to_train += [{"params": self.fc.weight, "lr": params["lr_l2"]}]

        return to_train
