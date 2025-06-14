import numpy as np
from nets.functional import *


class Parameter:
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)

    @classmethod
    def initialize(cls, shape, init_mode, fan_in, fan_out, gain=np.sqrt(2)):
        """
        Weight initialization:

        We have that:
            out_i = W_i1 x_1 + ... + W_id x_d

        Assume x ~ N(0,1) i.i.d, weights have mean 0, and x and w independent

        Then
            Var(out_i) = Var(W_i1)Var(x_1) + ... + Var(W_id)Var(x_d)
            = Var(W_i1) + ... + Var(W_id)

        This means the variance of each W_ij should be 1/d.


        For normally distributed weights:
            We can achieve this variance by scaling N(0,1) by 1/sqrt(d).

            The backwards pass is a matrix-multiply but with in_features, out_features transposed
            That means we'd have to scale be 1/sqrt(k)

            Instead, we take the average, scaling by sqrt(2/(d+k))


        For Uniformly distributed weights:
            Variance uniform distribution is (b-a)^2/12
            Var(U(-a, a)) -> (2a)^2 / 12 = 4a^2 / 12 = a^2 / 3

            So, to get a variance of 1/d:
            1/d = a^2/3
            3/d = a^2
            a = sqrt(3/d)
            Averaging between forward and backward passses:
            a = sqrt(6/(d+k))
        """
        if init_mode == "xavier_uniform":
            data = np.sqrt(6 / (fan_in + fan_out)) * (2 * np.random.rand(*shape) - 1)
        elif init_mode == "xavier_normal":
            data = np.sqrt(2 / (fan_in + fan_out)) * np.random.randn(*shape)
        elif init_mode == "kaiming_uniform":
            data = gain * np.sqrt(3 / (fan_in)) * (2 * np.random.rand(*shape) - 1)
        elif init_mode == "kaiming_normal":
            data = gain * np.sqrt(1 / (fan_in)) * np.random.randn(*shape)

        return cls(data)


class Module:
    def __init__(self, *args, **kwargs):
        self.params = {}

    def zero_grad(self):
        for p in self.params.values():
            p.grad = np.zeros_like(p.grad)

    def forward(self):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError


# Loss classes will have forward taking in x,y
# And backward taking in cache.
class MSELoss:
    def __init__(self):
        pass

    def forward(self, x, y):
        return mse_loss_forward(x, y)

    def backward(self, cache):
        return mse_loss_backward(cache)


class SoftMaxLoss:
    def __init__(self):
        pass

    def forward(self, x, y):
        return softmax_loss_forward(x, y)

    def backward(self, cache):
        return softmax_loss_backward(cache)


class Linear(Module):
    def __init__(self, in_features, out_features, init_mode="xavier_normal"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.params["W"] = Parameter.initialize(
            (out_features, in_features), init_mode, in_features, out_features
        )
        self.params["b"] = Parameter(np.zeros(out_features))

    def forward(self, x):
        return linear_forward(
            x,
            self.params["W"].data,
            self.params["b"].data,
        )

    def backward(self, dL_dout, cache):
        dL_dx, dL_dw, dL_db = linear_backward(dL_dout, cache)
        self.params["W"].grad = dL_dw
        self.params["b"].grad = dL_db
        return dL_dx


class Conv2d(Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, pad, init_mode="xavier_normal"
    ):
        super().__init__()
        self.pad = pad
        self.params["kernel"] = Parameter.initialize(
            (out_channels, in_channels, kernel_size, kernel_size),
            init_mode,
            fan_in=in_channels * kernel_size * kernel_size,
            fan_out=out_channels * kernel_size * kernel_size,
        )
        self.params["b"] = Parameter(np.zeros(out_channels))

    def forward(self, x):
        return conv2d_forward(x, self.params["kernel"], pad=self.pad)

    def backward(self, dL_dout, cache):
        return conv2d_backward(dL_dout, cache)


"""Activations"""


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return relu_forward(x)

    def backward(self, dL_dout, cache):
        return relu_backward(dL_dout, cache)


class Network(Module):
    def __init__(self, layers, loss_fcn):
        super().__init__()
        self.layers = layers
        self.loss_fcn = loss_fcn

        # Store parameters
        for i, layer in enumerate(self.layers):
            for name, p in layer.params.items():
                self.params[f"{i}_{name}"] = p

    def forward(self, x, y=None):
        caches = []
        for layer in self.layers:
            x, cache = layer.forward(x)
            caches.append(cache)

        if y is not None:
            loss, cache = self.loss_fcn.forward(x, y)
            caches.append(cache)
            return loss, caches
        else:
            return x

    def backward(self, caches):
        dL_dx = self.loss_fcn.backward(caches.pop())
        for layer in reversed(self.layers):
            dL_dx = layer.backward(dL_dx, caches.pop())
