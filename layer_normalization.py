import sys


import copy
from typing import Tuple

import cupy as cpy
from optimizer import Optimizer


class LayerNormalization:
    """Applies layer normalization"""

    def __init__(self, normalized_shape, epsilon=0.00001) -> None:

        self.normalized_shape = normalized_shape

        self.epsilon = epsilon

        self.gamma = cpy.ones(self.normalized_shape)
        self.beta = cpy.zeros(self.normalized_shape)

        self.mean = None
        self.var = None

        self.optimizer_w = None
        self.optimizer_b = None

    def set_optimizer(self, optimizer: Optimizer) -> None:
        """Set optimizer.

        Args:
            optimizer: optimizer.
        """
        self.optimizer_w = copy.deepcopy(optimizer)
        self.optimizer_b = copy.deepcopy(optimizer)

    def forward(self, x: cpy.ndarray) -> cpy.ndarray:
        """Forward propagation.

        Args:
            x: input array.

        Returns:
            computed linear layer output.
        """
        self.input_data = x
        self.mean = cpy.mean(self.input_data, axis=-1, keepdims=True)
        self.var = cpy.var(self.input_data, axis=-1, keepdims=True)
        self.x_centered = self.input_data - self.mean
        self.stddev_inv = 1 / cpy.sqrt(self.var + self.epsilon)
        self.x_hat = self.x_centered * self.stddev_inv
        self.output_data = self.gamma * self.x_hat + self.beta
        return self.output_data

    def __call__(self, x: cpy.ndarray) -> cpy.ndarray:
        """Defining __call__ method to enable function like call.

        Args:
            x: input array.

        Returns:
            computed layer normalization output.
        """
        return self.forward(x)

    def backward(self, error: cpy.ndarray) -> cpy.ndarray:
        """Backward propagation.

        Args:
            error: represents the gradient w.r.t. the output. Defaults to None.

        Returns:
            the gradients w.r.t. the input.
        """
        self.normalized_axis = tuple(range(self.input_data.ndim - 1))
        dx_hat = error * self.gamma[cpy.newaxis, cpy.newaxis, :]
        dvar = cpy.sum(dx_hat * self.x_centered, axis=-1, keepdims=True) * -0.5 * self.stddev_inv**3
        dmu = cpy.sum(dx_hat * -self.stddev_inv, axis=-1, keepdims=True) + dvar * cpy.mean(
            -2.0 * self.x_centered, axis=-1, keepdims=True
        )

        output_error = (
            (dx_hat * self.stddev_inv)
            + (dvar * 2 * self.x_centered / self.normalized_shape)
            + (dmu / self.normalized_shape)
        )

        self.grad_gamma = cpy.sum(error * self.x_hat, axis=self.normalized_axis)
        self.grad_beta = cpy.sum(error, axis=self.normalized_axis)

        return output_error

    def update_weights(self) -> None:
        """Update weights based on the calculated gradients."""
        self.gamma = self.optimizer_w.update(self.grad_gamma, self.gamma)
        self.beta = self.optimizer_b.update(self.grad_beta, self.beta)

    def get_grads(self) -> Tuple[cpy.ndarray, cpy.ndarray]:
        """Access gradients.used for testing.

        Returns:
            returns gradients
        """
        return self.grad_gamma, self.grad_beta
    
    def get_weights(self):
        return {"gamma": cpy.asnumpy(self.gamma), "beta": cpy.asnumpy(self.beta)}

    def set_weights(self, weights):
        self.gamma = cpy.array(weights["gamma"])
        self.beta = cpy.array(weights["beta"])
