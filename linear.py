import copy
import sys
from typing import Tuple

import cupy as cpy


from optimizer import Optimizer


class Linear:
    """Linear layer."""

    def __init__(self, in_features_size: int, out_features_size: int, bias: bool = True) -> None:
        """Initialize.

        Args:
            in_features_size: size of each input sample.
            out_features_size: size of each output sample.
            bias: if set to False layer will not learn additive bias. Defaults to True.
        """
        self.in_features_size = in_features_size
        # in general w is defined as [out_features_size, in_features_size] however used the opp.
        self.w = cpy.zeros([in_features_size, out_features_size])
        if bias:
            self.b = cpy.zeros([out_features_size])
        else:
            self.b = None
        self.cache = dict(input=None)
        self.set_parameters()
        self.optimizer_w = None
        self.optimizer_b = None

    def set_parameters(self) -> None:
        """Set parameters."""
        stdv = 1.0 / cpy.sqrt(self.in_features_size)
        self.w = cpy.random.uniform(-stdv, stdv, self.w.shape)
        if self.b is not None:
            self.b = cpy.random.uniform(-stdv, stdv, self.b.shape)

    def forward(self, x: cpy.ndarray) -> cpy.ndarray:
        """Forward propagation.

        Args:
            x: input array.

        Returns:
            computed linear layer output.
        """
        y = cpy.dot(x, self.w)
        if self.b is not None:
            y += self.b
        self.cache = dict(input=x)
        return y

    def backward(self, grad: cpy.ndarray) -> cpy.ndarray:
        """Backward propagation.

        Args:
            grad: represents the gradient w.r.t. the output. Defaults to None.

        Returns:
            the gradients w.r.t. the input.
        """
        input = self.cache["input"]
        if len(grad.shape) == 3:
            output_grad = cpy.dot(grad, self.w.T)
            self.grad_w = cpy.sum(cpy.matmul(input.transpose(0, 2, 1), grad), axis=0)
            if self.b is not None:
                self.grad_b = cpy.sum(grad, axis=(0, 1))
            return output_grad
        else:
            output_grad = cpy.dot(grad, self.w.T)
            self.grad_w = cpy.dot(input.T, grad)
            if self.b is not None:
                self.grad_b = grad.sum(axis=0)
            return output_grad

    def set_optimizer(self, optimizer: Optimizer) -> None:
        """Set optimizer.

        Args:
            optimizer: optimizer.
        """
        self.optimizer_w = copy.deepcopy(optimizer)
        self.optimizer_b = copy.deepcopy(optimizer)

    def update_weights(self) -> None:
        """Update weights based on the calculated gradients."""
        self.w = self.optimizer_w.update(self.grad_w, self.w)
        if self.b is not None:
            self.b = self.optimizer_b.update(self.grad_b, self.b)

    def __call__(self, x: cpy.ndarray) -> cpy.ndarray:
        """Defining __call__ method to enable function like call.

        Args:
            x: input array.

        Returns:
            computed linear output.
        """
        return self.forward(x)

    def set_parameters_externally(self, w: cpy.ndarray, b: cpy.ndarray) -> None:
        """Set parameters externally. used for testing.

        Args:
            w: weight.
            b: bias.
        """
        self.w = w
        self.b = b

    def get_grads(self) -> Tuple[cpy.ndarray, cpy.ndarray]:
        """Access gradients.used for testing.

        Returns:
            returns gradients
        """
        return self.grad_w, self.grad_b
    
    def get_weights(self):
        return {"w": cpy.asnumpy(self.w), "b": cpy.asnumpy(self.b)}

    def set_weights(self, weights):
        self.w = cpy.array(weights["w"])
        self.b = cpy.array(weights["b"])
