import copy

import cupy as cpy
from optimizer import Optimizer


class Parameter:
    """Parameter wrapper to handle cls."""

    def __init__(self, val) -> None:
        """Initialize."""
        self.val = val
        self.optimizer = None

    def backward(self, grad: cpy.ndarray) -> None:
        """Backward propagation.

        Args:
            grad: represents the gradient w.r.t. the output.
        """
        self.cache = dict(grad=cpy.sum(grad, axis=0)[None, :])

    def set_optimizer(self, optimizer: Optimizer) -> None:
        """Set optimizer.

        Args:
            optimizer: optimizer.
        """
        cloned_optimizer = copy.deepcopy(optimizer)
        self.optimizer = cloned_optimizer

    def update_weights(self) -> None:
        """Update weights based on the calculated gradients."""
        self.val = self.optimizer.update(self.cache["grad"], self.val)
