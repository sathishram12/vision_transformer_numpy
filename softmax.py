import cupy as cpy


class Softmax:
    """Computes softmax."""

    def __init__(self) -> None:
        """Initialize."""
        self.cache = dict(output=None)

    def forward(self, x: cpy.ndarray) -> cpy.ndarray:
        """Forward propagation.

        Args:
            x: input array.

        Returns:
            computed softmax output.
        """
        max_val = cpy.max(x, axis=-1, keepdims=True)  # Keep dimensions consistent
        exp_vals = cpy.exp(x - max_val)  # Subtract max for numerical stability
        y = exp_vals / cpy.sum(exp_vals, axis=-1, keepdims=True)
        self.cache = dict(output=y)
        return y

    def backward(self, grad: cpy.ndarray) -> cpy.ndarray:
        """Backward propagation.

        Args:
            grad: represents the gradient w.r.t. the output. Defaults to None.

        Returns:
            the gradients w.r.t. the input.
        """
        softmax = self.cache["output"]
        # ref - https://github.com/tensorflow/tensorflow/blob/0.5.0/tensorflow/python/ops/nn_grad.py
        # fails
        # return softmax * (grad -(grad * softmax).sum(axis=1)[:,None])
        # ref - https://github.com/AkiRusProd/numpy-transformer/blob/master/transformer/activations.py
        J = softmax[..., cpy.newaxis] * cpy.tile(
            cpy.identity(softmax.shape[-1]), (softmax.shape[0], *tuple(cpy.ones(softmax.ndim, dtype=cpy.int8).tolist()))
        ) - (
            softmax[..., cpy.newaxis, :].transpose(
                *tuple(cpy.arange(0, softmax.ndim - 1, 1, dtype=cpy.int8).tolist()), -1, -2
            )
            @ softmax[..., cpy.newaxis, :]
        )
        input_grad = grad[..., cpy.newaxis, :] @ J
        return input_grad.reshape(grad.shape)


    def __call__(self, x: cpy.ndarray) -> cpy.ndarray:
        """Defining __call__ method to enable function like call.

        Args:
            x: input array.

        Returns:
            computed softmax output.
        """
        return self.forward(x)
