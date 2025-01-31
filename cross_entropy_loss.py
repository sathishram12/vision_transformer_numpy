import cupy as cpy
from softmax import Softmax


class CrossEntropyLoss:
    """Computes the cross entropy loss between input logits and target."""

    def __init__(self) -> None:
        """Initialize."""
        self.softmax = Softmax()
        self.cache = dict(input=None, softmax_output=None, one_hot=None)

    def forward(self, y_pred: cpy.ndarray, y_true: cpy.ndarray) -> float:
        """Forward propagation.

        Args:
            y_pred: Logits, the input predictions.
            y_true: Ground truth labels (class indices).

        Returns:
            Computed cross entropy loss.
        """
        # Apply softmax to logits
        softmax_output = self.softmax(y_pred)
        self.cache["input"] = y_pred
        self.cache["softmax_output"] = softmax_output

        # Convert class indices to one-hot representation
        one_hot = cpy.zeros_like(softmax_output)
        one_hot[cpy.arange(len(y_pred)), y_true] = 1
        self.cache["one_hot"] = one_hot

        # Compute cross entropy loss
        log_softmax = cpy.log(softmax_output + 1e-9)  # Add epsilon to prevent log(0)
        loss = -cpy.sum(one_hot * log_softmax, axis=1)  # Loss for each sample
        return cpy.mean(loss)  # Average loss across the batch

    def backward(self) -> cpy.ndarray:
        """Backward propagation.

        Returns:
            Gradients w.r.t. the input logits.
        """
        softmax_output = self.cache["softmax_output"]
        one_hot = self.cache["one_hot"]

        # Gradient of loss w.r.t. logits
        grad = (softmax_output - one_hot) / one_hot.shape[0]
        return grad

    def __call__(self, y_pred: cpy.ndarray, y_true: cpy.ndarray) -> float:
        """Defining __call__ method to enable function-like call.

        Args:
            y_pred: Logits, the input predictions .
            y_true: Ground truth labels .

        Returns:
            Computed cross entropy loss (scalar).
        """
        return self.forward(y_pred, y_true)
