import sys

import cupy as cpy
from linear import Linear
from softmax import Softmax


class MultiHeadAttention:
    """Multi head attention"""

    def __init__(self, dimension: int, n_heads: int = 2) -> None:
        """Initialize.

        Args:
            dimension: input dimension.
            n_heads: number of heads.
        """
        self.n_heads = n_heads
        self.d_head = int(dimension / n_heads)
        # Q K V has a d_head * d_head size
        self.q_mappings = [Linear(self.d_head, self.d_head) for _ in range(self.n_heads)]
        self.k_mappings = [Linear(self.d_head, self.d_head) for _ in range(self.n_heads)]
        self.v_mappings = [Linear(self.d_head, self.d_head) for _ in range(self.n_heads)]
        self.softmax = [Softmax() for _ in range(self.n_heads)]

    def forward(self, sequences: cpy.ndarray) -> cpy.ndarray:
        """Forward propagation.

        Args:
            sequences: input array.

        Returns:
            computed multi head attention layer output.
        """
        self.sequences = sequences
        self.scale = cpy.sqrt(self.d_head)
        # convert to list of n_heads elements with info of size (N, seq_length, dimension / n_heads)
        sequences = cpy.split(sequences, self.n_heads, axis=-1)
        result = []
        q_seq = []
        k_seq = []
        v_seq = []
        attention_seq = []
        for head in range(self.n_heads):
            q_mapping = self.q_mappings[head]
            k_mapping = self.k_mappings[head]
            v_mapping = self.v_mappings[head]
            seq = sequences[head]
            q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)
            q_seq.append(q)
            k_seq.append(k)
            v_seq.append(v)
            attention_seq_head = self.softmax[head](q @ k.transpose(0, 2, 1) / (self.d_head**0.5))
            attention_seq.append(attention_seq_head)
            result.append(attention_seq_head @ v)
        # convert to (N, seq_length, dimension)
        self.result = cpy.dstack(result)
        self.q_seqs = cpy.dstack(q_seq)
        self.k_seqs = cpy.dstack(k_seq)
        self.v_seqs = cpy.dstack(v_seq)
        self.attention_seqs = cpy.dstack(attention_seq)
        return self.result

    def backward(self, error: cpy.ndarray) -> None:
        """Backward propagation..

        Args:
            grad: represents the gradient w.r.t. the output. Defaults to None.

        Returns:
            the gradients w.r.t. the input.
        """
        error_head_split = cpy.split(error, self.n_heads, axis=-1)
        attention_seqs_split = cpy.split(self.attention_seqs, self.n_heads, axis=-1)
        q_seqs_split = cpy.split(self.q_seqs, self.n_heads, axis=-1)
        k_seqs_split = cpy.split(self.k_seqs, self.n_heads, axis=-1)
        v_seqs_split = cpy.split(self.v_seqs, self.n_heads, axis=-1)

        final_error = []
        for i in range(self.n_heads):
            err_attn = error_head_split[i] @ v_seqs_split[i].transpose(0, 2, 1)
            pre_attn_error = self.softmax[i].backward(err_attn)
            v_grad_in = attention_seqs_split[i].transpose(0, 2, 1) @ error_head_split[i]
            error_v_out_i = self.v_mappings[i].backward(v_grad_in)

            k_error = (q_seqs_split[i].transpose(0, 2, 1) @ pre_attn_error) / (self.d_head**0.5)
            k_error = k_error.transpose(0, 2, 1)
            error_k_out_i = self.k_mappings[i].backward(k_error)

            q_error = (pre_attn_error @ k_seqs_split[i]) / (self.d_head**0.5)
            error_q_out_i = self.q_mappings[i].backward(q_error)

            seq_error = error_q_out_i + error_k_out_i + error_v_out_i
            final_error.append(seq_error)
        return cpy.dstack(final_error)

    def set_optimizer(self, optimizer: object) -> None:
        """Set optimizer.

        Args:
            optimizer: optimizer.
        """
        for v_mapping in self.v_mappings:
            v_mapping.set_optimizer(optimizer)
        for q_mapping in self.q_mappings:
            q_mapping.set_optimizer(optimizer)
        for k_mapping in self.k_mappings:
            k_mapping.set_optimizer(optimizer)

    def update_weights(self) -> None:
        """Update weights based on the calculated gradients."""
        for v_mapping in self.v_mappings:
            v_mapping.update_weights()
        for q_mapping in self.q_mappings:
            q_mapping.update_weights()
        for k_mapping in self.k_mappings:
            k_mapping.update_weights()
    
    def get_weights(self):
        """Get weights of the MultiHeadAttention module."""
        weights = {
            "q_mappings": [q_mapping.get_weights() for q_mapping in self.q_mappings],
            "k_mappings": [k_mapping.get_weights() for k_mapping in self.k_mappings],
            "v_mappings": [v_mapping.get_weights() for v_mapping in self.v_mappings],
        }
        return weights

    def set_weights(self, weights):
        """Set weights for the MultiHeadAttention module."""
        for i, q_mapping_weights in enumerate(weights["q_mappings"]):
            self.q_mappings[i].set_weights(q_mapping_weights)
        for i, k_mapping_weights in enumerate(weights["k_mappings"]):
            self.k_mappings[i].set_weights(k_mapping_weights)
        for i, v_mapping_weights in enumerate(weights["v_mappings"]):
            self.v_mappings[i].set_weights(v_mapping_weights)
