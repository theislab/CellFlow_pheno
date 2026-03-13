from collections.abc import Callable
from typing import Literal

import jax.numpy as jnp
from flax import linen as nn

from cellflow.networks._utils import BaseModule


class MLP(BaseModule):
    """Fully-connected layers with normalization, dropout, and activation.

    Parameters
    ----------
    n_output
        Number of output features.
    n_layers
        Number of hidden layers.
    n_hidden
        Number of hidden units per hidden layer.
    dropout_rate
        Dropout rate.
    normalization
        Type of normalization. One of ``["layer", "batch", "none"]``.
    act_fn
        Activation function.
    """

    n_output: int
    n_layers: int = 1
    n_hidden: int = 128
    dropout_rate: float = 0.1
    normalization: Literal["layer", "batch", "none"] = "layer"
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.leaky_relu

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Forward computation on ``x``.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch_size, n_input)``.
        training
            Whether the model is in training mode.

        Returns
        -------
        Output tensor of shape ``(batch_size, n_output)``.
        """
        z = x
        for _ in range(self.n_layers):
            z = nn.Dense(self.n_hidden)(z)
            if self.normalization == "layer":
                z = nn.LayerNorm()(z)
            elif self.normalization == "batch":
                z = nn.BatchNorm(use_running_average=not training)(z)
            z = self.act_fn(z)
            z = nn.Dropout(self.dropout_rate)(z, deterministic=not training)
        z = nn.Dense(self.n_output)(z)
        return z


class Aggregator(nn.Module):
    """Aggregator for sets of vectors using various pooling strategies.

    Parameters
    ----------
    scoring
        Pooling method. One of ``["attn", "gated_attn", "mean", "max", "sum"]``.
    attn_dim
        Hidden dimension of the attention layers.
    sample_batch_size
        Bag size used for scaling attention weights when ``scale=True``.
    scale
        Whether to scale attention weights by ``N / sample_batch_size``.
    dropout_rate
        Dropout rate (unused in pooling, reserved for subclasses).
    """

    scoring: Literal["attn", "gated_attn", "mean", "max", "sum"] = "gated_attn"
    attn_dim: int = 16
    sample_batch_size: int | None = None
    scale: bool = False
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Forward computation on ``x``.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch_size, N, n_input)``.
        training
            Whether the model is in training mode.

        Returns
        -------
        Pooled tensor of shape ``(batch_size, n_input)``.
        """
        if self.scoring == "sum":
            return jnp.sum(x, axis=-2)
        if self.scoring == "mean":
            return jnp.mean(x, axis=-2)
        if self.scoring == "max":
            return jnp.max(x, axis=-2)

        if self.scoring == "attn":
            # from https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py
            A = nn.Dense(1, use_bias=False)(jnp.tanh(nn.Dense(self.attn_dim)(x)))  # (batch, N, 1)
        elif self.scoring == "gated_attn":
            # from https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py
            A_V = jnp.tanh(nn.Dense(self.attn_dim)(x))        # (batch, N, attn_dim)
            A_U = nn.sigmoid(nn.Dense(self.attn_dim)(x))       # (batch, N, attn_dim)
            A = nn.Dense(1, use_bias=False)(A_V * A_U)         # (batch, N, 1)
        else:
            raise NotImplementedError(
                f"scoring={self.scoring!r} is not implemented. "
                'Must be one of ["attn", "gated_attn", "sum", "mean", "max"].'
            )

        A = jnp.swapaxes(A, -1, -2)          # (batch, 1, N)
        A = nn.softmax(A, axis=-1)            # (batch, 1, N)

        if self.scale:
            if self.sample_batch_size is None:
                raise ValueError("sample_batch_size must be set when scale=True.")
            A = A * A.shape[-1] / self.sample_batch_size

        pooled = jnp.matmul(A, x).squeeze(-2)  # (batch, n_input)
        return pooled
