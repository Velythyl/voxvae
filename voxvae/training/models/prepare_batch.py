from jax import numpy as jnp


def prepare_batch(x):
    if len(x.shape) == 3:
        return x[None, :, :, :]
    if len(x.shape) == 4:
        x = jnp.expand_dims(x, axis=1)  # Add channel dimension
        return x
