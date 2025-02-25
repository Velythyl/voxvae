from jax import numpy as jnp

def NormalizeData(data, min=0):
    return (data - min) / (jnp.max(data) - min)

def prepare_batch(x):
    if len(x.shape) == 3:
        x = x[None, :, :, :]
    elif len(x.shape) == 4:
        x = jnp.expand_dims(x, axis=1)  # Add channel dimension
    # makes arrays 0-1
    #x = NormalizeData(x, min=0)
    return x
