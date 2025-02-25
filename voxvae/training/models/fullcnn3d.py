import equinox as eqx
import jax
from jax import numpy as jnp

from voxvae.utils.jaxutils import split_key


class Conv3D_Encoder(eqx.Module):
    convlayers: list
    embedlayers: list

    def __init__(self, key, N, L):

        _, keys = split_key(key, 6)

        self.convlayers = [
            eqx.nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1, key=keys[0]),
            jax.nn.swish,
            eqx.nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, key=keys[1]),
            jax.nn.swish,
            eqx.nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1, key=keys[2]),
            jax.nn.swish,
        ]

        self.embedlayers = [
            eqx.nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1, key=keys[3]),  # 256x2x2x2
            jax.nn.swish,
            eqx.nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1, key=keys[4]),  # 512x1x1x1
            jax.nn.swish,
            eqx.nn.Conv3d(512, L, kernel_size=1, key=keys[5]),  # Lx1x1x1
        ]

    def __call__(self, x):
        conved = x
        for layer in self.convlayers:
            conved = layer(conved)

        final = conved
        for layer in self.embedlayers:
            final = layer(final)
        return final.squeeze()


class Conv3D_Decoder(eqx.Module):
    layers: list
    use_softmax: bool

    def __init__(self, key, N, L, use_onehot=False):
        self.use_softmax = use_onehot

        if use_onehot:
            # For classification, output 4 channels (one for each voxel type)
            final_output_channels = 4
        else:
            # For regression, output 1 channel
            final_output_channels = 1

        _, keys = split_key(key, 6)

        self.layers = [
            eqx.nn.ConvTranspose3d(L, 512, kernel_size=1, key=keys[0]),
            jax.nn.swish,
            eqx.nn.ConvTranspose3d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, key=keys[1]),
            jax.nn.swish,
            eqx.nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, key=keys[2]),
            jax.nn.swish,
            eqx.nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, key=keys[3]),
            jax.nn.swish,
            eqx.nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, key=keys[4]),
            jax.nn.swish,
            eqx.nn.ConvTranspose3d(32, final_output_channels, kernel_size=3, stride=2, padding=1, output_padding=1,
                                   key=keys[5]),
        ]

    def __call__(self, x):
        x = x[:, None, None, None]

        for layer in self.layers:
            x = layer(x)

        if self.use_softmax:
            # Apply softmax along the last dimension (voxel types)
            x = jax.nn.log_softmax(x, axis=0)

        return x  # .clip(0,1)

    def call_shunt(self, x):
        x = self(x)

        if self.use_softmax:
            # For classification, take the argmax along the channel dimension (axis 0)
            x = jnp.argmax(x, axis=0)
            # Map the predicted class indices back to the admissible values
            return x
        elif not self.use_softmax:
            # Round to the nearest admissible value
            admissible_values = jnp.array([0.0, 0.33, 0.66, 0.99])
            x = jnp.argmin(jnp.abs(x[..., None] - admissible_values), axis=-1)
            x = admissible_values[x]
            return x


def Downsample3D(grid, block_size):
    """
    A layer that downsamples a 3D voxel grid by dividing it into MxMxM cubes
    and computing the mean of each cube.

    Args:
        block_size (int): The size of the cube (M) to downsample. The input grid
                          dimensions must be divisible by M.
    """

    # Reshape the grid into blocks of size MxMxM
    _, D, H, W = grid.shape
    x = grid.reshape(1, D // block_size, block_size, H // block_size, block_size, W // block_size, block_size)

    # Compute the mean over the MxMxM blocks
    x = jnp.mean(x, axis=(2, 4, 6))

    return x


def Upsample3D(grid, block_size):
    """
    A layer that upsamples a 3D voxel grid by repeating each element in a MxMxM cube.

    Args:
        block_size (int): The size of the cube (M) to upsample. The output grid
                          dimensions will be M times the input dimensions.
    """

    # Get the shape of the input grid
    _, D, H, W = grid.shape

    # Repeat each element in the grid to fill a MxMxM cube
    x = jnp.repeat(grid, block_size, axis=1)
    x = jnp.repeat(x, block_size, axis=2)
    x = jnp.repeat(x, block_size, axis=3)

    return x