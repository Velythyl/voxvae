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
            eqx.nn.BatchNorm(32, 0),  # Add batch norm
            jax.nn.swish,
            eqx.nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, key=keys[1]),
            eqx.nn.BatchNorm(64, 0),  # Add batch norm
            jax.nn.swish,
            eqx.nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1, key=keys[2]),
            eqx.nn.BatchNorm(128, 0),  # Add batch norm
            jax.nn.swish,
        ]

        self.embedlayers = [
            eqx.nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1, key=keys[3]),
            eqx.nn.BatchNorm(256, 0),  # Add batch norm
            jax.nn.swish,
            eqx.nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1, key=keys[4]),
            eqx.nn.BatchNorm(512, 0),  # Add batch norm
            jax.nn.swish,
            eqx.nn.Conv3d(512, L, kernel_size=1, key=keys[5]),
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
            final_output_channels = 4  # For classification
        else:
            final_output_channels = 1  # For regression

        _, keys = split_key(key, 6)

        self.layers = [
            eqx.nn.ConvTranspose3d(L, 512, kernel_size=1, key=keys[0]),
            eqx.nn.BatchNorm(512, 0),  # Add batch norm
            jax.nn.swish,
            eqx.nn.ConvTranspose3d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, key=keys[1]),
            eqx.nn.BatchNorm(256, 0),  # Add batch norm
            jax.nn.swish,
            eqx.nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, key=keys[2]),
            eqx.nn.BatchNorm(128, 0),  # Add batch norm
            jax.nn.swish,
            eqx.nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, key=keys[3]),
            eqx.nn.BatchNorm(64, 0),  # Add batch norm
            jax.nn.swish,
            eqx.nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, key=keys[4]),
            eqx.nn.BatchNorm(32, 0),  # Add batch norm
            jax.nn.swish,
            eqx.nn.ConvTranspose3d(32, final_output_channels, kernel_size=3, stride=2, padding=1, output_padding=1, key=keys[5]),
        ]

    def __call__(self, x):
        x = x[:, None, None, None]

        for layer in self.layers:
            x = layer(x)

        if self.use_softmax:
            x = jax.nn.log_softmax(x, axis=0)  # For classification
        else:
            x = jax.nn.sigmoid(x)  # For regression, constrain output to [0, 1]

        return x