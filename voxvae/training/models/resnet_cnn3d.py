import equinox as eqx
import jax
from jax import numpy as jnp

from voxvae.utils.jaxutils import split_key


class ResidualBlock3D(eqx.Module):
    conv1: eqx.nn.Conv3d
    conv2: eqx.nn.Conv3d
    activation: callable

    def __init__(self, in_channels, out_channels, key, activation=jax.nn.swish):
        key1, key2 = jax.random.split(key)
        self.conv1 = eqx.nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, key=key1)
        self.conv2 = eqx.nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, key=key2)
        self.activation = activation

    def __call__(self, x):
        residual = x
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x += residual  # Add the residual connection
        return self.activation(x)


class ResConv3D_Encoder(eqx.Module):
    conv_layers: list
    embed_layers: list

    def __init__(self, key, N, L, deeper_embed=False):
        _, keys = split_key(key, 7)

        # Initial convolution layer
        self.conv_layers = [
            eqx.nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1, key=keys[0]),
            jax.nn.swish,
            ResidualBlock3D(32, 32, key=keys[1]),
            eqx.nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, key=keys[2]),
            jax.nn.swish,
            ResidualBlock3D(64, 64, key=keys[3]),
            eqx.nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1, key=keys[4]),
            jax.nn.swish,
            ResidualBlock3D(128, 128, key=keys[5]),
        ]

        if not deeper_embed:
            # Flatten and embed
            self.embed_layers = [
                lambda x: jnp.reshape(x, (x.shape[0], -1)).reshape(128 * (N // 8) ** 3),
                eqx.nn.Linear(128 * (N // 8) ** 3, L, key=keys[6]),
            ]
        else:
            _, last_keys = split_key(keys[6], 3)
            # Flatten and embed
            self.embed_layers = [
                lambda x: jnp.reshape(x, (x.shape[0], -1)).reshape(128 * (N // 8) ** 3),
                eqx.nn.Linear(128 * (N // 8) ** 3, 512, key=last_keys[0]),
                jax.nn.swish,
                eqx.nn.Linear(512, L, key=last_keys[2]),
            ]

    def __call__(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        for layer in self.embed_layers:
            x = layer(x)
        return x


class ResConv3D_Decoder(eqx.Module):
    layers: list
    use_softmax: bool

    def __init__(self, key, N, L, use_onehot=False, deeper_embed=False):
        self.use_softmax = use_onehot
        _, keys = split_key(key, 7)

        if use_onehot:
            final_output_channels = 4  # For classification
        else:
            final_output_channels = 1  # For regression

        if not deeper_embed:
            self.layers = [
                eqx.nn.Linear(L, 128 * (N // 8) ** 3, key=keys[0]),
                lambda x: jnp.reshape(x, (128, N // 8, N // 8, N // 8)),
                jax.nn.swish,
            ]
        else:
            _, last_keys = split_key(keys[0], 3)
            # Flatten and embed
            self.layers = [
                eqx.nn.Linear(L, 512, key=last_keys[2]),
                jax.nn.swish,
                eqx.nn.Linear(512, 128 * (N // 8) ** 3, key=last_keys[0]),
                lambda x: jnp.reshape(x, (x.shape[0], -1)).reshape(128 * (N // 8) ** 3),
                jax.nn.swish,
            ]

        self.layers = self.layers + [
            ResidualBlock3D(128, 128, key=keys[1]),
            eqx.nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, key=keys[2]),
            jax.nn.swish,
            ResidualBlock3D(64, 64, key=keys[3]),
            eqx.nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, key=keys[4]),
            jax.nn.swish,
            ResidualBlock3D(32, 32, key=keys[5]),
            eqx.nn.ConvTranspose3d(32, final_output_channels, kernel_size=3, stride=2, padding=1, output_padding=1,
                                   key=keys[6]),
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.use_softmax:
            x = jax.nn.log_softmax(x, axis=0)
        return x

    def call_shunt(self, x):
        x = self(x)
        if self.use_softmax:
            x = jnp.argmax(x, axis=0)
            return x
        else:
            admissible_values = jnp.array([0.0, 0.33, 0.66, 0.99])
            x = jnp.argmin(jnp.abs(x[..., None] - admissible_values), axis=-1)
            x = admissible_values[x]
            return x
