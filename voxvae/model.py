import equinox as eqx
import jax
from jax import numpy as jnp

from voxvae.jaxutils import split_key


class Encoder(eqx.Module):
    layers: list

    def __init__(self, key, N, L):
        _, (key1, key2, key3, key4) = split_key(key, 4)
        self.layers = [
            eqx.nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1, key=key1),
            jax.nn.relu,
            eqx.nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, key=key2),
            jax.nn.relu,
            eqx.nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1, key=key3),
            jax.nn.relu,
            lambda x: jnp.reshape(x, (x.shape[0], -1)).reshape(128 * (N // 8) ** 3),  # Flatten
            eqx.nn.Linear(128 * (N // 8) ** 3, L, key=key4),  # Adjust N and L as needed
            jax.nn.relu
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(eqx.Module):
    layers: list

    def __init__(self, key, N, L):
        _, (key1, key2, key3, key4) = split_key(key, 4)
        self.layers = [
            eqx.nn.Linear(L, 128 * (N // 8) ** 3, key=key1),
            lambda x: jnp.reshape(x, (128, N // 8, N // 8, N // 8)),  # Reshape
            jax.nn.relu,
            eqx.nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, key=key2),
            jax.nn.relu,
            eqx.nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, key=key3),
            jax.nn.relu,
            eqx.nn.ConvTranspose3d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1, key=key4),
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x  #.clip(0,1)

    def call_shunt(self, x):
        x = self(x)
        # Round to the nearest admissible value
        admissible_values = jnp.array([0.0, 0.33, 0.66, 0.99])
        x = jnp.argmin(jnp.abs(x[..., None] - admissible_values), axis=-1)
        x = admissible_values[x]
        return x


class Autoencoder(eqx.Module):
    encoder: Encoder
    decoder: Decoder

    def __init__(self, key, N, L):
        key1, key2 = jax.random.split(key)
        self.encoder = Encoder(key1, N, L)
        self.decoder = Decoder(key2, N, L)

    def __call__(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)

    def call_shunt(self, x):
        latent = self.encoder(x)
        return self.decoder.call_shunt(latent)

def prepare_batch(x):
    if len(x.shape) == 3:
        return x[None,:,:,:]
    if len(x.shape) == 4:
        x = jnp.expand_dims(x, axis=1)  # Add channel dimension
        return x