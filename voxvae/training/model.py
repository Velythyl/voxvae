import equinox as eqx
import jax
from jax import numpy as jnp

from voxvae.utils.jaxutils import split_key


class Conv3D_Encoder(eqx.Module):
    convlayers: list
    skiplayers: list
    embedlayers: list
    firstlast: bool

    def __init__(self, key, N, L, skip_firstlast=False):
        self.firstlast = skip_firstlast

        _, (key1, key2, key3, key4) = split_key(key, 4)

        self.convlayers = [
            eqx.nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1, key=key1),
            jax.nn.relu,
            eqx.nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, key=key2),
            jax.nn.relu,
            eqx.nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1, key=key3),
            jax.nn.relu,
        ]

        self.skiplayers = [lambda y, x: (y, x)] if not self.firstlast else [
            lambda y, x: (y, Downsample3D(x, (N // (N // 8)))),
            lambda y, x: (y, jnp.repeat(x, 128, axis=0).at[128//2:].set(0)),    # add the value information only to half the channels
            lambda y,x: (y + x, None)
        ]

        self.embedlayers = [lambda x: jnp.reshape(x, (x.shape[0], -1)).reshape(128 * (N // 8) ** 3),  # Flatten
            eqx.nn.Linear(128 * (N // 8) ** 3, L, key=key4),  # Adjust N and L as needed
        ]

    def __call__(self, x):
        conved = x
        for layer in self.convlayers:
            conved = layer(conved)

        mid = conved, x
        for layer in self.skiplayers:
            mid = layer(mid[0], mid[1])
        mid = mid[0]

        final = mid
        for layer in self.embedlayers:
            final = layer(final)
        return final


class Conv3D_Decoder(eqx.Module):
    layers: list
    use_softmax:  bool

    def __init__(self, key, N, L, use_onehot=False):
        self.use_softmax = use_onehot

        if use_onehot:
            # For classification, output 4 channels (one for each voxel type)
            final_output_channels = 4
        else:
            # For regression, output 1 channel
            final_output_channels = 1

        _, (key1, key2, key3, key4) = split_key(key, 4)
        self.layers = [
            eqx.nn.Linear(L, 128 * (N // 8) ** 3, key=key1),
            lambda x: jnp.reshape(x, (128, N // 8, N // 8, N // 8)),  # Reshape
            jax.nn.relu,
            eqx.nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, key=key2),
            jax.nn.relu,
            eqx.nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, key=key3),
            jax.nn.relu,
            eqx.nn.ConvTranspose3d(32, final_output_channels, kernel_size=3, stride=2, padding=1, output_padding=1,
                                   key=key4),
        ]

    def __call__(self, x):
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

class ResidualConv3D_Decoder(eqx.Module):
    layers: list
    use_softmax:  bool

    def __init__(self, key, N, L, use_onehot=False):
        self.use_softmax = use_onehot

        if use_onehot:
            # For classification, output 4 channels (one for each voxel type)
            final_output_channels = 4
        else:
            # For regression, output 1 channel
            final_output_channels = 1

        _, (key1, key2, key3, key4) = split_key(key, 4)

        self._latent_decoder = [eqx.nn.Linear(L, 128 * (N // 8) ** 3, key=key1),
            lambda x: jnp.reshape(x, (128, N // 8, N // 8, N // 8)),  # Reshape
            jax.nn.relu]
        self.layers = [
            eqx.nn.Linear(L, 128 * (N // 8) ** 3, key=key1),
            lambda x: jnp.reshape(x, (128, N // 8, N // 8, N // 8)),  # Reshape
            jax.nn.relu,
            eqx.nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, key=key2),
            jax.nn.relu,
            eqx.nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, key=key3),
            jax.nn.relu,
            eqx.nn.ConvTranspose3d(32, final_output_channels, kernel_size=3, stride=2, padding=1, output_padding=1,
                                   key=key4),
        ]

    def __call__(self, x):
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

class Autoencoder(eqx.Module):
    encoder: Conv3D_Encoder
    decoder: Conv3D_Decoder

    def __init__(self, encoder, decoder):
        #key1, key2 = jax.random.split(key)
        self.encoder = encoder #Conv3D_Encoder(key1, N, L)
        self.decoder = decoder # Conv3D_Decoder(key2, N, L, use_onehot=use_onehot)

    def __call__(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)

    def call_shunt(self, x):
        latent = self.encoder(x)
        return self.decoder.call_shunt(latent)


def prepare_batch(x):
    if len(x.shape) == 3:
        return x[None, :, :, :]
    if len(x.shape) == 4:
        x = jnp.expand_dims(x, axis=1)  # Add channel dimension
        return x

def build_model(key, grid_size, use_onehot, model_cfg):
    if model_cfg.encoder.type == "conv3d":
        key, rng = jax.random.split(key)
        encoder = Conv3D_Encoder(key, grid_size, model_cfg.latent_size, skip_firstlast=model_cfg.encoder.skip_firstlast)
    if model_cfg.decoder.type == "conv3d":
        key, rng = jax.random.split(key)
        decoder = Conv3D_Decoder(key, grid_size, model_cfg.latent_size, use_onehot=use_onehot)

    return Autoencoder(encoder, decoder)




