from typing import Any

import equinox as eqx
import jax

from voxvae.training.models import cnn3d, resnet_cnn3d, resnet_fullcnn3d


class Autoencoder(eqx.Module):
    encoder: Any
    decoder: Any

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


def build_model(key, grid_size, use_onehot, model_cfg):
    if model_cfg.encoder.type == "conv3d":
        key, rng = jax.random.split(key)
        encoder = cnn3d.Conv3D_Encoder(key, grid_size, model_cfg.latent_size, skip_firstlast=model_cfg.encoder.skip_firstlast)
    elif model_cfg.encoder.type == "resconv3d":
        key, rng = jax.random.split(key)
        encoder = resnet_cnn3d.ResConv3D_Encoder(key, grid_size, model_cfg.latent_size, deeper_embed=model_cfg.encoder.deeper_embed)
    elif model_cfg.encoder.type == "resfullconv3d":
        key, rng = jax.random.split(key)
        encoder = resnet_fullcnn3d.ResConv3D_Encoder(key, grid_size, model_cfg.latent_size)

    if model_cfg.decoder.type == "conv3d":
        key, rng = jax.random.split(key)
        decoder = cnn3d.Conv3D_Decoder(key, grid_size, model_cfg.latent_size, use_onehot=use_onehot)
    elif model_cfg.decoder.type == "resconv3d":
        key, rng = jax.random.split(key)
        decoder = resnet_cnn3d.ResConv3D_Decoder(key, grid_size, model_cfg.latent_size, use_onehot=use_onehot, deeper_embed=model_cfg.encoder.deeper_embed)
    elif model_cfg.decoder.type == "resfullconv3d":
        key, rng = jax.random.split(key)
        decoder = resnet_fullcnn3d.ResConv3D_Decoder(key, grid_size, model_cfg.latent_size, use_onehot=use_onehot)

    return Autoencoder(encoder, decoder)
