import jax
import torch

from voxvae.training.models import cnn3d, resnet_cnn3d, resnet_fullcnn3d, fullcnn3d
from voxvaetorch.training.models import fca


def call_shunt(model, batch):
    pred_batch = model(batch)
    return torch.argmax(pred_batch, dim=1)

def build_model(model_cfg):
    if model_cfg.type == "fullconv3d":
        autoencoder = fca.Autoencoder((1,32,32,32), model_cfg.latent_size, 4)

    return autoencoder
