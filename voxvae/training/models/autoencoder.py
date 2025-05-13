import torch

from voxvae.training.models import fca, resnet, resnet_upsample, resnet_linear, resnet_otherlinear, vit, fcn, resnet_new_decoder


def call_shunt(model, batch):
    pred_batch = model(batch)
    #pred_batch = torch.nn.functional.log_softmax(pred_batch, dim=1)
    return torch.argmax(pred_batch, dim=1)

def build_model(model_cfg):
    if model_cfg.type == "fca":
        autoencoder = fca.Autoencoder((1,32,32,32), model_cfg.latent_size, 4)
    elif model_cfg.type == "resnet":
        autoencoder = resnet.Autoencoder((1,32,32,32), model_cfg.latent_size, 4)
    elif model_cfg.type == "resnet_upsample":
        autoencoder = resnet_upsample.Autoencoder((1,32,32,32), model_cfg.latent_size, 4)
    elif model_cfg.type == "resnet_linear":
        autoencoder = resnet_linear.Autoencoder((1,32,32,32), model_cfg.latent_size, 4)
    elif model_cfg.type == "resnet_otherlinear":
        autoencoder = resnet_otherlinear.Autoencoder((1,32,32,32), model_cfg.latent_size, 4)
    elif model_cfg.type == "vit":
        autoencoder = vit.ViT3DAutoencoder((1,32,32,32), model_cfg.latent_size, 4)
    elif model_cfg.type == "fcn":
        autoencoder = fcn.Linear3DAutoencoder((1, 32, 32, 32), model_cfg.latent_size, 4)
    elif model_cfg.type == "resnet_new_decoder":
        autoencoder = resnet_new_decoder.Autoencoder((1, 32, 32, 32), model_cfg.latent_size, 4)

    return autoencoder
