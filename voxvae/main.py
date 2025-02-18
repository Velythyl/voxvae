import shutil

import hydra
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import plotly
import wandb

from voxvae.dataloader import get_dataloaders
from voxvae.jaxutils import split_key
from voxvae.model import Autoencoder
from voxvae.o3d_utils import plotly_v

from hydra import initialize, compose
from omegaconf import OmegaConf

from voxvae.train import train
from voxvae.wandb_hydra import wandb_init


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    wandb_init(cfg)

    # Load data
    train_loader, test_loader, val_loader = get_dataloaders(
        cfg.dataloader.data_path,
        cfg.dataloader.grid_size,
        cfg.dataloader.batch_size,
    )

    # Initialize model and optimizer
    key = jax.random.PRNGKey(cfg["meta"].seed)
    key, rng = jax.random.split(key)
    model = Autoencoder(rng, cfg.dataloader.grid_size, cfg.model.latent_size)

    if cfg.optimizer.type == "adam":
        optimizer = optax.adam(learning_rate=cfg.optimizer.learning_rate)
    else:
        raise NotImplementedError(f"Optimizer {cfg.optimizer.type} is not supported")

    # Train the model
    trained_model = train(key, model, train_loader, test_loader, val_loader, optimizer, num_epochs=cfg.train.num_epochs)

    eqx.tree_serialise_leaves("trained.eqx", trained_model)

    shutil.copy("trained.eqx", f"{wandb.run.dir}/trained.eqx")

    return
    _, x = train_loader.get_batch_(key)
    x = x[0]
    wandb.log({"vis/gt": wandb.Html(plotly_v(x))})
    wandb.log({"vis/pred": wandb.Html(plotly_v(trained_model.call_shunt(x[None, :, :, :])))})
    wandb.finish()

    pred_x = trained_model.call_shunt(x[None, :, :, :])
    plotly_v(pred_x)


if __name__ == "__main__":
    main()
