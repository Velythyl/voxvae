import shutil

import hydra
import jax
import equinox as eqx
import numpy as np
import optax
import wandb

from voxvae.dataloading.dataloader import get_dataloaders
from voxvae.training.models.autoencoder import build_model

from voxvae.training.train import train
from voxvae.utils.wandb_hydra import wandb_init


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    wandb_init(cfg)

    def exit_self(self, signum: int, frame = None) -> None: # fixme mila cluster weird
        exit(0) # exit normally
    signal.signal(signal.SIGTERM, exit_self)

    if cfg.datarep.onehot:
        values = [cfg.datarep.pcd_is, cfg.datarep.pcd_isnotis, cfg.datarep.pcd_isnot]
        assert np.all([isinstance(x, int) for x in values])
        assert np.sum(values) == 6  # 1 + 2 + 3

    # Load data
    splitloaders = get_dataloaders(
        cfg.dataloader.data_path,
        cfg.dataloader.grid_size,
        cfg.dataloader.batch_size,
        cfg.dataloader.fewer_files,
        pcd_is=cfg.datarep.pcd_is,
        pcd_isnotis=cfg.datarep.pcd_isnotis,
        pcd_isnot=cfg.datarep.pcd_isnot,
    )

    # Initialize model and optimizer
    key = jax.random.PRNGKey(cfg["meta"].seed)
    key, rng = jax.random.split(key)
    model = build_model(rng, cfg.dataloader.grid_size, cfg.datarep.onehot, cfg.model)

    if cfg.optimizer.type == "adam":
        optimizer = optax.adam(learning_rate=cfg.optimizer.learning_rate)
    else:
        raise NotImplementedError(f"Optimizer {cfg.optimizer.type} is not supported")

    # Train the model
    trained_model = train(key, model, splitloaders, optimizer, num_epochs=cfg.train.num_epochs, evaltestcfg=cfg.evaltest, use_onehot=cfg.datarep.onehot)

    eqx.tree_serialise_leaves("trained.eqx", trained_model)

    shutil.copy("trained.eqx", f"{wandb.run.dir}/trained.eqx")
    wandb.finish()


if __name__ == "__main__":
    main()
