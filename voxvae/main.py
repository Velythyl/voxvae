import os

from voxvae.utils.wandb_hydra import signals

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import shutil
import signal
import sys

import hydra
import numpy as np
import wandb




@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    sys.stderr = sys.stdout

    import jax.numpy as jnp
    print("Smoketest for jax...")
    print(jnp.ones(10) * 2 * jnp.ones(10))
    print("...passed!")

    from voxvae.utils.wandb_hydra import wandb_init

    wandb_init(cfg)
    signals()

    #def exit_self(self, signum: int, frame = None) -> None: # fixme mila cluster weird
    #    exit(0) # exit normally
    #signal.signal(signal.SIGTERM, exit_self)

    if cfg.datarep.onehot:
        values = [cfg.datarep.pcd_is, cfg.datarep.pcd_isnotis, cfg.datarep.pcd_isnot]
        assert np.all([isinstance(x, int) for x in values])
        assert np.sum(values) == 6  # 1 + 2 + 3

    from voxvae.dataloading.dataloader import get_dataloaders
    # Load data
    splitloaders = get_dataloaders(
        root=cfg.dataloader.data_path,
        grid_size=cfg.dataloader.grid_size,
        batch_size=cfg.dataloader.batch_size,
        fewer_files=cfg.dataloader.fewer_files,
        splits=cfg.dataloader.splits,
        pcd_is=cfg.datarep.pcd_is,
        pcd_isnotis=cfg.datarep.pcd_isnotis,
        pcd_isnot=cfg.datarep.pcd_isnot,
        data_aug=cfg.dataloader.data_aug,
    )

    # Initialize model and optimizer
    from voxvae.training.models.autoencoder import build_model
    model = build_model(cfg.model)
    model = model.cuda()

    import torch
    if cfg.optimizer.type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.learning_rate)
    else:
        raise NotImplementedError(f"Optimizer {cfg.optimizer.type} is not supported")

    device=torch.device('cuda')
    from torch import nn

    if cfg.loss.weighted_loss:
        class_weights = splitloaders.class_weights.to(device=device)
    else:
        class_weights = None

    if cfg.loss.type == "CE":
        loss_func = nn.CrossEntropyLoss(weight=class_weights)
    elif cfg.loss.type == "FL":
        from voxvae.training.losses.focal_loss import focal_loss
        loss_func = focal_loss(alpha=class_weights, gamma=cfg.loss.gamma, device=device)


    #if cfg.optimizer.clip_by_global_norm:
    #    optimizer = optax.chain(
    #        optax.clip_by_global_norm(cfg.optimizer.clip_by_global_norm),
    #        optimizer
    #    )

    # Train the model
    from voxvae.training.train import train
    trained_model = train(model, splitloaders, optimizer, num_epochs=cfg.train.num_epochs, loss_func=loss_func,  evaltestcfg=cfg.evaltest, device=device)

    wandb.finish()


if __name__ == "__main__":
    main()


"""

python3 voxvae/main.py --multirun hydra/launcher=sbatch +hydra/sweep=sbatch hydra.launcher.timeout_min=5700 hydra.launcher.gres=gpu:l40s:1 hydra.launcher.cpus_per_task=8 hydra.launcher.mem_gb=32 hydra.launcher.array_parallelism=80 hydra.launcher.partition=main meta.project=voxvae_notest meta.run_name=plswork meta.seed=-1 dataloader.fewer_files=-1 dataloader.splits="train-/test/-/train/" loss=ce_yesW model=resnet_new_decoder model.latent_size=16 dataloader.data_path=/network/scratch/c/charlie.gauthier/unimals_100 dataloader.batch_size=256 dataloader.data_aug.patch_shuf=False dataloader.data_aug.hybridize=False dataloader.data_aug.mutate=False
"""