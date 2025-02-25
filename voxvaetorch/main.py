import shutil
import signal

import hydra
import numpy as np
import wandb




@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    from voxvaetorch.utils.wandb_hydra import wandb_init

    wandb_init(cfg)

    #def exit_self(self, signum: int, frame = None) -> None: # fixme mila cluster weird
    #    exit(0) # exit normally
    #signal.signal(signal.SIGTERM, exit_self)

    if cfg.datarep.onehot:
        values = [cfg.datarep.pcd_is, cfg.datarep.pcd_isnotis, cfg.datarep.pcd_isnot]
        assert np.all([isinstance(x, int) for x in values])
        assert np.sum(values) == 6  # 1 + 2 + 3

    from voxvaetorch.dataloading.dataloader import get_dataloaders
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
    from voxvaetorch.training.models.autoencoder import build_model
    model = build_model(cfg.model)
    model = model.cuda()

    import torch
    if cfg.optimizer.type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.learning_rate)
    else:
        raise NotImplementedError(f"Optimizer {cfg.optimizer.type} is not supported")

    #if cfg.optimizer.clip_by_global_norm:
    #    optimizer = optax.chain(
    #        optax.clip_by_global_norm(cfg.optimizer.clip_by_global_norm),
    #        optimizer
    #    )

    # Train the model
    from voxvaetorch.training.train import train
    trained_model = train(model, splitloaders, optimizer, num_epochs=cfg.train.num_epochs, evaltestcfg=cfg.evaltest)

    wandb.finish()


if __name__ == "__main__":
    main()
