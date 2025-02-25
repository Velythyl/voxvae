import random

import wandb
from omegaconf import omegaconf, OmegaConf


def set_seed(cfg, meta_key="meta"):
    seed = cfg[meta_key]["seed"]
    if seed == -1:
        seed = random.randint(0, 20000)
        cfg[meta_key]["seed"] = seed

def wandb_init(cfg, meta_key="meta"):
    set_seed(cfg,meta_key)

    run = wandb.init(
        # entity=cfg.wandb.entity,
        project=cfg[meta_key].project,
        name=cfg[meta_key]["run_name"],  # todo
        save_code=True,
        settings=wandb.Settings(start_method="thread", code_dir=".."),
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        tags=cfg[meta_key]["tags"],
        # mode="disabled"
    )

    cfg_yaml = OmegaConf.to_yaml(cfg)
    print(cfg_yaml)
    with open(f"{wandb.run.dir}/hydra_config.yaml", "w") as f:
        f.write(cfg_yaml)

    return run