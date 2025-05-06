import os
import random
from pathlib import Path

import wandb
import yaml
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


def load_wandbconfig_as_hydraconfig(path):
    if isinstance(path, str):
        path = Path(path)
    if not str(path).endswith('files'):
        path = path / 'files'
    assert os.path.exists(path)

    cfg = yaml.load(open(path / "config.yaml"), Loader=yaml.SafeLoader)

    # Step 2: Strip `value` wrappers
    def unwrap_values(d):
        if isinstance(d, dict):
            if "value" in d and len(d) == 1:
                return unwrap_values(d["value"])
            else:
                return {k: unwrap_values(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [unwrap_values(i) for i in d]
        else:
            return d

    clean_cfg = unwrap_values(cfg)
    return OmegaConf.create(clean_cfg)
