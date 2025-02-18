import jax
import numpy as np
import wandb

from voxvae.model import prepare_batch

import jax.numpy as jnp
import equinox as eqx
from voxvae.o3d_utils import plotly_v


@eqx.filter_jit
def accuracy(model, batch):
    batch = prepare_batch(batch)
    pred_batch = jax.vmap(model.call_shunt)(batch)

    batch = batch.flatten()
    pred_batch = pred_batch.flatten()

    nonzero_mask = jnp.logical_not(batch == 0)
    num_nonzero = nonzero_mask.sum()

    return ((batch == pred_batch) * nonzero_mask).sum() / num_nonzero


def get_vis(model, x, prefix, midfix=""):
    return {
        f"{prefix}/{midfix}_gt": wandb.Html(plotly_v(x)),
        f"{prefix}/{midfix}_pred": wandb.Html(plotly_v(model.call_shunt(x[None, :, :, :])))
    }

def metrics(key, model, loader, prefix):
    _, first_batch = loader.get_batch_(key)
    first_item = first_batch[0]

    accs = []
    for _ in range(loader.num_batch_per_epoch):
        loader, batch = loader.get_batch_(key)
        accs.append(accuracy(model, batch))

    dico = get_vis(model, first_item, prefix, "A")
    dico.update(
        get_vis(model, first_item, prefix, "B"),
    )
    dico[f"{prefix}/acc"] = np.mean(np.array(accs))

    return dico