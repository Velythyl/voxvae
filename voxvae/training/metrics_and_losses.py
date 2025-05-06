import numpy as np
import torch
import wandb

from voxvae.pcd.pcd_vis import plotly_v
from voxvae.training.models.autoencoder import call_shunt


def accuracy_nonzero_voxels(model, batch):
    pred_batch = call_shunt(model, batch) # model.call_shunt(batch)

    batch = batch.flatten()
    pred_batch = pred_batch.flatten()

    nonzero_mask = torch.logical_not(batch == 0)
    num_nonzero = nonzero_mask.sum()

    return (((batch == pred_batch) * nonzero_mask).sum() / num_nonzero).cpu().item()

def accuracy(model, batch):
    pred_batch = call_shunt(model, batch)

    batch = batch.flatten()
    pred_batch = pred_batch.flatten()

    return (batch == pred_batch).float().mean().cpu().item()


def get_vis(model, x, prefix, midfix=""):
    x = x[None]
    pred_x = call_shunt(model, x) #model.call_shunt(x[None, :, :, :])

    min, max = x.min(), x.max()
    assert min == 0
    x = x / max

    _min, _max = pred_x.min(), pred_x.max()
    assert _max <= max
    pred_x = pred_x / max

    x = x.squeeze().cpu().numpy()
    pred_x = pred_x.squeeze().cpu().numpy()

    return {
        f"{prefix}/{midfix}_gt": wandb.Html(plotly_v(x)),
        f"{prefix}/{midfix}_pred": wandb.Html(plotly_v(pred_x))
    }

def vis(model, loader, prefix):
    for i, batch in enumerate(loader):
        if i == 0:
            first_element = batch[0]
    last_element = batch[-1]

    dico = get_vis(model, first_element, prefix, "A")
    dico.update(get_vis(model, last_element, prefix, "B"))
    return dico

def metrics(model, loader, prefix, loss_fn):
    accs_nonzero = []
    accs_full = []
    losses = []
    for batch in loader:
        accs_nonzero.append(accuracy_nonzero_voxels(model, batch))
        accs_full.append(accuracy(model, batch))

        pred_batch = model(batch)
        losses.append(loss_fn(pred_batch, batch.squeeze(1).long()).cpu().item())

    dico = {}
    dico[f"{prefix}/acc_nonzero"] = np.mean(np.array(accs_nonzero))
    dico[f"{prefix}/acc_full"] = np.mean(np.array(accs_full))
    dico[f"{prefix}/losses"] = np.mean(np.array(losses))

    return dico
