import numpy as np
import torch
import wandb

from voxvae.pcd.pcd_vis import plotly_v
from voxvae.training.models.autoencoder import call_shunt


def accuracy_nonzero_voxels(model, batch):
    pred_batch = call_shunt(model, batch)

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


def precision_recall_per_class(model, batch, ignore_class=0):
    pred_batch = call_shunt(model, batch)

    batch = batch.flatten()
    pred_batch = pred_batch.flatten()

    classes = torch.unique(batch)
    metrics = {}

    for cls in classes:
        if cls.item() == ignore_class:
            continue  # skip background or ignored class

        true_positive = ((pred_batch == cls) & (batch == cls)).sum()
        false_positive = ((pred_batch == cls) & (batch != cls)).sum()
        false_negative = ((pred_batch != cls) & (batch == cls)).sum()

        precision = true_positive / (true_positive + false_positive + 1e-8)
        recall = true_positive / (true_positive + false_negative + 1e-8)

        metrics[int(cls.item())] = {
            "precision": precision.item(),
            "recall": recall.item(),
        }

    return metrics


def get_vis(model, x, prefix, midfix=""):
    x = x[None]
    pred_x = call_shunt(model, x)

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
    if loader is None:
        return {}
    for i, batch in enumerate(loader):
        if i == 0:
            first_element = batch[0]
    last_element = batch[-1]

    dico = get_vis(model, first_element, prefix, "A")
    dico.update(get_vis(model, last_element, prefix, "B"))
    return dico


def metrics(model, loader, prefix, loss_fn):
    if loader is None:
        return {}
    accs_nonzero = []
    accs_full = []
    losses = []
    all_precisions = []
    all_recalls = []

    for batch in loader:
        accs_nonzero.append(accuracy_nonzero_voxels(model, batch))
        accs_full.append(accuracy(model, batch))

        pred_batch = model(batch)
        losses.append(loss_fn(pred_batch, batch.squeeze(1).long()).cpu().item())

        pr_metrics = precision_recall_per_class(model, batch)
        if pr_metrics:
            all_precisions.append(np.mean([m["precision"] for m in pr_metrics.values()]))
            all_recalls.append(np.mean([m["recall"] for m in pr_metrics.values()]))

    dico = {}
    dico[f"{prefix}/acc_nonzero"] = np.mean(np.array(accs_nonzero))
    dico[f"{prefix}/acc_full"] = np.mean(np.array(accs_full))
    dico[f"{prefix}/losses"] = np.mean(np.array(losses))

    if all_precisions and all_recalls:
        dico[f"{prefix}/precision_avg"] = np.mean(np.array(all_precisions))
        dico[f"{prefix}/recall_avg"] = np.mean(np.array(all_recalls))

    return dico
