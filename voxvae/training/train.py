import functools

import jax
import jax.numpy as jnp
import equinox as eqx
import wandb
from jax._src.tree_util import Partial

from voxvae.training.metrics import metrics, vis
from voxvae.training.models.prepare_batch import prepare_batch


# Loss function
def mse_loss_fn(model, x):
    reconstructed = jax.vmap(model)(x)
    return jnp.mean((x - reconstructed) ** 2)  # MSE loss

def cat_loss_fn(proportions, model, x):
    """
    Categorical cross-entropy loss for the classification approach.
    - proportions: Array of shape (4,) containing the percentage of each class in the dataset.
    - model: The autoencoder model.
    - x: Input voxel grid of shape (batch_size, 1, 32, 32, 32).
    """
    # Get the reconstructed output (shape: batch_size, 4, 32, 32, 32)
    reconstructed = jax.vmap(model)(x)

    x = x.squeeze(1).astype(jnp.int32)  # Remove channel dim and convert to int
    x = jax.nn.one_hot(x, num_classes=4)
    x = jnp.moveaxis(x, -1, 1)

    x = x.reshape((x.shape[0], -1, 4))
    reconstructed = reconstructed.reshape((x.shape[0], -1, 4))

    def per_element(proportions, gt, pred):
        # Compute the cross-entropy loss
        cross_entropy = -jnp.sum(gt * pred, axis=1)

        # Apply the weights
        weighted_loss = cross_entropy * proportions[jnp.argmax(gt, axis=1)]

        # Return the mean loss
        return weighted_loss.mean()

    per_element_loss = jax.vmap(functools.partial(per_element, 1/proportions))(x, reconstructed)
    return per_element_loss.mean()


    # One-hot encode the ground truth voxel types
    x_labels = x.squeeze(1).astype(jnp.int32)  # Remove channel dim and convert to int
    x_one_hot = jax.nn.one_hot(x_labels, num_classes=4)  # Shape: (batch_size, 32, 32, 32, 4)
    x_one_hot = jnp.moveaxis(x_one_hot, -1, 1)

    # Compute the cross-entropy loss
    log_probs = reconstructed # jnp.log(reconstructed)  # Log of predicted probabilities
    cross_entropy = jnp.sum(x_one_hot * log_probs, axis=(1, 2, 3, 4))  # Sum over spatial and class dimensions

    # Apply class weighting using the provided proportions
    class_weights = 1.0 / (proportions + 1e-8)  # Inverse of class proportions (add epsilon to avoid division by zero)
    weighted_loss = cross_entropy * class_weights[x_labels]  # Weight the loss for each voxel

    # Return the mean loss over the batch
    return -jnp.mean(weighted_loss)

# Training step
@eqx.filter_jit
def train_step(loss_func: Partial, optimizer, model, opt_state, x):
    x = prepare_batch(x)
    loss, grads = eqx.filter_value_and_grad(loss_func)(model, x)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

# Training loop
def train(key, model, splitloaders, optimizer, num_epochs, evaltestcfg, use_onehot):
    train_dl = splitloaders.train
    val_dl = splitloaders.val
    test_dl = splitloaders.test

    loss_func = mse_loss_fn
    if use_onehot:
        loss_func = functools.partial(cat_loss_fn, jnp.array([splitloaders.prop_empty, splitloaders.prop_is, splitloaders.prop_isnotis, splitloaders.prop_isnot]))

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i in range(train_dl.num_batch_per_epoch):
            key, rng = jax.random.split(key)
            train_dl, x = train_dl.get_batch_(rng)
            model, opt_state, loss = train_step(loss_func, optimizer, model, opt_state, x)
            epoch_loss += loss

        epoch_loss = epoch_loss / train_dl.num_batch_per_epoch

        wandb_dict = {}
        if epoch % evaltestcfg.metrics_log_freq == 0:
            key, rng = jax.random.split(key)
            wandb_dict.update(metrics(rng, model, test_dl, "test", loss_func))
            key, rng = jax.random.split(key)
            wandb_dict.update(metrics(rng, model, val_dl, "val", loss_func))
            wandb_dict["train/loss"] = epoch_loss

        if epoch % evaltestcfg.vis_log_freq == 0:
            key, rng = jax.random.split(key)
            wandb_dict.update(vis(rng, model, test_dl, "test"))
            key, rng = jax.random.split(key)
            wandb_dict.update(vis(rng, model, val_dl, "val"))

        if len(wandb_dict) > 0:
            wandb.log(wandb_dict)

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")
    return model

"""
# Load data
train_loader, test_loader, val_loader = get_dataloaders("/home/charlie/Desktop/MJCFConvert/mjcf2o3d/unimals_100/dynamics/xml", 32, 20)

# Initialize model and optimizer
key = jax.random.PRNGKey(0)
key, rng = jax.random.split(key)
model = Autoencoder(rng, 32, 128)
optimizer = optax.adam(learning_rate=1e-3)

# Train the model
#trained_model = train(key, model, train_loader, optimizer, num_epochs=1)

_, x = train_loader.get_batch_(key)
x = x[0]
plotly_v(x)

pred_x = model.call_shunt(x[None,:,:,:])
plotly_v(pred_x)
"""