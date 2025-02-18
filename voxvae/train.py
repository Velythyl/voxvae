import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import wandb

from voxvae.dataloader import get_dataloaders
from voxvae.metrics import accuracy, metrics
from voxvae.model import Autoencoder, prepare_batch
from voxvae.o3d_utils import plotly_v

# Loss function
def loss_fn(model, x):
    reconstructed = jax.vmap(model)(x)
    return jnp.mean((x - reconstructed) ** 2)  # MSE loss

# Training step
@eqx.filter_jit
def train_step(optimizer, model, opt_state, x):
    x = prepare_batch(x)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

# Training loop
def train(key, model, train_loader, test_loader, val_loader, optimizer, num_epochs):
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i in range(train_loader.num_batch_per_epoch):
            key, rng = jax.random.split(key)
            train_loader, x = train_loader.get_batch_(rng)
            model, opt_state, loss = train_step(optimizer, model, opt_state, x)
            epoch_loss += loss

        epoch_loss = epoch_loss / train_loader.num_batch_per_epoch

        key, rng = jax.random.split(key)
        wandb_dict = metrics(rng, model, test_loader, "test")
        wandb_dict.update(metrics(rng, model, val_loader, "val"))
        wandb_dict["train/loss"] = epoch_loss
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