import torch
import torch.nn as nn
import wandb

from voxvae.training.metrics_and_losses import metrics, vis


# Training step
def train_step(loss_func, optimizer, model, x):
    optimizer.zero_grad()
    output = model(x)

    loss = loss_func(output, x.squeeze(1).long())
    loss.backward()
    optimizer.step()
    return loss.item()

# Training loop
def train(model, splitloaders, optimizer, num_epochs, loss_func, evaltestcfg, device):
    train_dl = splitloaders.train
    val_dl = splitloaders.val
    test_dl = splitloaders.test



    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        # Training loop
        for x in train_dl:
            x = x.to(device)  # Move data to device (e.g., GPU)
            loss = train_step(loss_func, optimizer, model, x)
            epoch_loss += loss

        epoch_loss /= len(train_dl)

        # Logging and evaluation
        wandb_dict = {}
        if epoch % evaltestcfg.metrics_log_freq == 0 and epoch > 0: # skip first because loss is super high on first pass
            model.eval()
            with torch.no_grad():
                wandb_dict.update(metrics(model, test_dl, "test", loss_func))
                wandb_dict.update(metrics(model, val_dl, "val", loss_func))
            wandb_dict["train/loss"] = epoch_loss
            model.train()

        if epoch % evaltestcfg.vis_log_freq == 0: # keep first to make sure vis works
            model.eval()
            wandb_dict.update(vis( model, test_dl, "test"))
            wandb_dict.update(vis( model, val_dl, "val"))
            model.train()

        if epoch % evaltestcfg.model_log_freq == 0 and epoch > 0: # skip first because model is untrained
            model.eval()
            torch.save(model.state_dict(), f"{wandb.run.dir}/trained_{epoch}.pt")
            model.train()


        if len(wandb_dict) > 0:
            wandb.log(wandb_dict)

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")

    return model
