import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear3DAutoencoder(nn.Module):
    def __init__(self, input_shape=32, latent_dim=128, num_classes=10):
        """
        3D Autoencoder using only linear layers.

        Args:
            voxel_size (int): Size of the input voxel grid (V x V x V)
            latent_dim (int): Dimension of the latent space (L)
            num_classes (int): Number of classes for classification (C)
        """
        super(Linear3DAutoencoder, self).__init__()
        import numpy as np
        self.voxel_size = input_shape[1]
        assert self.voxel_size ** 3 == np.prod(input_shape)
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Calculate total number of voxels
        self.total_voxels = self.voxel_size ** 3

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.total_voxels, 16384),
            nn.ReLU(),
            nn.Linear(16384, 8192),
            nn.ReLU(),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )

        # Decoder layers (for reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 8192),
            nn.ReLU(),
            nn.Linear(8192, 16384),
            nn.ReLU(),
            nn.Linear(16384, self.total_voxels),  # Output flattened class predictions
            nn.Linear(self.total_voxels, self.total_voxels*num_classes),
            # Reshape to (batch_size, num_classes, D, H, W)
            nn.Unflatten(1, (num_classes, self.voxel_size, self.voxel_size, self.voxel_size))  # Add spatial dimensions
            # No activation here - we'll use CrossEntropy loss which includes SoftMax
        )

    def get_latent(self, x):
        latent = self.encoder(x)
        return latent

    def forward(self, x):
        # Flatten the input voxel grid
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)

        # Encode to latent space
        latent = self.encoder(x_flat)

        # Decode back to voxel space
        reconstructed_flat = self.decoder(latent)
        reconstructed = reconstructed_flat.view(batch_size, self.voxel_size, self.voxel_size, self.voxel_size)


        return reconstructed, classes


# Example usage
if __name__ == "__main__":
    # Hyperparameters
    V = 32  # Voxel grid size
    L = 128  # Latent dimension
    C = 5  # Number of classes

    # Create model
    model = Linear3DAutoencoder(voxel_size=V, latent_dim=L, num_classes=C)

    # Create dummy input (batch_size=4, V=32)
    dummy_input = torch.rand(4, V, V, V)

    # Forward pass
    reconstructed, classes = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Reconstructed shape:", reconstructed.shape)
    print("Class probabilities shape:", classes.shape)