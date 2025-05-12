import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class Autoencoder(nn.Module):
    def __init__(self, input_shape, latent_size, num_classes):
        super(Autoencoder, self).__init__()

        # Input shape is expected to be (C, D, H, W) for voxel data
        self.input_shape = input_shape
        self.latent_size = latent_size
        self.num_classes = num_classes

        # Linear embedding layer
        self.embedding = nn.Linear(input_shape[0], 64)  # Project input channels to 64 channels

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),  # (64, D, H, W) -> (32, D, H, W)
            nn.LeakyReLU(),
            ResidualBlock(32, 64, stride=2),  # (32, D, H, W) -> (64, D/2, H/2, W/2)
            ResidualBlock(64, 128, stride=2),  # (64, D/2, H/2, W/2) -> (128, D/4, H/4, W/4)
            ResidualBlock(128, 256, stride=2),  # (128, D/4, H/4, W/4) -> (256, D/8, H/8, W/8)
            nn.Flatten(),  # Flatten spatial dimensions
            nn.Linear(256 * (input_shape[1] // 8) * (input_shape[2] // 8) * (input_shape[3] // 8), latent_size),  # Project to latent space
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 256 * (input_shape[1] // 8) * (input_shape[2] // 8) * (input_shape[3] // 8)),  # Project back to spatial dimensions
            nn.Unflatten(1, (256, input_shape[1] // 8, input_shape[2] // 8, input_shape[3] // 8)),  # Reshape to (256, D/8, H/8, W/8)
            ResidualBlock(256, 128),
            nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2, padding=0),  # (128, D/8, H/8, W/8) -> (128, D/4, H/4, W/4)
            nn.LeakyReLU(),
            ResidualBlock(128, 64),
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2, padding=0),  # (64, D/4, H/4, W/4) -> (64, D/2, H/2, W/2)
            nn.LeakyReLU(),
            ResidualBlock(64, 32),
            nn.ConvTranspose3d(32, 32, kernel_size=2, stride=2, padding=0),  # (32, D/2, H/2, W/2) -> (32, D, H, W)
            nn.LeakyReLU(),
            nn.Conv3d(32, num_classes, kernel_size=1, stride=1, padding=0),  # (32, D, H, W) -> (num_classes, D, H, W)
        )

    def forward(self, x):
        # Linear embedding
        x = x.permute(0, 2, 3, 4, 1)  # (B, C, D, H, W) -> (B, D, H, W, C)
        x = self.embedding(x)  # (B, D, H, W, C) -> (B, D, H, W, 64)
        x = x.permute(0, 4, 1, 2, 3)  # (B, D, H, W, 64) -> (B, 64, D, H, W)

        # Encode
        latent = self.encoder(x)

        # Decode
        reconstruction = self.decoder(latent)  # Output reconstruction
        return reconstruction

    def get_latent(self, x):
        # Linear embedding
        x = x.permute(0, 2, 3, 4, 1)  # (B, C, D, H, W) -> (B, D, H, W, C)
        x = self.embedding(x)  # (B, D, H, W, C) -> (B, D, H, W, 64)
        x = x.permute(0, 4, 1, 2, 3)  # (B, D, H, W, 64) -> (B, 64, D, H, W)

        # Encode
        latent = self.encoder(x)
        return latent