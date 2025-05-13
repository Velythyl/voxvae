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
            # Initial downsampling with more gradual channel increase
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),  # Keep resolution (64,32,32,32)
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),
            nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1),  # (64,16,16,16)
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),

            # Residual blocks with controlled downsampling
            ResidualBlock(64, 128, stride=2),  # (128,8,8,8)
            nn.BatchNorm3d(128),
            ResidualBlock(128, 128, stride=1),  # Preserve resolution
            nn.BatchNorm3d(128),
            ResidualBlock(128, 256, stride=2),  # (256,4,4,4)
            nn.BatchNorm3d(256),
            ResidualBlock(256, 256, stride=1),  # Preserve resolution
            nn.BatchNorm3d(256),

            # Final projection to latent space
            nn.Conv3d(256, 256, kernel_size=3, stride=2, padding=1),  # (256,2,2,2)
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
            nn.Conv3d(256, latent_size, kernel_size=2, stride=1),  # (L,1,1,1) (no padding!)
            # No activation here to avoid constraining latent space
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(latent_size, 256, kernel_size=2, stride=2),  # 1³ → 2³
            nn.BatchNorm3d(256),
            nn.LeakyReLU(),
            ResidualBlock(256, 256),
            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2),  # 2³ → 4³
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            ResidualBlock(128, 128),
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),  # 4³ → 8³
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            ResidualBlock(64, 64),
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),  # 8³ → 16³
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            ResidualBlock(32, 32),
            nn.ConvTranspose3d(32, 32, kernel_size=2, stride=2),  # 16³ → 32³
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.Conv3d(32, num_classes, kernel_size=3, padding=1),
            #nn.Sigmoid()  # or Tanh, depending on your data
        )

    def forward(self, x):
        assert len(x.shape) == 5
        # Linear embedding
        x = x.permute(0, 2, 3, 4, 1)  # (B, C, D, H, W) -> (B, D, H, W, C)
        x = self.embedding(x)  # (B, D, H, W, C) -> (B, D, H, W, 64)
        x = x.permute(0, 4, 1, 2, 3)  # (B, D, H, W, 64) -> (B, 64, D, H, W)

        # Encode
        latent = self.encoder(x)

        # Decode
        logits = self.decoder(latent)  # Output logits (before softmax)

        return logits

    def get_latent(self, x):
        assert len(x.shape) == 5
        # Linear embedding
        x = x.permute(0, 2, 3, 4, 1)  # (B, C, D, H, W) -> (B, D, H, W, C)
        x = self.embedding(x)  # (B, D, H, W, C) -> (B, D, H, W, 64)
        x = x.permute(0, 4, 1, 2, 3)  # (B, D, H, W, 64) -> (B, 64, D, H, W)

        # Encode
        latent = self.encoder(x)
        return latent