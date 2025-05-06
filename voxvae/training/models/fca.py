import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_shape, latent_size, num_classes):
        super(Autoencoder, self).__init__()

        # Input shape is expected to be (C, D, H, W) for voxel data
        self.input_shape = input_shape
        self.latent_size = latent_size
        self.num_classes = num_classes

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(input_shape[0], 32, kernel_size=3, stride=2, padding=1),  # (C, D, H, W) -> (32, D/2, H/2, W/2)
            nn.LeakyReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # (32, D/2, H/2, W/2) -> (64, D/4, H/4, W/4)
            nn.LeakyReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),  # (64, D/4, H/4, W/4) -> (128, D/8, H/8, W/8)
            nn.LeakyReLU(),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),  # (128, D/8, H/8, W/8) -> (256, D/16, H/16, W/16)
            nn.LeakyReLU(),
            nn.Conv3d(256, latent_size, kernel_size=3, stride=2, padding=1),
            # (256, D/16, H/16, W/16) -> (L, D/32, H/32, W/32)
            nn.LeakyReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(latent_size, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            # (L, D/32, H/32, W/32) -> (256, D/16, H/16, W/16)
            nn.LeakyReLU(),
            nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            # (256, D/16, H/16, W/16) -> (128, D/8, H/8, W/8)
            nn.LeakyReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            # (128, D/8, H/8, W/8) -> (64, D/4, H/4, W/4)
            nn.LeakyReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            # (64, D/4, H/4, W/4) -> (32, D/2, H/2, W/2)
            nn.LeakyReLU(),
            nn.ConvTranspose3d(32, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1),
            # (32, D/2, H/2, W/2) -> (num_classes, D, H, W)
        )

    def forward(self, x):
        # Encode
        latent = self.encoder(x)

        # Decode
        logits = self.decoder(latent)  # Output logits (before softmax)

        return logits

    def get_latent(self, x):
        return self.encoder(x)

