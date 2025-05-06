import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT3DAutoencoder(nn.Module):
    def __init__(self, input_shape, latent_size, num_classes, patch_size=8, dim=512, depth=6, heads=8, mlp_dim=1024,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.input_shape = input_shape  # (C, D, H, W)
        self.latent_size = latent_size
        self.num_classes = num_classes

        channels, depth_dim, height, width = input_shape
        patch_depth, patch_height, patch_width = (patch_size, patch_size, patch_size)

        assert depth_dim % patch_depth == 0 and height % patch_height == 0 and width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (depth_dim // patch_depth) * (height // patch_height) * (width // patch_width)
        patch_dim = channels * patch_depth * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (d p1) (h p2) (w p3) -> b (d h w) (p1 p2 p3 c)', p1=patch_depth, p2=patch_height,
                      p3=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # Latent projection
        self.latent_proj = nn.Linear(dim, latent_size)

        # Decoder setup
        self.decoder_embed = nn.Linear(latent_size, dim)
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.decoder_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.decoder_to_patch = nn.Sequential(
            nn.Linear(dim, patch_dim),
            Rearrange('b (d h w) (p1 p2 p3 c) -> b c (d p1) (h p2) (w p3)',
                      p1=patch_depth, p2=patch_height, p3=patch_width,
                      d=depth_dim // patch_depth, h=height // patch_height, w=width // patch_width),
        )

        # Final projection to number of classes
        self.final_proj = nn.Conv3d(channels, num_classes, kernel_size=1)

    def forward(self, x):
        b, c, d, h, w = x.shape

        # Encode
        x = self.to_patch_embedding(x)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(x.shape[1])]
        x = self.dropout(x)

        x = self.transformer(x)

        # Get latent from cls token
        latent = x[:, 0]
        latent = self.latent_proj(latent)

        # Decode
        x = self.decoder_embed(latent)
        cls_tokens = x.unsqueeze(1)
        x = repeat(self.decoder_pos_embed[:, 1:], '1 n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.decoder_transformer(x)
        x = x[:, 1:]  # Remove cls token

        x = self.decoder_to_patch(x)
        x = self.final_proj(x)

        return x

    def get_latent(self, x):
        b, c, d, h, w = x.shape

        x = self.to_patch_embedding(x)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(x.shape[1])]
        x = self.dropout(x)

        x = self.transformer(x)
        latent = x[:, 0]
        latent = self.latent_proj(latent)

        return latent.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Shape to match CNN version (B, L, 1, 1, 1)