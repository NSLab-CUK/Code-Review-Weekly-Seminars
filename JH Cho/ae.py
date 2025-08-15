import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.losses_ae import (
    hybrid_loss, ssim_loss, structural_loss, perceptual_loss
)

class AutoEncoder(nn.Module):
    def __init__(self, loss_type="hybrid"):
        super().__init__()
        self.loss_type = loss_type.lower()

        # Encoder: 1 → 64 → 128 → 256
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.ReLU()
        )

        # Bottleneck: 256 → 512
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), nn.ReLU()
        )

        # Decoder: 512 → 256 → 128 → 64 → 1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1), nn.Sigmoid()
        )

    def minmax_norm(self, x, eps=1e-6):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)
        min_val = x_flat.min(dim=2, keepdim=True)[0].view(B, C, 1, 1)
        max_val = x_flat.max(dim=2, keepdim=True)[0].view(B, C, 1, 1)
        return (x - min_val) / (max_val - min_val + eps)

    def forward(self, x, return_loss=False):
        # 🔒 tuple 방어
        if isinstance(x, tuple):
            x = x[0]
        # 🔒 dtype/스케일 안전
        if x.dtype != torch.float32:
            x = x.float()
        if torch.is_tensor(x) and x.max() > 1.5:
            x = x / 255.0
        # 🔒 3ch 입력이 들어오면 1ch로 변환(가중치 그레이스케일)
        if x.shape[1] == 3:
            r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:2 + 1]
            x = 0.2989 * r + 0.5870 * g + 0.1140 * b  # (B,1,H,W)

        i = x  # original input
        enc = self.encoder(i)
        bottleneck = self.bottleneck(enc)
        i_hat = self.decoder(bottleneck)

        # Normalize reconstruction and diff
        diff_norm = self.minmax_norm(torch.abs(i - i_hat))
        i_hat_norm = self.minmax_norm(i_hat)
        i_norm = self.minmax_norm(i)
        # Merge into 3-channel output
        three_channel_output = torch.cat([i, i_hat_norm, diff_norm], dim=1)

        if return_loss:
            if self.loss_type == "l1":
                loss = F.l1_loss(i_hat, i)
            elif self.loss_type == "ssim":
                loss = ssim_loss(i_hat, i)
            elif self.loss_type == "structural":
                loss = structural_loss(i_hat, i)
            elif self.loss_type == "perceptual":
                loss = perceptual_loss(i_hat, i)
            elif self.loss_type == "hybrid":
                loss = hybrid_loss(i_hat, i)
            else:
                raise ValueError(f"Invalid loss_type: {self.loss_type}")
            return three_channel_output, loss
        else:
            return three_channel_output

    def get_reconstruction(self, x):
        """순수히 i_hat만 반환"""
        enc = self.encoder(x)
        bottleneck = self.bottleneck(enc)
        i_hat = self.decoder(bottleneck)
        return i_hat
