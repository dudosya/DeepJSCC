from __future__ import annotations

import torch
from torch import nn


def _conv_block(in_ch: int, out_ch: int, *, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _deconv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    def __init__(self, *, base_channels: int = 64, latent_channels: int = 16) -> None:
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            _conv_block(3, c, stride=2),  # 32 -> 16
            _conv_block(c, c * 2, stride=2),  # 16 -> 8
            _conv_block(c * 2, c * 4, stride=2),  # 8 -> 4
            nn.Conv2d(c * 4, latent_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, *, base_channels: int = 64, latent_channels: int = 16, output_activation: str = "sigmoid") -> None:
        super().__init__()
        if output_activation != "sigmoid":
            raise ValueError("MVP expects sigmoid output_activation")

        c = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, c * 4, kernel_size=1, stride=1, padding=0),
            _deconv_block(c * 4, c * 2),  # 4 -> 8
            _deconv_block(c * 2, c),  # 8 -> 16
            _deconv_block(c, c),  # 16 -> 32
            nn.Conv2d(c, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DeepJSCCAutoencoder(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        *,
        power_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.power_normalize = power_normalize

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        return self.decoder(y)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # `y` is the channel output already.
        _ = x
        return self.decode(y)
