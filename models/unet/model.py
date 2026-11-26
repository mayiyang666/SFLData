"""经典 U-Net（Ronneberger et al., 2015）实现。

结构：4 级下采样编码 + 双线性上采样解码 + 跳跃连接，
基础通道 64 -> 128 -> 256 -> 512 -> 1024。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x 2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # 尺寸对齐保护（输入非 2^n 时）
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dx != 0 or dy != 0:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """经典 U-Net。

    Args:
        in_channels: 输入通道（RGB=3）
        num_classes: 输出类别数（含背景，本数据集为 4）
        base_channels: 基础通道数（默认 64）
        depth: 下采样级数（默认 4，输入 1024 -> bottleneck 64）
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 4,
                 base_channels: int = 64, depth: int = 4):
        super().__init__()
        assert 1 <= depth <= 5, "depth 建议 1~5"
        c = base_channels
        self.inc = DoubleConv(in_channels, c)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        ch = c
        enc_channels = [ch]
        for _ in range(depth):
            self.downs.append(Down(ch, ch * 2))
            ch *= 2
            enc_channels.append(ch)
        # bottleneck 不再加倍，保持经典 1024 上限
        for i in range(depth - 1, -1, -1):
            in_ch = enc_channels[i + 1] + enc_channels[i]
            out_ch = enc_channels[i]
            self.ups.append(Up(in_ch, out_ch))
        self.outc = nn.Conv2d(c, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [self.inc(x)]
        for d in self.downs:
            feats.append(d(feats[-1]))
        x = feats[-1]
        for i, u in enumerate(self.ups):
            x = u(x, feats[-(i + 2)])
        return self.outc(x)
