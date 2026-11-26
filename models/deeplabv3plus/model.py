"""DeepLabV3+（Chen et al., 2018）+ ResNet50 骨干实现。

结构：torchvision ResNet50 骨（可选 ImageNet 预训练，layer3/layer4 空洞化，
output_stride=16）+ ASPP（rates 6/12/18 + 全局池化分支）+ 轻量解码器
（low-level layer1 特征 48ch 融合），最终双线性插值回输入分辨率。
输入 1024 -> 骨干输出 64x64 -> 解码后 256x256 -> 4x 上采样回 1024。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet50_Weights


class _ASPPConv(nn.Sequential):
    """ASPP 分支：空洞卷积（rate=1 时退化为普通 3x3）+ BN + ReLU。"""

    def __init__(self, in_ch: int, out_ch: int, rate: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=rate, dilation=rate, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _ASPPPooling(nn.Module):
    """ASPP 全局池化分支：GAP -> 1x1 conv -> 双线性回特征尺寸。"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        y = self.proj(x)
        return F.interpolate(y, size=size, mode="bilinear", align_corners=True)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling（OS16 惯例 rates=6/12/18）。"""

    def __init__(self, in_ch: int, out_ch: int = 256,
                 rates=(6, 12, 18), dropout: float = 0.1):
        super().__init__()
        branches = [nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )]
        branches += [_ASPPConv(in_ch, out_ch, r) for r in rates]
        branches.append(_ASPPPooling(in_ch, out_ch))
        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * len(branches), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [b(x) for b in self.branches]
        return self.project(torch.cat(y, dim=1))


class DeepLabV3PlusResNet50(nn.Module):
    """DeepLabV3+（ResNet50 骨干，output_stride=16）。

    Args:
        num_classes: 输出类别数（含背景，本数据集为 4）
        pretrained: 骨干是否加载 ImageNet 预训练权重（首次运行需联网下载并缓存）
        dropout: ASPP 汇聚层 dropout 率
    """

    def __init__(self, num_classes: int = 4, pretrained: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        # layer3/layer4 空洞化（stride 16 输出）；与 FCN 基线同用 V2 权重口径
        bb = torchvision.models.resnet50(
            weights=weights, replace_stride_with_dilation=[False, True, True])
        self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1   # stride 4,  256 ch（low-level 特征）
        self.layer2 = bb.layer2   # stride 8,  512 ch
        self.layer3 = bb.layer3   # stride 16, 1024 ch（dilated）
        self.layer4 = bb.layer4   # stride 16, 2048 ch（dilated）

        self.aspp = ASPP(2048, 256, rates=(6, 12, 18), dropout=dropout)

        # 解码器：low-level 1x1 投影到 48ch（原论文设定），拼接后两层 conv 细化
        self.low_proj = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.decode = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        low = self.layer1(self.stem(x))     # /4
        c4 = self.layer4(self.layer3(self.layer2(low)))   # /16

        y = self.aspp(c4)
        y = F.interpolate(y, size=low.shape[-2:],
                          mode="bilinear", align_corners=True)   # /4
        y = torch.cat([y, self.low_proj(low)], dim=1)
        y = self.decode(y)
        y = self.classifier(y)
        return F.interpolate(y, size=(h, w),
                             mode="bilinear", align_corners=True)   # /1
