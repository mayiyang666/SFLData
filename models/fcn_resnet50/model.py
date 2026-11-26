"""FCN-8s（Long et al., 2015）+ ResNet50 骨干实现。

结构：torchvision ResNet50 骨干（可选 ImageNet 预训练）+
经典 FCN-8s 三路融合解码头（score_fr / score_pool4 / score_pool3），
最终双线性插值回输入分辨率。输入 1024 -> layer4 步幅 32 -> 32x32 分数图。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet50_Weights


class FCNResNet50(nn.Module):
    """FCN-8s（ResNet50 骨干）。

    Args:
        num_classes: 输出类别数（含背景，本数据集为 4）
        pretrained: 骨干是否加载 ImageNet 预训练权重（首次运行需联网下载并缓存）
        dropout: score_fr 中间层 dropout 率
    """

    def __init__(self, num_classes: int = 4, pretrained: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        bb = torchvision.models.resnet50(weights=weights)
        # 拆分骨干，逐层取多尺度特征
        self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1   # stride 4,  256 ch
        self.layer2 = bb.layer2   # stride 8,  512 ch
        self.layer3 = bb.layer3   # stride 16, 1024 ch
        self.layer4 = bb.layer4   # stride 32, 2048 ch

        mid_ch = 512
        self.score_fr = nn.Sequential(
            nn.Conv2d(2048, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(mid_ch, num_classes, 1),
        )
        # 侧分支 1x1 卷积：将骨干中间特征投影到类别分数
        self.score_pool4 = nn.Conv2d(1024, num_classes, 1)  # layer3, stride 16
        self.score_pool3 = nn.Conv2d(512, num_classes, 1)   # layer2, stride 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        c1 = self.layer1(self.stem(x))
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        score = self.score_fr(c4)                              # /32
        score = F.interpolate(score, scale_factor=2,
                              mode="bilinear", align_corners=True)   # /16
        score = score + self.score_pool4(c3)
        score = F.interpolate(score, scale_factor=2,
                              mode="bilinear", align_corners=True)   # /8
        score = score + self.score_pool3(c2)
        score = F.interpolate(score, size=(h, w),
                              mode="bilinear", align_corners=True)   # /1
        return score
