"""MFFENet（Xu et al., Landslides 2022, 19:1617-1647）+ ResNet101 骨干实现。

论文: "MFFENet and ADANet: a robust deep transfer learning method and its
application in high precision and fast cross-scene recognition of
earthquake-induced landslides"。

本实现忠实移植官方仓库 LandslideNet-master/MFFENet 的 denseGates_decoder
（ATF 自适应三角分叉模块 + DFPM 稠密自顶向下金字塔），并按 benchmark 约定
做如下适配：
- 骨干: 与官方一致为 ResNet101（此处用 torchvision 实现，
  replace_stride_with_dilation=[False, True, True] 达成 OS16，
  ImageNet V2 预训练），可经 backbone 参数切回 resnet50；解码器所需
  64/256/512 三级低层特征通道数在 ResNet50/101 上完全相同，无需改动
- SyncBatchNorm 替换为普通 BatchNorm2d（benchmark 用 DDP 单卡 BN 口径）
- 去除官方代码中的 .cuda() 硬编码（index_select -> 张量切片），兼容 CPU 前向
- ASPP 采样率按官方代码 OS16 设定 rates={6,12,18}+1x1+全局池化（论文正文
  写 {6,12,24,36}，以官方代码为准，可经 arch.aspp_rates 覆盖）

编码器输出（与官方一致）:
    feat1 = conv1+bn+relu（maxpool 前）   /2,   64ch
    feat2 = layer1                        /4,  256ch
    feat3 = layer2                        /8,  512ch
    x     = layer3+layer4 -> ASPP         /16, 256ch

解码器三级 ATF（自深向浅）:
    same-scales:  Attention_block 以相邻更深层特征为门控 g，对同尺度编码
                  特征做 soft-attention 加权（论文公式 (1)-(2)）
    cross-scales: CA 模块将 ASPP 与各级编码特征 1x1 投影到同一通道并
                  双线性对齐到本层；Self_attention 生成 2 通道（/1 通道）
                  注意力分别加权两分支（论文 AFM，公式 (6)-(7)）
    输出拼接后经 1x1-3x3-1x1 残差细化（论文 RCM）
DFPM: 将 ASPP 与各级 ATF 输出逐级上采样稠密拼接，经 3x3 conv 汇聚
    （four_conv/third_conv/second_conv，论文公式 (9)-(10)），最终在 /2
    分辨率四路拼接（1024ch）经 last_conv 输出 logits，插值回输入尺寸。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet50_Weights, ResNet101_Weights

# 支持的骨干: torchvision 构建函数 + ImageNet V2 权重（均 OS16 空洞化）
_BACKBONES = {
    "resnet50": (torchvision.models.resnet50, ResNet50_Weights.IMAGENET1K_V2),
    "resnet101": (torchvision.models.resnet101, ResNet101_Weights.IMAGENET1K_V2),
}


# =====================================================================
# ASPP（官方 aspp.py 移植，OS16 默认 rates={6,12,18}）
# =====================================================================
class _ASPPModule(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 padding: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=1,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ASPP(nn.Module):
    """官方 ASPP：1x1 + 3 个空洞分支 + 全局池化，拼接后 1x1 投影。"""

    def __init__(self, in_ch: int = 2048, out_ch: int = 256,
                 rates=(6, 12, 18), dropout: float = 0.5):
        super().__init__()
        self.aspp1 = _ASPPModule(in_ch, out_ch, 1, padding=0, dilation=1)
        self.aspp2 = _ASPPModule(in_ch, out_ch, 3, padding=rates[0], dilation=rates[0])
        self.aspp3 = _ASPPModule(in_ch, out_ch, 3, padding=rates[1], dilation=rates[1])
        self.aspp4 = _ASPPModule(in_ch, out_ch, 3, padding=rates[2], dilation=rates[2])
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_ch, out_ch, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU())
        self.conv1 = nn.Conv2d(out_ch * 5, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat((self.aspp1(x), self.aspp2(x), self.aspp3(x),
                       self.aspp4(x), x5), dim=1)
        return self.dropout(self.relu(self.bn1(self.conv1(x))))


# =====================================================================
# ATF 基础组件（官方 denseGates_decoder.py 移植）
# =====================================================================
class AttentionBlock(nn.Module):
    """AGM：attention U-Net 风格软注意力门（论文公式 (1)-(2)）。

    g 为门控（相邻更深层特征，已对齐尺寸），x 为同尺度编码特征；
    输出为 F_int 通道的加权特征 x * sigmoid(psi(relu(Wg(g)+Wx(x))))。
    """

    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(f_int))
        self.W_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g = self.W_g(g)
        x = self.W_x(x)
        psi = self.relu(g + x)
        psi = self.psi(psi)
        return x * psi


class CA(nn.Module):
    """跨尺度通道对齐：三路 1x1 conv+BN+ReLU 投影到统一通道数。"""

    def __init__(self, in_ch1: int, in_ch2: int, in_ch3: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch1, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_ch2, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_ch3, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU())

    def forward(self, x1, x2, x3):
        return self.conv1(x1), self.conv2(x2), self.conv3(x3)


class SelfAttention(nn.Module):
    """AFM：cat(g,x) -> 3x3 -> 1x1 -> sigmoid 生成 2 通道注意力，
    分别加权 g 与 x（论文公式 (6)-(7)）。"""

    def __init__(self, f: int):
        super().__init__()
        self.C1 = nn.Sequential(
            nn.Conv2d(f, f // 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(f // 2))
        self.C2 = nn.Sequential(
            nn.Conv2d(f // 2, 2, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2),
            nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        x1 = torch.cat((g, x), dim=1)
        x1 = self.relu(self.C1(x1))
        a = self.C2(x1)
        return a[:, :1] * g, a[:, 1:2] * x


class SelfAttentionTop(nn.Module):
    """最浅层用单通道注意力版：对拼接特征整体加权。"""

    def __init__(self, f: int):
        super().__init__()
        self.C1 = nn.Sequential(
            nn.Conv2d(f, f // 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(f // 2))
        self.C2 = nn.Sequential(
            nn.Conv2d(f // 2, 1, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g):
        g1 = self.relu(self.C1(g))
        return self.C2(g1) * g


class Decoder(nn.Module):
    """官方 denseGates_decoder.Decoder 移植（resnet 通道设定）。

    通道对应（ResNet50/101 相同）:
        x=ASPP 256@/16, feat1=64@/2, feat2=256@/4, feat3=512@/8
    """

    def __init__(self, num_classes: int):
        super().__init__()
        # ---- ATF 第 3 级（/8，对齐 feat3）----
        self.Att3 = AttentionBlock(f_g=256, f_l=512, f_int=192)
        self.Con3 = SelfAttention(f=576)
        self.CA3 = CA(256, 256, 64, 192)
        # ---- ATF 第 2 级（/4，对齐 feat2）----
        self.Att2 = AttentionBlock(f_g=512, f_l=256, f_int=96)
        self.Con2 = SelfAttention(f=288)
        self.CA2 = CA(256, 512, 64, 96)
        # ---- ATF 第 1 级（/2，对齐 feat1）----
        self.Att1 = AttentionBlock(f_g=256, f_l=64, f_int=48)
        self.Con1 = SelfAttentionTop(f=144)
        self.CA1 = CA(256, 512, 256, 48)

        # ---- RCM：1x1-3x3-1x1 残差细化 ----
        self.conv_3 = nn.Sequential(
            nn.Conv2d(768, 256, 1, bias=False),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 768, 1, bias=False))
        self.conv_2 = nn.Sequential(
            nn.Conv2d(384, 128, 1, bias=False),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 384, 1, bias=False))
        self.conv_1 = nn.Sequential(
            nn.Conv2d(192, 64, 1, bias=False),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 192, 1, bias=False))

        # ---- DFPM：稠密自顶向下汇聚 ----
        self.four_conv = nn.Sequential(nn.Conv2d(1024, 256, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(256), nn.ReLU())
        self.third_conv = nn.Sequential(nn.Conv2d(896, 256, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(256), nn.ReLU())
        self.second_conv = nn.Sequential(nn.Conv2d(960, 256, 3, padding=1, bias=False),
                                         nn.BatchNorm2d(256), nn.ReLU())
        self.last_conv = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1))
        self._init_weight()

    def forward(self, x, low_level_feat1, low_level_feat2, low_level_feat3):
        # ---- ATF 第 3 级（/8）----
        x1 = F.interpolate(x, size=low_level_feat3.size()[2:],
                           mode='bilinear', align_corners=True)
        c11 = self.Att3(g=x1, x=low_level_feat3)
        x11, x13, x14 = self.CA3(x1=x, x2=low_level_feat2, x3=low_level_feat1)
        s3 = low_level_feat3.size()[2:]
        x11 = F.interpolate(x11, size=s3, mode='bilinear', align_corners=True)
        x13 = F.interpolate(x13, size=s3, mode='bilinear', align_corners=True)
        x14 = F.interpolate(x14, size=s3, mode='bilinear', align_corners=True)
        c12, c13 = self.Con3(g=x11, x=torch.cat((x13, x14), dim=1))
        x1 = torch.cat((c11, c12, c13), dim=1)
        x1 = x1 + self.conv_3(x1)                          # 768ch @ /8

        # ---- ATF 第 2 级（/4）----
        x2 = F.interpolate(low_level_feat3, size=low_level_feat2.size()[2:],
                           mode='bilinear', align_corners=True)
        c21 = self.Att2(g=x2, x=low_level_feat2)
        x21, x22, x24 = self.CA2(x1=x, x2=low_level_feat3, x3=low_level_feat1)
        s2 = low_level_feat2.size()[2:]
        x21 = F.interpolate(x21, size=s2, mode='bilinear', align_corners=True)
        x22 = F.interpolate(x22, size=s2, mode='bilinear', align_corners=True)
        x24 = F.interpolate(x24, size=s2, mode='bilinear', align_corners=True)
        c22, c23 = self.Con2(g=torch.cat((x21, x22), dim=1), x=x24)
        x2 = torch.cat((c21, c22, c23), dim=1)
        x2 = x2 + self.conv_2(x2)                          # 384ch @ /4

        # ---- ATF 第 1 级（/2）----
        x3 = F.interpolate(low_level_feat2, size=low_level_feat1.size()[2:],
                           mode='bilinear', align_corners=True)
        c31 = self.Att1(g=x3, x=low_level_feat1)
        x31, x32, x33 = self.CA1(x1=x, x2=low_level_feat3, x3=low_level_feat2)
        s1 = low_level_feat1.size()[2:]
        x31 = F.interpolate(x31, size=s1, mode='bilinear', align_corners=True)
        x32 = F.interpolate(x32, size=s1, mode='bilinear', align_corners=True)
        x33 = F.interpolate(x33, size=s1, mode='bilinear', align_corners=True)
        c32 = self.Con1(g=torch.cat((x31, x32, x33), dim=1))
        x3 = torch.cat((c31, c32), dim=1)
        x3 = x3 + self.conv_1(x3)                          # 192ch @ /2

        # ---- DFPM：稠密自顶向下（ASPP + 各级 ATF 输出逐级汇入）----
        x11 = F.interpolate(x, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x1 = self.four_conv(torch.cat((x11, x1), dim=1))                    # 256 @ /8

        x21 = F.interpolate(x, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x22 = F.interpolate(x1, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x2 = self.third_conv(torch.cat((x2, x21, x22), dim=1))              # 256 @ /4

        x31 = F.interpolate(x, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x32 = F.interpolate(x1, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x33 = F.interpolate(x2, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x3 = self.second_conv(torch.cat((x3, x33, x32, x31), dim=1))        # 256 @ /2

        x41 = F.interpolate(x, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x42 = F.interpolate(x1, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x43 = F.interpolate(x2, size=x3.size()[2:], mode='bilinear', align_corners=True)
        return self.last_conv(torch.cat((x3, x43, x42, x41), dim=1))        # /2

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class MFFENet(nn.Module):
    """MFFENet（torchvision ResNet 骨干，output_stride=16）。

    Args:
        num_classes: 输出类别数（含背景，本数据集为 3）
        backbone: 骨干名称（resnet101 与官方/论文一致；resnet50 可消融）
        pretrained: 骨干是否加载 ImageNet 预训练权重（V2，首次运行需联网）
        aspp_rates: ASPP 空洞采样率（官方 OS16 代码设定 {6,12,18}）
        aspp_dropout: ASPP 投影后 dropout（官方 0.5）
    """

    def __init__(self, num_classes: int = 3, backbone: str = "resnet101",
                 pretrained: bool = True,
                 aspp_rates=(6, 12, 18), aspp_dropout: float = 0.5):
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"不支持的骨干: {backbone}, 可选 {list(_BACKBONES)}")
        builder, weights_cls = _BACKBONES[backbone]
        weights = weights_cls if pretrained else None
        bb = builder(weights=weights,
                     replace_stride_with_dilation=[False, True, True])
        # 骨干拆分：conv1 输出（maxpool 前）作为 /2 低层特征（官方 feat1）
        self.conv1 = bb.conv1
        self.bn1 = bb.bn1
        self.relu = bb.relu
        self.maxpool = bb.maxpool
        self.layer1 = bb.layer1   # /4,  256ch（官方 feat2）
        self.layer2 = bb.layer2   # /8,  512ch（官方 feat3）
        self.layer3 = bb.layer3   # /16, 1024ch（dilated）
        self.layer4 = bb.layer4   # /16, 2048ch（dilated）

        self.aspp = ASPP(2048, 256, rates=aspp_rates, dropout=aspp_dropout)
        self.decoder = Decoder(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        f1 = self.relu(self.bn1(self.conv1(x)))   # /2, 64ch
        f2 = self.layer1(self.maxpool(f1))        # /4, 256ch
        f3 = self.layer2(f2)                      # /8, 512ch
        e = self.layer4(self.layer3(f3))          # /16, 2048ch
        e = self.aspp(e)                          # /16, 256ch
        out = self.decoder(e, f1, f2, f3)         # /2, num_classes
        return F.interpolate(out, size=size, mode='bilinear', align_corners=True)


# 向后兼容别名（历史配置/脚本引用）
class MFFENetResNet50(MFFENet):
    def __init__(self, num_classes: int = 3, pretrained: bool = True,
                 aspp_rates=(6, 12, 18), aspp_dropout: float = 0.5):
        super().__init__(num_classes=num_classes, backbone="resnet50",
                         pretrained=pretrained, aspp_rates=aspp_rates,
                         aspp_dropout=aspp_dropout)
