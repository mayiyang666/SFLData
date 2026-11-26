"""Mask R-CNN (ResNet50-FPN v2) 模型构建（torchvision 官方实现封装）。

- 骨干: ResNet50-FPN v2（ImageNet-1K v2 预训练），统一协议下与其他
  模型的 ImageNet 预训练骨干保持公平
- 输入: 1024x1024（protocol.fixed.input_size），min_size=max_size 固定，
  禁用检测库内部多尺度缩放，保证与语义分割模型输入完全一致
- 类别: num_classes=3（含背景），landslide=1 / road=2
"""
from __future__ import annotations

import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def build_mask_rcnn(num_classes: int = 3,
                    input_size: int = 1024,
                    pretrained_backbone: bool = True,
                    trainable_backbone_layers: int = 5,
                    box_detections_per_img: int = 300,
                    rpn_pre_nms_top_n_train: int = 2000,
                    rpn_post_nms_top_n_train: int = 1000,
                    rpn_pre_nms_top_n_test: int = 1000,
                    rpn_post_nms_top_n_test: int = 300) -> nn.Module:
    """构建 Mask R-CNN R50-FPN v2。

    num_classes 含背景（torchvision 约定，类别 0 为背景）。
    """
    weights_backbone = "IMAGENET1K_V2" if pretrained_backbone else None
    model = maskrcnn_resnet50_fpn_v2(
        weights=None,                       # 不用 COCO 整模型权重，仅骨干 ImageNet
        weights_backbone=weights_backbone,
        num_classes=num_classes,
        trainable_backbone_layers=trainable_backbone_layers,
        min_size=input_size,                # 固定输入尺寸，不做内部缩放
        max_size=input_size,
        box_detections_per_img=box_detections_per_img,
        # 内部 score 预过滤置 0：协议统一 conf_threshold=0.001 在
        # instance_infer 写 results 时应用，库内默认 0.05 会预筛低分预测、
        # 与 YOLO 系模型的评估口径不对齐
        box_score_thresh=0.0,
        rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
        rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
        rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
        rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
    )
    return model
