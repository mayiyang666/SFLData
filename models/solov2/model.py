#!/usr/bin/env python
"""SOLOv2 (ResNet50-FPN) 纯 PyTorch 复现 —— 接入统一 benchmark。

参考实现: mmdetection `SOLOV2Head` / `SOLOV2MaskFeatureHead` / `mask_matrix_nms`
（逐函数对齐，https://github.com/open-mmlab/mmdetection），超参取官方
`solov2_r50_fpn_1x_coco.py`。不依赖 mmcv/mmengine。

结构:
    ResNet50(ImageNet-1K-V2) + FPN(5 级, stride 4..64, P6=maxpool(P5))
    -> MaskFeatModule: 统一 mask 特征 (stride 4, 256ch, GN32, 末级输入拼坐标图)
    -> SOLOV2Head: kernel/cate 双分支(4 层 GN 卷积), 坐标卷积, interp 到 num_grids
    -> 训练: center-of-mass 网格指派(pos_scale=0.2) + Focal(cls w1.0, γ2 α0.25)
             + Dice(mask w3.0, sigmoid, eps=1e-5)
       推理: 3x3 局部极大过滤 -> 动态 1x1 卷积 -> maskness 重加权
             -> Matrix NMS(gaussian σ2.0, filter_thr 0.05, max 100)

与协议的对齐说明:
    - 输出统一为 torchvision 检测格式 {boxes, labels, scores, masks[N,1,H,W]}，
      与 benchmark.eval.instance_infer.infer_to_coco 直接对接；
    - 类别通道数 = num_classes（无背景通道，SOLOHead 惯例，BG 标号=num_classes
      经 one-hot 展开为全零目标）；
    - score_thr=0.1 为 SOLOv2 官方推理阈值（官方 test_cfg），早于动态卷积执行，
      属模型固有设计，协议 conf_threshold=0.001 在其之外不追加截断。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# ---------------------------------------------------------------------------
# 基础组件
# ---------------------------------------------------------------------------
def generate_coordinate(featmap_sizes, device) -> torch.Tensor:
    """[B,2,H,W] 坐标图（x/y ∈ [-1,1]，x 在前），mmdet generate_coordinate 对齐。"""
    h, w = int(featmap_sizes[-2]), int(featmap_sizes[-1])
    x_range = torch.linspace(-1, 1, w, device=device)
    y_range = torch.linspace(-1, 1, h, device=device)
    y, x = torch.meshgrid(y_range, x_range, indexing="ij")
    b = int(featmap_sizes[0])
    return torch.cat([x.expand(b, 1, h, w), y.expand(b, 1, h, w)], 1)


class ConvGNReLU(nn.Module):
    """ConvModule(norm=GN32, act=ReLU) 等价实现（norm 存在时 conv 无 bias）。"""

    def __init__(self, cin: int, cout: int, k: int, gn_groups: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, padding=k // 2, bias=False)
        self.gn = nn.GroupNorm(gn_groups, cout)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class ResNetFPN(nn.Module):
    """ResNet50 + mmdet 风格 FPN（5 级 stride 4/8/16/32/64，第 5 级 = maxpool(P5)）。"""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = (torchvision.models.ResNet50_Weights.IMAGENET1K_V2
                   if pretrained else None)
        r50 = torchvision.models.resnet50(weights=weights)
        self.stem = nn.Sequential(r50.conv1, r50.bn1, r50.relu, r50.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            r50.layer1, r50.layer2, r50.layer3, r50.layer4)
        chans = (256, 512, 1024, 2048)
        self.laterals = nn.ModuleList(nn.Conv2d(c, 256, 1) for c in chans)
        self.fpn_convs = nn.ModuleList(
            nn.Conv2d(256, 256, 3, padding=1) for _ in chans)
        for m in list(self.laterals) + list(self.fpn_convs):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        c = []
        x = self.stem(x)
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = layer(x)
            c.append(x)
        lat = [l(t) for l, t in zip(self.laterals, c)]
        for i in range(len(lat) - 2, -1, -1):
            lat[i] = lat[i] + F.interpolate(lat[i + 1], size=lat[i].shape[-2:],
                                            mode="nearest")
        outs = [fc(t) for fc, t in zip(self.fpn_convs, lat)]
        outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return outs


class MaskFeatModule(nn.Module):
    """SOLOv2 统一 mask 特征分支（mmdet SOLOV2MaskFeatureHead 对齐）。

    各层独立 conv + 2x 上采样链路后逐元素相加（输出 stride=mask_stride），
    最后 1x1 conv 输出统一特征；末级输入额外拼接 2 通道坐标图。
    """

    def __init__(self, in_channels: int, feat_channels: int, end_level: int,
                 out_channels: int, mask_stride: int = 4):
        super().__init__()
        self.start_level = 0
        self.end_level = end_level
        self.mask_stride = mask_stride
        self.convs_all_levels = nn.ModuleList()
        for i in range(self.start_level, end_level + 1):
            convs = nn.Sequential()
            if i == 0:
                convs.add_module("conv0", ConvGNReLU(in_channels, feat_channels, 3))
                self.convs_all_levels.append(convs)
                continue
            for j in range(i):
                if j == 0:
                    chn = in_channels + 2 if i == end_level else in_channels
                    convs.add_module(f"conv{j}", ConvGNReLU(chn, feat_channels, 3))
                else:
                    convs.add_module(f"conv{j}", ConvGNReLU(feat_channels, feat_channels, 3))
                convs.add_module(f"up{j}", nn.Upsample(scale_factor=2, mode="bilinear",
                                                       align_corners=False))
            self.convs_all_levels.append(convs)
        self.conv_pred = ConvGNReLU(feat_channels, out_channels, 1)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        inputs = x[self.start_level:self.end_level + 1]
        feat = self.convs_all_levels[0](inputs[0])
        for i in range(1, len(inputs)):
            input_p = inputs[i]
            if i == len(inputs) - 1:
                coord = generate_coordinate(input_p.size(), input_p.device)
                input_p = torch.cat([input_p, coord], 1)
            feat = feat + self.convs_all_levels[i](input_p)
        return self.conv_pred(feat)


# ---------------------------------------------------------------------------
# 损失（mmdet FocalLoss / DiceLoss 对齐）
# ---------------------------------------------------------------------------
def focal_loss(pred: torch.Tensor, target: torch.Tensor, num_classes: int,
               gamma: float = 2.0, alpha: float = 0.25,
               avg_factor: int | None = None) -> torch.Tensor:
    """mmdet FocalLoss(use_sigmoid=True) 对齐：BG 类（标号 num_classes）
    one-hot 展开后丢弃末通道 -> 全零目标（等价对每通道作负样本）。"""
    onehot = F.one_hot(target, num_classes + 1)[:, :-1].float()
    bce = F.binary_cross_entropy_with_logits(pred, onehot, reduction="none")
    p_t = torch.exp(-bce)
    a_t = torch.where(onehot > 0.5, alpha, 1.0 - alpha)
    loss = a_t * (1 - p_t) ** gamma * bce
    return loss.sum() / max(avg_factor, 1)


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-5,
              reduction: str = "mean") -> torch.Tensor:
    """mmdet DiceLoss(use_sigmoid=True, activate=True, eps=1e-5) 对齐。"""
    pred = pred.sigmoid().flatten(1)
    target = target.flatten(1)
    num = 2 * (pred * target).sum(-1)
    den = pred.sum(-1) + target.sum(-1)
    loss = 1 - (num + eps) / (den + eps)
    return loss if reduction == "none" else loss.mean()


def center_of_mass(mask: torch.Tensor):
    """单实例 2D 掩码质心 (cy, cx)，mmdet models.utils.center_of_mass 对齐。"""
    h, w = mask.shape
    ys = torch.arange(h, device=mask.device, dtype=torch.float32)
    xs = torch.arange(w, device=mask.device, dtype=torch.float32)
    area = mask.float().sum().clamp(min=1.0)
    cy = (mask.float() * ys[:, None]).sum() / area
    cx = (mask.float() * xs[None, :]).sum() / area
    return cy, cx


def mask_matrix_nms(masks, labels, scores, filter_thr=-1, nms_pre=-1, max_num=-1,
                    kernel="gaussian", sigma=2.0, mask_area=None):
    """Matrix NMS（mmdet mask_matrix_nms 逐行对齐，纯 PyTorch）。

    返回 (scores, labels, masks, keep_inds)；masks 输入为二值。
    """
    assert len(labels) == len(masks) == len(scores)
    if len(labels) == 0:
        return (scores.new_zeros(0), labels.new_zeros(0),
                masks.new_zeros(0, *masks.shape[-2:]), labels.new_zeros(0))
    if mask_area is None:
        mask_area = masks.sum((1, 2)).float()

    scores, sort_inds = torch.sort(scores, descending=True)
    keep_inds = sort_inds
    if nms_pre > 0 and len(sort_inds) > nms_pre:
        sort_inds = sort_inds[:nms_pre]
        keep_inds = keep_inds[:nms_pre]
        scores = scores[:nms_pre]
    masks = masks[sort_inds]
    mask_area = mask_area[sort_inds]
    labels = labels[sort_inds]

    num_masks = len(labels)
    flatten_masks = masks.reshape(num_masks, -1).float()
    inter_matrix = torch.mm(flatten_masks, flatten_masks.transpose(1, 0))
    expanded_mask_area = mask_area.expand(num_masks, num_masks)
    iou_matrix = (inter_matrix /
                  (expanded_mask_area + expanded_mask_area.transpose(1, 0) -
                   inter_matrix)).triu(diagonal=1)
    expanded_labels = labels.expand(num_masks, num_masks)
    label_matrix = (expanded_labels == expanded_labels.transpose(1, 0)).triu(diagonal=1)

    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(num_masks, num_masks).transpose(1, 0)
    decay_iou = iou_matrix * label_matrix

    if kernel == "gaussian":
        decay_matrix = torch.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == "linear":
        decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError(f"{kernel} kernel is not supported in matrix nms!")
    scores = scores * decay_coefficient

    if filter_thr > 0:
        keep = scores >= filter_thr
        keep_inds = keep_inds[keep]
        if not keep.any():
            return (scores.new_zeros(0), labels.new_zeros(0),
                    masks.new_zeros(0, *masks.shape[-2:]), labels.new_zeros(0))
        masks = masks[keep]
        scores = scores[keep]
        labels = labels[keep]

    scores, sort_inds = torch.sort(scores, descending=True)
    keep_inds = keep_inds[sort_inds]
    if max_num > 0 and len(sort_inds) > max_num:
        sort_inds = sort_inds[:max_num]
        keep_inds = keep_inds[:max_num]
        scores = scores[:max_num]
    masks = masks[sort_inds]
    labels = labels[sort_inds]
    return scores, labels, masks, keep_inds


# ---------------------------------------------------------------------------
# SOLOv2 Head
# ---------------------------------------------------------------------------
class SOLOV2Head(nn.Module):
    """kernel/cate 双分支 + 动态 1x1 卷积（mmdet SOLOV2Head 逐函数对齐）。"""

    def __init__(self, num_classes: int, in_channels: int = 256,
                 feat_channels: int = 512, stacked_convs: int = 4,
                 strides=(8, 8, 16, 32, 32),
                 scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
                 pos_scale: float = 0.2, num_grids=(40, 36, 24, 16, 12),
                 mask_feature_head: dict | None = None,
                 loss_cls_weight: float = 1.0, loss_mask_weight: float = 3.0,
                 test_cfg: dict | None = None):
        super().__init__()
        self.num_classes = num_classes
        self.cls_out_channels = num_classes  # SOLOHead 惯例：无背景通道
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = tuple(strides)
        self.num_grids = list(num_grids)
        self.num_levels = len(self.strides)
        self.scale_ranges = tuple(tuple(r) for r in scale_ranges)
        self.pos_scale = pos_scale
        self.dynamic_conv_size = 1

        mfc = dict(in_channels=in_channels, feat_channels=128, end_level=3,
                   out_channels=256, mask_stride=4)
        mfc.update(mask_feature_head or {})
        self.mask_feature_head = MaskFeatModule(
            mfc["in_channels"], mfc["feat_channels"], mfc["end_level"],
            mfc["out_channels"], mfc["mask_stride"])
        self.mask_stride = mfc["mask_stride"]
        self.kernel_out_channels = mfc["out_channels"] * self.dynamic_conv_size ** 2

        self.kernel_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        for i in range(stacked_convs):
            chn = in_channels + 2 if i == 0 else feat_channels
            self.kernel_convs.append(ConvGNReLU(chn, feat_channels, 3))
            chn = in_channels if i == 0 else feat_channels
            self.cls_convs.append(ConvGNReLU(chn, feat_channels, 3))
        self.conv_kernel = nn.Conv2d(feat_channels, self.kernel_out_channels, 3, padding=1)
        self.conv_cls = nn.Conv2d(feat_channels, self.cls_out_channels, 3, padding=1)

        self.loss_cls_weight = loss_cls_weight
        self.loss_mask_weight = loss_mask_weight
        self.test_cfg = dict(score_thr=0.1, mask_thr=0.5, nms_pre=500,
                             filter_thr=0.05, kernel="gaussian", sigma=2.0,
                             max_per_img=100)
        self.test_cfg.update(test_cfg or {})

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=0.01)

    # ---------------- 前向 ----------------
    @staticmethod
    def resize_feats(x: list[torch.Tensor]) -> list[torch.Tensor]:
        """首级下采样 0.5、末级上采样至前一级尺寸（mmdet SOLOHead.resize_feats），
        等效 strides=[8,8,16,32,32]。"""
        out = []
        for i in range(len(x)):
            if i == 0:
                out.append(F.interpolate(x[0], scale_factor=0.5, mode="bilinear"))
            elif i == len(x) - 1:
                out.append(F.interpolate(x[i], size=x[i - 1].shape[-2:], mode="bilinear"))
            else:
                out.append(x[i])
        return out

    def forward(self, feats: list[torch.Tensor]):
        mask_feats = self.mask_feature_head(feats)
        ins_kernel_feats = self.resize_feats(feats)
        kernel_preds, cls_preds = [], []
        for i in range(self.num_levels):
            x = ins_kernel_feats[i]
            coord = generate_coordinate(x.size(), x.device)
            x = torch.cat([x, coord], 1)
            kf = F.interpolate(x, size=self.num_grids[i], mode="bilinear",
                               align_corners=False)
            cf = kf[:, :-2]  # cate 分支不含坐标通道（mmdet 对齐）
            for conv in self.kernel_convs:
                kf = conv(kf)
            kernel_preds.append(self.conv_kernel(kf))
            for conv in self.cls_convs:
                cf = conv(cf)
            cls_preds.append(self.conv_cls(cf))
        return kernel_preds, cls_preds, mask_feats

    # ---------------- 训练目标 ----------------
    def _get_targets_single(self, gt: dict, featmap_sizes: tuple):
        """center-of-mass 网格指派（mmdet _get_targets_single 对齐）。"""
        gt_labels = gt["labels"]
        device = gt_labels.device
        gt_bboxes = gt["boxes"]
        gt_areas = torch.sqrt((gt_bboxes[:, 2] - gt_bboxes[:, 0]) *
                              (gt_bboxes[:, 3] - gt_bboxes[:, 1]))
        gt_masks = gt["masks"].to(device=device, dtype=torch.bool)  # [N,H,W]

        mlvl_pos_mask_targets, mlvl_labels = [], []
        mlvl_pos_masks, mlvl_pos_indexes = [], []
        up_h = featmap_sizes[0] * self.mask_stride
        up_w = featmap_sizes[1] * self.mask_stride
        for (lower, upper), num_grid in zip(self.scale_ranges, self.num_grids):
            mask_target, pos_index = [], []
            labels = torch.full((num_grid, num_grid), self.num_classes,
                                dtype=torch.int64, device=device)
            pos_mask = torch.zeros(num_grid ** 2, dtype=torch.bool, device=device)
            gt_inds = ((gt_areas >= lower) &
                       (gt_areas <= upper)).nonzero().flatten()
            if gt_inds.numel() == 0:
                mlvl_pos_mask_targets.append(
                    torch.zeros(0, *featmap_sizes, device=device))
                mlvl_labels.append(labels)
                mlvl_pos_masks.append(pos_mask)
                mlvl_pos_indexes.append([])
                continue
            for idx in gt_inds.tolist():
                gt_mask = gt_masks[idx]
                if float(gt_mask.sum()) == 0:
                    continue
                label = int(gt_labels[idx])
                pos_h_range = 0.5 * float(gt_bboxes[idx, 3] - gt_bboxes[idx, 1]) * self.pos_scale
                pos_w_range = 0.5 * float(gt_bboxes[idx, 2] - gt_bboxes[idx, 0]) * self.pos_scale
                center_h, center_w = center_of_mass(gt_mask)
                ch, cw = float(center_h), float(center_w)
                coord_h = int(ch / up_h * num_grid)
                coord_w = int(cw / up_w * num_grid)
                top_box = max(0, int((ch - pos_h_range) / up_h * num_grid))
                down_box = min(num_grid - 1, int((ch + pos_h_range) / up_h * num_grid))
                left_box = max(0, int((cw - pos_w_range) / up_w * num_grid))
                right_box = min(num_grid - 1, int((cw + pos_w_range) / up_w * num_grid))
                top = max(top_box, coord_h - 1)
                down = min(down_box, coord_h + 1)
                left = max(coord_w - 1, left_box)
                right = min(right_box, coord_w + 1)

                labels[top:down + 1, left:right + 1] = label
                # mask target：全分辨率掩码按 1/mask_stride 双线性缩放后截断取整
                # （对齐 mmdet: cv2.imrescale + np.uint8 截断）
                m = F.interpolate(gt_mask[None, None].float(),
                                  scale_factor=1.0 / self.mask_stride,
                                  mode="bilinear", align_corners=False)[0, 0]
                m = torch.floor(m).clamp_(0, 1)
                for i in range(top, down + 1):
                    for j in range(left, right + 1):
                        t = torch.zeros(*featmap_sizes, device=device)
                        t[:m.shape[0], :m.shape[1]] = m
                        mask_target.append(t)
                        pos_mask[i * num_grid + j] = True
                        pos_index.append(i * num_grid + j)
            mlvl_pos_mask_targets.append(
                torch.stack(mask_target) if mask_target else
                torch.zeros(0, *featmap_sizes, device=device))
            mlvl_labels.append(labels)
            mlvl_pos_masks.append(pos_mask)
            mlvl_pos_indexes.append(pos_index)
        return mlvl_pos_mask_targets, mlvl_labels, mlvl_pos_masks, mlvl_pos_indexes

    # ---------------- 训练 loss ----------------
    def loss(self, feats: list[torch.Tensor], targets: list[dict]) -> dict:
        kernel_preds, cls_preds, mask_feats = self.forward(feats)
        featmap_sizes = mask_feats.shape[-2:]
        per_img = [self._get_targets_single(t, featmap_sizes) for t in targets]
        mlvl_pos_mask_targets = list(zip(*[m[0] for m in per_img]))
        mlvl_labels = list(zip(*[m[1] for m in per_img]))
        mlvl_pos_masks = list(zip(*[m[2] for m in per_img]))
        mlvl_pos_indexes = list(zip(*[m[3] for m in per_img]))
        num_pos = sum(int(pm.sum()) for lvl_pm in mlvl_pos_masks
                      for pm in lvl_pm)

        # 正样本格的动态卷积 mask 预测
        mlvl_mask_preds = []
        for lvl, lvl_indexes in enumerate(mlvl_pos_indexes):
            lvl_preds = []
            for img_id, idx in enumerate(lvl_indexes):
                if len(idx) == 0:
                    continue
                kp = kernel_preds[lvl][img_id]           # [Ck, g, g]
                c = kp.shape[0]                          # Ck = kernel_out_channels
                kp = kp.reshape(c, -1)[:, idx]           # [Ck, npos]
                n = kp.shape[1]
                pred = F.conv2d(mask_feats[img_id][None],
                                kp.permute(1, 0).reshape(n, c, 1, 1), stride=1)
                lvl_preds.append(pred.reshape(n, *featmap_sizes))
            mlvl_mask_preds.append(torch.cat(lvl_preds) if lvl_preds else None)

        loss_mask = mask_feats.sum() * 0
        if num_pos > 0:
            per = []
            for pred, lvl_targets in zip(mlvl_mask_preds, mlvl_pos_mask_targets):
                if pred is None:
                    continue
                tgt = torch.cat(lvl_targets).float()
                per.append(dice_loss(pred.float(), tgt, reduction="none"))
            loss_mask = torch.cat(per).sum() / num_pos
        loss_mask = loss_mask * self.loss_mask_weight

        flat_labels = torch.cat(
            [torch.cat([img.reshape(-1) for img in lvl]) for lvl in mlvl_labels])
        flat_cls = torch.cat(
            [lvl.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
             for lvl in cls_preds])
        loss_cls = focal_loss(flat_cls.float(), flat_labels,
                              self.cls_out_channels, avg_factor=num_pos + 1)
        loss_cls = loss_cls * self.loss_cls_weight

        # DDP 图保持项：无正样本步 kernel 分支也能收到零梯度
        keeper = kernel_preds[0].sum() * 0
        for k in kernel_preds[1:]:
            keeper = keeper + k.sum() * 0
        return {"loss_cls": loss_cls, "loss_mask": loss_mask + keeper}

    # ---------------- 推理 ----------------
    def predict(self, feats: list[torch.Tensor], img_hw: list[tuple]) -> list[dict]:
        kernel_preds, cls_preds, mask_feats = self.forward(feats)
        cfg = self.test_cfg
        h4, w4 = mask_feats.shape[-2:]
        up = (h4 * self.mask_stride, w4 * self.mask_stride)

        # 3x3 局部极大值过滤（mmdet: max_pool2d(2,1,1)[:, :, :-1, :-1]）
        kept_cls_preds = []
        for lvl in range(self.num_levels):
            s = cls_preds[lvl].sigmoid()
            local_max = F.max_pool2d(s, 2, stride=1, padding=1)
            kept = s * (local_max[:, :, :-1, :-1] == s)
            kept_cls_preds.append(kept.permute(0, 2, 3, 1))

        results = []
        for b in range(mask_feats.shape[0]):
            img_cls = torch.cat([kept_cls_preds[lvl][b].reshape(
                -1, self.cls_out_channels) for lvl in range(self.num_levels)])
            img_ker = torch.cat([kernel_preds[lvl][b].permute(1, 2, 0).reshape(
                -1, self.kernel_out_channels) for lvl in range(self.num_levels)])
            results.append(self._predict_single(
                img_ker, img_cls, mask_feats[b][None], img_hw[b], cfg, up))
        return results

    def _empty_result(self, h: int, w: int, device) -> dict:
        return {"boxes": torch.zeros(0, 4, device=device),
                "labels": torch.zeros(0, dtype=torch.int64, device=device),
                "scores": torch.zeros(0, device=device),
                "masks": torch.zeros(0, 1, h, w, device=device)}

    def _predict_single(self, kernel_pred: torch.Tensor, cls_scores: torch.Tensor,
                        mask_feats: torch.Tensor, img_hw: tuple,
                        cfg: dict, up: tuple) -> dict:
        h, w = img_hw
        score_mask = cls_scores > cfg["score_thr"]
        cls_scores = cls_scores[score_mask]
        if cls_scores.numel() == 0:
            return self._empty_result(h, w, cls_scores.device)
        inds = score_mask.nonzero()
        cls_labels = inds[:, 1]
        kernels = kernel_pred[inds[:, 0]]

        lvl_interval = torch.cumsum(torch.tensor(
            [n ** 2 for n in self.num_grids]), 0).to(inds.device)
        strides = torch.ones(int(lvl_interval[-1]), device=inds.device)
        strides[:lvl_interval[0]] *= self.strides[0]
        for lvl in range(1, self.num_levels):
            strides[lvl_interval[lvl - 1]:lvl_interval[lvl]] *= self.strides[lvl]
        strides = strides[inds[:, 0]]

        mask_preds = F.conv2d(
            mask_feats, kernels.reshape(kernels.shape[0], -1, 1, 1),
            stride=1).squeeze(0).sigmoid()
        masks = mask_preds > cfg["mask_thr"]
        sum_masks = masks.sum((1, 2)).float()
        keep = sum_masks > strides
        if keep.sum() == 0:
            return self._empty_result(h, w, cls_scores.device)
        masks, mask_preds = masks[keep], mask_preds[keep]
        sum_masks, cls_scores, cls_labels = (sum_masks[keep], cls_scores[keep],
                                             cls_labels[keep])

        # maskness 重加权
        mask_scores = (mask_preds * masks).sum((1, 2)) / sum_masks
        cls_scores = cls_scores * mask_scores

        scores, labels, _, keep_inds = mask_matrix_nms(
            masks, cls_labels, cls_scores, mask_area=sum_masks,
            nms_pre=cfg["nms_pre"], max_num=cfg["max_per_img"],
            kernel=cfg["kernel"], sigma=cfg["sigma"], filter_thr=cfg["filter_thr"])
        if scores.numel() == 0:
            return self._empty_result(h, w, cls_scores.device)

        mask_preds = mask_preds[keep_inds]
        mask_preds = F.interpolate(mask_preds.unsqueeze(0), size=up,
                                   mode="bilinear", align_corners=False)[:, :, :h, :w]
        mask_preds = F.interpolate(mask_preds, size=(h, w), mode="bilinear",
                                   align_corners=False).squeeze(0)

        # 全分辨率下清掉零面积掩码并产出 xyxy 框
        boxes, sel = [], []
        for i in range(mask_preds.shape[0]):
            m = mask_preds[i] > cfg["mask_thr"]
            ys, xs = m.nonzero(as_tuple=True)
            if xs.numel() == 0:
                continue
            sel.append(i)
            boxes.append([float(xs.min()), float(ys.min()),
                          float(xs.max()) + 1.0, float(ys.max()) + 1.0])
        if not sel:
            return self._empty_result(h, w, cls_scores.device)
        sel_t = torch.tensor(sel, device=cls_scores.device)
        return {"boxes": torch.tensor(boxes, device=cls_scores.device),
                "labels": labels[sel_t],
                "scores": scores[sel_t],
                "masks": mask_preds[sel_t][:, None]}


# ---------------------------------------------------------------------------
# 整网封装
# ---------------------------------------------------------------------------
class SOLOv2(nn.Module):
    """SOLOv2 整网：ImageNet 归一化 + ResNet50-FPN + SOLOV2Head。

    训练: forward(images, targets) -> {"loss_cls", "loss_mask"}
    评估: forward(images) -> [{boxes, labels, scores, masks[N,1,H,W] 概率}]
    """

    def __init__(self, num_classes: int = 3, pretrained_backbone: bool = True,
                 arch: dict | None = None):
        super().__init__()
        arch = dict(arch or {})
        self.register_buffer("pixel_mean",
                             torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std",
                             torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.backbone = ResNetFPN(pretrained=pretrained_backbone)
        self.head = SOLOV2Head(num_classes=num_classes, in_channels=256, **arch)

    def forward(self, images: list[torch.Tensor], targets: list[dict] | None = None):
        x = torch.stack(images, 0)
        x = (x - self.pixel_mean) / self.pixel_std
        feats = self.backbone(x)
        if self.training:
            assert targets is not None, "训练模式必须提供 targets"
            return self.head.loss(feats, targets)
        img_hw = [(int(im.shape[-2]), int(im.shape[-1])) for im in images]
        return self.head.predict(feats, img_hw)


def build_solov2(num_classes: int = 3, input_size: int = 1024,
                 pretrained_backbone: bool = True, arch: dict | None = None) -> nn.Module:
    """统一构建入口（train.py 与 tools/evaluate.py 共用）。input_size 由协议固定。"""
    return SOLOv2(num_classes=num_classes, pretrained_backbone=pretrained_backbone,
                  arch=arch)
