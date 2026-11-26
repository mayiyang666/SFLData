#!/usr/bin/env python
"""FastInst (R50 + PPM-FPN, CVPR 2023) 纯 PyTorch 复现 —— 接入统一 benchmark。

参考实现: junjiehe96/FastInst（detectron2/Mask2Former 代码风格，逐函数对齐）:
    fastinst/fastinst.py                                  (meta arch: 加权/推理)
    fastinst/modeling/criterion.py                        (SetCriterion)
    fastinst/modeling/matcher.py                          (HungarianMatcher, location cost)
    fastinst/modeling/pixel_decoder/fastinst_encoder.py   (PPM-FPN)
    fastinst/modeling/transformer_decoder/fastinst_decoder.py
    fastinst/modeling/transformer_decoder/utils.py        (QueryProposal 等)
超参取 configs/coco/instance-segmentation/fastinst_R50_ppm-fpn_x3_640.yaml
(DEC_LAYERS=3)。不依赖 detectron2/fvcore。

与 Mask2Former 的关键差异（均按官方实现保留）:
    - IA-guided query selection(QueryProposal): res4 特征 -> 3x3+1x1 proposal
      分类图 -> softmax 前景化 -> 8 邻域局部极大 -> top-K 初始 query
      （另 8 个可学 aux query）；
    - proposal 分类监督 loss_proposal(CE, w20)；匹配代价含 location cost
      (w1000, query 是否落在 GT 掩码内)；
    - 双路更新: mask 分支(pixel 为 query 的 cross-attn+FFN)与 query 分支
      (cross-attn+self-attn+FFN)每层各一次；
    - GT-mask-guided learning: 训练时以匹配 GT mask 构造注意力掩码重跑各层
      （guided 分支，预测前置于 aux_outputs，深度监督共 2*DEC_LAYERS 份）；
    - 掩码 = mask_embed · pixel_features(stride 8)，点采样 BCE+Dice；
    - LOCATION_WEIGHT 只进匹配代价，不作为独立损失（官方 meta arch 如此）。

与协议的对齐说明:
    - 输出 torchvision 检测格式 {boxes, labels, scores, masks[N,1,H,W]}，
      masks 为 sigmoid 概率（infer_to_coco 以 0.5 二值化 == 官方 >0.5）；
    - 框由二值 mask 外接矩形导出（官方默认零框，仅为对接 COCO bbox）；
    - score 分布为 softmax 前景概率 × maskness 重加权（官方一致）。
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# PointRend 点采样（detectron2 point_features 对齐）
# ---------------------------------------------------------------------------
def point_sample(input: torch.Tensor, point_coords: torch.Tensor,
                 **kwargs) -> torch.Tensor:
    """input [N,C,H,W], point_coords [N,P,2]∈[0,1] -> [N,C,P]。"""
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def get_uncertain_point_coords_with_randomness(
        logits: torch.Tensor, uncertainty_func, num_points: int,
        oversample_ratio: float, importance_sample_ratio: float) -> torch.Tensor:
    """重要性 + 均匀混合点采样（detectron2/mmseg PointRend 对齐）：
    先过采样 num_points*oversample_ratio 个候选点，按不确定度取 top
    importance 比例，其余用均匀随机点补足。"""
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    batch_size = logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(batch_size, num_sampled, 2, device=logits.device)
    point_logits = point_sample(logits, point_coords, align_corners=False)
    # 必须基于采样点的预测值算不确定度（先在粗图上算会得到错误结果，官方注释）
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(batch_size, dtype=torch.long,
                                       device=logits.device)
    idx += shift[:, None]
    point_coords = point_coords.reshape(-1, 2)[idx.reshape(-1), :].reshape(
        batch_size, num_uncertain_points, 2)
    if num_random_points > 0:
        rand_roi_coords = torch.rand(batch_size, num_random_points, 2,
                                     device=logits.device)
        point_coords = torch.cat((point_coords, rand_roi_coords), dim=1)
    return point_coords


def calculate_uncertainty(logits: torch.Tensor) -> torch.Tensor:
    """不确定度 = -|logit|（前景通道）。"""
    assert logits.shape[1] == 1
    return -(torch.abs(logits.clone()))


# ---------------------------------------------------------------------------
# 杂项（Mask2Former utils 对齐，单文件自包含）
# ---------------------------------------------------------------------------
def get_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _max_by_dim(tensor_list: list[torch.Tensor]) -> list[int]:
    return [max(s) for s in zip(*[t.shape for t in tensor_list])]


def nested_tensor_from_tensor_list(tensor_list: list[torch.Tensor]):
    """掩码批量填充（Mask2Former utils 对齐），返回 (tensor, valid_mask)。"""
    max_size = _max_by_dim(tensor_list)
    batch_shape = [len(tensor_list)] + max_size
    b, h, w = batch_shape[0], batch_shape[-2], batch_shape[-1]
    device = tensor_list[0].device
    dtype = tensor_list[0].dtype
    tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
    mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
    for img, pad_img, m in zip(tensor_list, tensor, mask):
        pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
        m[:img.shape[1], :img.shape[2]] = False
    return tensor, mask


# ---------------------------------------------------------------------------
# Decoder 基础层（transformer_decoder/utils.py 对齐）
# ---------------------------------------------------------------------------
def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class QueryProposal(nn.Module):
    """IA-guided query selection：由单级特征图选出 top-K 实例激活点。"""

    def __init__(self, num_features: int, num_queries: int, num_classes: int):
        super().__init__()
        self.topk = num_queries
        self.num_classes = num_classes
        self.conv_proposal_cls_logits = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_classes + 1, kernel_size=1, stride=1, padding=0),
        )

    @torch.no_grad()
    def compute_coordinates(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.size(2), x.size(3)
        y_loc = torch.linspace(0, 1, h, device=x.device)
        x_loc = torch.linspace(0, 1, w, device=x.device)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc, indexing="ij")
        return torch.stack([x_loc, y_loc], 0).unsqueeze(0)

    def seek_local_maximum(self, x: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        """8 邻域局部极大图（官方实现）。"""
        x_pad = F.pad(x, (1, 1, 1, 1), "constant", 0)
        maximum = (x >= x_pad[:, :, :-2, 1:-1]) & \
                  (x >= x_pad[:, :, 2:, 1:-1]) & \
                  (x >= x_pad[:, :, 1:-1, :-2]) & \
                  (x >= x_pad[:, :, 1:-1, 2:]) & \
                  (x >= x_pad[:, :, :-2, :-2]) & \
                  (x >= x_pad[:, :, :-2, 2:]) & \
                  (x >= x_pad[:, :, 2:, :-2]) & \
                  (x >= x_pad[:, :, 2:, 2:]) & \
                  (x >= epsilon)
        return maximum.to(x)

    def forward(self, x: torch.Tensor, pos_embeddings: torch.Tensor):
        proposal_cls_logits = self.conv_proposal_cls_logits(x)          # b, c+1, h, w
        proposal_cls_probs = proposal_cls_logits.softmax(dim=1)
        proposal_cls_one_hot = F.one_hot(
            proposal_cls_probs[:, :-1, :, :].max(1)[1],
            num_classes=self.num_classes + 1).permute(0, 3, 1, 2)
        proposal_cls_probs = proposal_cls_probs.mul(proposal_cls_one_hot)
        proposal_local_maximum_map = self.seek_local_maximum(proposal_cls_probs)
        proposal_cls_probs = proposal_cls_probs + proposal_local_maximum_map

        topk_indices = torch.topk(
            proposal_cls_probs[:, :-1, :, :].flatten(2).max(1)[0], self.topk, dim=1)[1]
        topk_indices = topk_indices.unsqueeze(1)                        # b, 1, q

        topk_proposals = torch.gather(
            x.flatten(2), dim=2,
            index=topk_indices.repeat(1, x.shape[1], 1))                # b, c, q
        pos_embeddings = pos_embeddings.repeat(x.shape[0], 1, 1, 1).flatten(2)
        topk_pos_embeddings = torch.gather(
            pos_embeddings, dim=2,
            index=topk_indices.repeat(1, pos_embeddings.shape[1], 1))   # b, c, q
        if self.training:
            locations = self.compute_coordinates(x).repeat(x.shape[0], 1, 1, 1)
            topk_locations = torch.gather(
                locations.flatten(2), dim=2,
                index=topk_indices.repeat(1, locations.shape[1], 1))
            topk_locations = topk_locations.transpose(-1, -2)           # b, q, 2
        else:
            topk_locations = None
        return topk_proposals, topk_pos_embeddings, topk_locations, proposal_cls_logits


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, activation="relu",
                 normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def with_pos_embed(tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, tgt_mask=None, tgt_key_padding_mask=None,
                     query_pos=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt, tgt_mask=None, tgt_key_padding_mask=None,
                    query_pos=None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None,
                query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, activation="relu",
                 normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def with_pos_embed(tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, memory_mask=None,
                     memory_key_padding_mask=None, pos=None, query_pos=None):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos), value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt, memory, memory_mask=None,
                    memory_key_padding_mask=None, pos=None, query_pos=None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos), value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def with_pos_embed(tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class MLP(nn.Module):
    """多层感知机头（DETR 风格）。"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# ---------------------------------------------------------------------------
# PPM-FPN 像素解码器（pixel_decoder/fastinst_encoder.py 对齐）
# ---------------------------------------------------------------------------
class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels: int, channels: int = 512, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, channels, size) for size in sizes])
        self.bottleneck = nn.Conv2d(in_channels + len(sizes) * channels,
                                    in_channels, 1)

    @staticmethod
    def _make_stage(features: int, out_features: int, size: int) -> nn.Module:
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, 1)
        return nn.Sequential(prior, conv)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=F.relu_(stage(feats)), size=(h, w),
                                mode="bilinear", align_corners=False)
                  for stage in self.stages] + [feats]
        return F.relu_(self.bottleneck(torch.cat(priors, 1)))


class PyramidPoolingModuleFPN(nn.Module):
    """PPM-FPN：res5 支路加 PPM 后与 res4/res3 逐级上采样融合（官方 BaseFPN）。

    forward_features 返回 (mask_features=None, multi_scale_features)，
    multi_scale = [res5 级, res4 级, res3 级]（自顶向下）。
    """

    def __init__(self, in_channels: dict, convs_dim: int = 256, norm_groups: int = 32):
        super().__init__()
        self.num_feature_levels = 3  # 官方固定 3 尺度
        self.in_features = sorted(in_channels, key=lambda k: in_channels[k][1])  # res3..res5
        feature_channels = [in_channels[k][0] for k in self.in_features]

        lateral_convs, output_convs = [], []
        for idx, chn in enumerate(feature_channels):
            lateral_conv = nn.Conv2d(chn, convs_dim, kernel_size=1, bias=False)
            output_conv = nn.Conv2d(convs_dim, convs_dim, kernel_size=3, stride=1,
                                    padding=1, bias=False)
            nn.init.xavier_uniform_(lateral_conv.weight)
            nn.init.xavier_uniform_(output_conv.weight)
            self.add_module(f"adapter_{idx + 1}", lateral_conv)
            self.add_module(f"layer_{idx + 1}", output_conv)
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # 反转为自顶向下顺序（res5 -> res3），与官方一致
        self.lateral_convs = nn.ModuleList(lateral_convs[::-1])
        self.output_convs = nn.ModuleList(output_convs[::-1])
        self.ppm = PyramidPoolingModule(convs_dim, convs_dim // 4)

    def forward_features(self, features: dict) -> tuple:
        multi_scale_features = []
        num_cur_levels = 0
        y = None
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if idx == 0:
                y = self.ppm(lateral_conv(x))
            else:
                cur_fpn = lateral_conv(x)
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:],
                                            mode="bilinear", align_corners=False)
            y = output_conv(y)
            if num_cur_levels < self.num_feature_levels:
                multi_scale_features.append(y)
                num_cur_levels += 1
        return None, multi_scale_features


# ---------------------------------------------------------------------------
# HungarianMatcher（modeling/matcher.py 对齐）
# ---------------------------------------------------------------------------
def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    return 1 - (numerator + 1) / (denominator + 1)


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    hw = inputs.shape[1]
    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs),
                                             reduction="none")
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs),
                                             reduction="none")
    loss = torch.einsum("nc,mc->nm", pos, targets) + \
        torch.einsum("nc,mc->nm", neg, (1 - targets))
    return loss / hw


class HungarianMatcher(nn.Module):
    """二分匹配（含 FastInst 特有的 location cost 与 proposal 匹配分支）。"""

    def __init__(self, cost_class: float = 1, cost_mask: float = 1,
                 cost_dice: float = 1, cost_location: float = 1e3,
                 num_points: int = 0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_location = cost_location
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0
        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        indices = []
        for b in range(bs):
            out_query_loc = outputs["query_locations"][b]            # [q, 2(x,y)]
            out_prob = outputs["pred_logits"][b].softmax(-1)         # [q, C+1]
            out_mask = outputs["pred_masks"][b].float()              # [q, H, W]
            tgt_mask = targets[b]["masks"].to(out_mask)              # [m, H, W]
            tgt_ids = targets[b]["labels"]

            if tgt_mask.shape[0] > 0:
                cost_location = point_sample(
                    tgt_mask.unsqueeze(0), out_query_loc.unsqueeze(0).float(),
                    align_corners=False).squeeze(0)                  # [m, q]
                cost_location = (cost_location > 0).to(out_mask)
                cost_location = -cost_location.transpose(0, 1)       # [q, m]
                cost_class = -out_prob[:, tgt_ids]                   # [q, m]

                point_coords = torch.rand(1, self.num_points, 2,
                                          device=out_mask.device)
                tgt_mask_pts = point_sample(tgt_mask.unsqueeze(0), point_coords,
                                            align_corners=False).squeeze(0)
                out_mask_pts = point_sample(out_mask.unsqueeze(0), point_coords,
                                            align_corners=False).squeeze(0)
                with torch.amp.autocast("cuda", enabled=False):
                    out_mask_pts = out_mask_pts.float()
                    tgt_mask_pts = tgt_mask_pts.float()
                    cost_mask = batch_sigmoid_ce_loss(out_mask_pts, tgt_mask_pts)
                    if tgt_mask_pts.shape[0] > 0:
                        cost_dice = batch_dice_loss(out_mask_pts, tgt_mask_pts)
                    else:
                        cost_dice = batch_dice_loss(out_mask_pts, tgt_mask_pts)
            else:
                cost_location = out_mask.new_zeros(num_queries, 0)
                cost_class = out_prob.new_zeros(num_queries, 0)
                cost_mask = out_mask.new_zeros(num_queries, 0)
                cost_dice = out_mask.new_zeros(num_queries, 0)

            C = (self.cost_mask * cost_mask + self.cost_class * cost_class +
                 self.cost_dice * cost_dice + self.cost_location * cost_location)
            C = C.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))
        return [(torch.as_tensor(i, dtype=torch.int64),
                 torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    @torch.no_grad()
    def memory_efficient_forward_for_proposal(self, outputs, targets):
        bs = outputs["proposal_cls_logits"].shape[0]
        proposal_size = outputs["proposal_cls_logits"].shape[-2:]
        indices = []
        for b in range(bs):
            proposal_cls_prob = outputs["proposal_cls_logits"][b].flatten(1) \
                .transpose(0, 1).softmax(-1)                          # [hw, C+1]
            tgt_mask = targets[b]["masks"].to(proposal_cls_prob)
            tgt_ids = targets[b]["labels"]
            if tgt_mask.shape[0] > 0:
                scaled_tgt_mask = F.adaptive_avg_pool2d(
                    tgt_mask.unsqueeze(0), output_size=proposal_size)
                scaled_tgt_mask = (scaled_tgt_mask.squeeze(0) > 0.).to(proposal_cls_prob)
            else:
                scaled_tgt_mask = torch.zeros([tgt_mask.shape[0], *proposal_size],
                                              device=proposal_cls_prob.device)
            cost_location = -scaled_tgt_mask.flatten(1).transpose(0, 1)
            cost_class = -proposal_cls_prob[:, tgt_ids]
            C = self.cost_class * cost_class + self.cost_location * cost_location
            C = C.reshape(proposal_size[0] * proposal_size[1], -1).cpu()
            indices.append(linear_sum_assignment(C))
        return [(torch.as_tensor(i, dtype=torch.int64),
                 torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    @torch.no_grad()
    def forward(self, outputs, targets):
        if outputs.get("proposal_cls_logits", None) is not None:
            return self.memory_efficient_forward_for_proposal(outputs, targets)
        return self.memory_efficient_forward(outputs, targets)


# ---------------------------------------------------------------------------
# SetCriterion（modeling/criterion.py 对齐）
# ---------------------------------------------------------------------------
def dice_loss(inputs: torch.Tensor, targets: torch.Tensor,
              num_masks: float) -> torch.Tensor:
    inputs = inputs.sigmoid().float()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor,
                    num_masks: float) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(inputs.float(), targets.float(),
                                              reduction="none")
    return loss.mean(1).sum() / num_masks


class SetCriterion(nn.Module):
    """集合预测损失：labels(CE) + masks(BCE+Dice 点采样) + proposal(CE)。"""

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64,
            device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes,
                                  self.empty_weight)
        return {"loss_ce": loss_ce}

    def loss_masks(self, outputs, targets, indices, num_masks):
        assert "pred_masks" in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        target_masks, _ = nested_tensor_from_tensor_list(masks)
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks, calculate_uncertainty, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            point_labels = point_sample(target_masks, point_coords,
                                        align_corners=False).squeeze(1)
        point_logits = point_sample(src_masks, point_coords,
                                    align_corners=False).squeeze(1)
        return {
            "loss_mask": sigmoid_ce_loss(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss(point_logits, point_labels, num_masks),
        }

    def loss_proposals(self, output_proposals, targets, indices):
        assert "proposal_cls_logits" in output_proposals
        proposal_size = output_proposals["proposal_cls_logits"].shape[-2:]
        proposal_cls_logits = output_proposals["proposal_cls_logits"] \
            .flatten(2).float()                                       # b, C+1, hw
        target_classes = self.num_classes * torch.ones(
            [proposal_cls_logits.shape[0], proposal_size[0] * proposal_size[1]],
            device=proposal_cls_logits.device).to(torch.int64)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)])
        idx = self._get_src_permutation_idx(indices)
        target_classes[idx] = target_classes_o
        loss_proposal = F.cross_entropy(proposal_cls_logits, target_classes,
                                        ignore_index=-1)
        return {"loss_proposal": loss_proposal}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i)
                               for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i)
                               for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {"labels": self.loss_labels, "masks": self.loss_masks}
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        # proposal loss（独立匹配 proposal 分类图）
        proposal_loss_dict = {}
        if outputs.get("proposal_cls_logits") is not None:
            output_proposals = {"proposal_cls_logits": outputs.pop("proposal_cls_logits")}
            indices = self.matcher(output_proposals, targets)
            proposal_loss_dict = self.loss_proposals(output_proposals, targets, indices)

        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        if outputs_without_aux.get("pred_matching_indices") is not None:
            indices = outputs_without_aux["pred_matching_indices"]
        else:
            indices = self.matcher(outputs_without_aux, targets)

        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float,
                                    device=next(iter(outputs.values())).device
                                    if outputs else targets[0]["labels"].device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                if aux_outputs.get("pred_matching_indices") is not None:
                    indices = aux_outputs["pred_matching_indices"]
                else:
                    indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices,
                                           num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        losses.update(proposal_loss_dict)
        return losses


# ---------------------------------------------------------------------------
# FastInstDecoder（transformer_decoder/fastinst_decoder.py 对齐）
# ---------------------------------------------------------------------------
class FastInstDecoder(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int,
                 num_queries: int, num_aux_queries: int, nheads: int,
                 dim_feedforward: int, dec_layers: int, pre_norm: bool,
                 mask_dim: int):
        super().__init__()
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.num_queries = num_queries
        self.num_aux_queries = num_aux_queries
        self.criterion = None  # meta arch 注入（GT-guided 分支需要 matcher）

        meta_pos_size = int(round(math.sqrt(self.num_queries)))
        self.meta_pos_embed = nn.Parameter(
            torch.empty(1, hidden_dim, meta_pos_size, meta_pos_size))
        nn.init.trunc_normal_(self.meta_pos_embed, std=0.02)
        if num_aux_queries > 0:
            self.empty_query_features = nn.Embedding(num_aux_queries, hidden_dim)
            self.empty_query_pos_embed = nn.Embedding(num_aux_queries, hidden_dim)

        self.query_proposal = QueryProposal(hidden_dim, num_queries, num_classes)

        self.transformer_query_cross_attention_layers = nn.ModuleList()
        self.transformer_query_self_attention_layers = nn.ModuleList()
        self.transformer_query_ffn_layers = nn.ModuleList()
        self.transformer_mask_cross_attention_layers = nn.ModuleList()
        self.transformer_mask_ffn_layers = nn.ModuleList()
        for idx in range(self.num_layers):
            self.transformer_query_cross_attention_layers.append(CrossAttentionLayer(
                d_model=hidden_dim, nhead=nheads, dropout=0.0,
                normalize_before=pre_norm))
            self.transformer_query_self_attention_layers.append(SelfAttentionLayer(
                d_model=hidden_dim, nhead=nheads, dropout=0.0,
                normalize_before=pre_norm))
            self.transformer_query_ffn_layers.append(FFNLayer(
                d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0,
                normalize_before=pre_norm))
            self.transformer_mask_cross_attention_layers.append(CrossAttentionLayer(
                d_model=hidden_dim, nhead=nheads, dropout=0.0,
                normalize_before=pre_norm))
            self.transformer_mask_ffn_layers.append(FFNLayer(
                d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0,
                normalize_before=pre_norm))

        self.decoder_query_norm_layers = nn.ModuleList()
        self.class_embed_layers = nn.ModuleList()
        self.mask_embed_layers = nn.ModuleList()
        self.mask_features_layers = nn.ModuleList()
        for idx in range(self.num_layers + 1):
            self.decoder_query_norm_layers.append(nn.LayerNorm(hidden_dim))
            self.class_embed_layers.append(
                MLP(hidden_dim, hidden_dim, num_classes + 1, 3))
            self.mask_embed_layers.append(MLP(hidden_dim, hidden_dim, mask_dim, 3))
            self.mask_features_layers.append(nn.Linear(hidden_dim, mask_dim))

    def forward(self, x: list, mask_features, targets=None):
        bs = x[0].shape[0]
        proposal_size = x[1].shape[-2:]
        pixel_feature_size = x[2].shape[-2:]

        pixel_pos_embeds = F.interpolate(self.meta_pos_embed, size=pixel_feature_size,
                                         mode="bilinear", align_corners=False)
        proposal_pos_embeds = F.interpolate(self.meta_pos_embed, size=proposal_size,
                                            mode="bilinear", align_corners=False)

        pixel_features = x[2].flatten(2).permute(2, 0, 1)             # [HW, B, C]
        pixel_pos_embeds = pixel_pos_embeds.flatten(2).permute(2, 0, 1)

        query_features, query_pos_embeds, query_locations, proposal_cls_logits = \
            self.query_proposal(x[1], proposal_pos_embeds)
        query_features = query_features.permute(2, 0, 1)              # [q, B, C]
        query_pos_embeds = query_pos_embeds.permute(2, 0, 1)
        if self.num_aux_queries > 0:
            aux_query_features = self.empty_query_features.weight.unsqueeze(1) \
                .repeat(1, bs, 1)
            aux_query_pos_embed = self.empty_query_pos_embed.weight.unsqueeze(1) \
                .repeat(1, bs, 1)
            query_features = torch.cat([query_features, aux_query_features], dim=0)
            query_pos_embeds = torch.cat([query_pos_embeds, aux_query_pos_embed], dim=0)

        outputs_class, outputs_mask, attn_mask, _, _ = self.forward_prediction_heads(
            query_features, pixel_features, pixel_feature_size, -1,
            return_attn_mask=True)
        predictions_class = [outputs_class]
        predictions_mask = [outputs_mask]
        predictions_matching_index = [None]
        query_feature_memory = [query_features]
        pixel_feature_memory = [pixel_features]

        for i in range(self.num_layers):
            query_features, pixel_features = self.forward_one_layer(
                query_features, pixel_features, query_pos_embeds,
                pixel_pos_embeds, attn_mask, i)
            if i < self.num_layers - 1:
                outputs_class, outputs_mask, attn_mask, _, _ = \
                    self.forward_prediction_heads(
                        query_features, pixel_features, pixel_feature_size, i,
                        return_attn_mask=True)
            else:
                outputs_class, outputs_mask, _, matching_indices, gt_attn_mask = \
                    self.forward_prediction_heads(
                        query_features, pixel_features, pixel_feature_size, i,
                        return_gt_attn_mask=self.training, targets=targets,
                        query_locations=query_locations)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_matching_index.append(None)
            query_feature_memory.append(query_features)
            pixel_feature_memory.append(pixel_features)

        # GT-mask-guided learning：训练时以 GT 注意力掩码重跑各层
        guided_predictions_class = []
        guided_predictions_mask = []
        guided_predictions_matching_index = []
        if self.training:
            for i in range(self.num_layers):
                query_features, pixel_features = self.forward_one_layer(
                    query_feature_memory[i + 1], pixel_feature_memory[i + 1],
                    query_pos_embeds, pixel_pos_embeds, gt_attn_mask, i)
                outputs_class, outputs_mask, _, _, _ = \
                    self.forward_prediction_heads(
                        query_features, pixel_features, pixel_feature_size,
                        idx_layer=i)
                guided_predictions_class.append(outputs_class)
                guided_predictions_mask.append(outputs_mask)
                guided_predictions_matching_index.append(matching_indices)

        predictions_class = guided_predictions_class + predictions_class
        predictions_mask = guided_predictions_mask + predictions_mask
        predictions_matching_index = (guided_predictions_matching_index +
                                      predictions_matching_index)

        out = {
            "proposal_cls_logits": proposal_cls_logits,
            "query_locations": query_locations,
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "pred_matching_indices": predictions_matching_index[-1],
            "aux_outputs": self._set_aux_loss(
                predictions_class, predictions_mask, predictions_matching_index,
                query_locations),
        }
        return out

    def forward_one_layer(self, query_features, pixel_features, query_pos_embeds,
                          pixel_pos_embeds, attn_mask, i):
        pixel_features = self.transformer_mask_cross_attention_layers[i](
            pixel_features, query_features, query_pos=pixel_pos_embeds,
            pos=query_pos_embeds)
        pixel_features = self.transformer_mask_ffn_layers[i](pixel_features)

        query_features = self.transformer_query_cross_attention_layers[i](
            query_features, pixel_features, memory_mask=attn_mask,
            query_pos=query_pos_embeds, pos=pixel_pos_embeds)
        query_features = self.transformer_query_self_attention_layers[i](
            query_features, query_pos=query_pos_embeds)
        query_features = self.transformer_query_ffn_layers[i](query_features)
        return query_features, pixel_features

    def forward_prediction_heads(self, query_features, pixel_features,
                                 pixel_feature_size, idx_layer,
                                 return_attn_mask=False, return_gt_attn_mask=False,
                                 targets=None, query_locations=None):
        decoder_query_features = self.decoder_query_norm_layers[idx_layer + 1](
            query_features[:self.num_queries])
        decoder_query_features = decoder_query_features.transpose(0, 1)
        if self.training or idx_layer + 1 == self.num_layers:
            outputs_class = self.class_embed_layers[idx_layer + 1](
                decoder_query_features)
        else:
            outputs_class = None
        outputs_mask_embed = self.mask_embed_layers[idx_layer + 1](
            decoder_query_features)
        outputs_mask_features = self.mask_features_layers[idx_layer + 1](
            pixel_features.transpose(0, 1))

        outputs_mask = torch.einsum("bqc,blc->bql", outputs_mask_embed,
                                    outputs_mask_features)
        outputs_mask = outputs_mask.reshape(-1, self.num_queries,
                                            *pixel_feature_size)

        if return_attn_mask:
            attn_mask = F.pad(outputs_mask, (0, 0, 0, 0, 0, self.num_aux_queries),
                              "constant", 1)
            attn_mask = (attn_mask < 0.).flatten(2)                   # b, q, hw
            invalid_query = attn_mask.all(-1, keepdim=True)           # b, q, 1
            attn_mask = (~invalid_query) & attn_mask
            attn_mask = attn_mask.unsqueeze(1).repeat(
                1, self.num_heads, 1, 1).flatten(0, 1)
            attn_mask = attn_mask.detach()
        else:
            attn_mask = None

        if return_gt_attn_mask:
            assert targets is not None and query_locations is not None
            matching_indices = self.criterion.matcher(
                {"pred_logits": outputs_class, "pred_masks": outputs_mask,
                 "query_locations": query_locations}, targets)
            src_idx = self.criterion._get_src_permutation_idx(matching_indices)
            tgt_idx = self.criterion._get_tgt_permutation_idx(matching_indices)

            mask = [t["masks"] for t in targets]
            target_mask, _ = nested_tensor_from_tensor_list(mask)
            if target_mask.shape[1] > 0:
                target_mask = target_mask.to(outputs_mask)
                target_mask = F.interpolate(target_mask, size=pixel_feature_size,
                                            mode="nearest").bool()
            else:
                target_mask_size = [target_mask.shape[0], target_mask.shape[1],
                                    *pixel_feature_size]
                target_mask = torch.zeros(size=target_mask_size,
                                          device=outputs_mask.device).bool()

            gt_attn_mask_size = [outputs_mask.shape[0],
                                 self.num_queries + self.num_aux_queries,
                                 *pixel_feature_size]
            gt_attn_mask = torch.zeros(size=gt_attn_mask_size,
                                       device=outputs_mask.device).bool()
            gt_attn_mask[src_idx] = ~target_mask[tgt_idx]
            gt_attn_mask = gt_attn_mask.flatten(2)

            invalid_gt_query = gt_attn_mask.all(-1, keepdim=True)     # b, n, 1
            gt_attn_mask = (~invalid_gt_query) & gt_attn_mask
            gt_attn_mask = gt_attn_mask.unsqueeze(1).repeat(
                1, self.num_heads, 1, 1).flatten(0, 1)
            gt_attn_mask = gt_attn_mask.detach()
        else:
            matching_indices = None
            gt_attn_mask = None

        return outputs_class, outputs_mask, attn_mask, matching_indices, gt_attn_mask

    @staticmethod
    def _set_aux_loss(outputs_class, outputs_seg_masks, output_indices,
                      output_query_locations):
        return [
            {"query_locations": output_query_locations,
             "pred_logits": a, "pred_masks": b, "pred_matching_indices": c}
            for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1],
                               output_indices[:-1])
        ]


# ---------------------------------------------------------------------------
# 整网封装（fastinst/fastinst.py 对齐）
# ---------------------------------------------------------------------------
class FastInst(nn.Module):
    """FastInst 整网：归一化 + ResNet50 + PPM-FPN + FastInstDecoder + SetCriterion。

    训练: forward(images, targets) -> 加权 loss dict（loss_ce/loss_mask/loss_dice/
          loss_proposal + 深度监督 aux 项）
    评估: forward(images) -> [{boxes, labels, scores, masks[N,1,H,W] 概率}]
    """

    def __init__(self, num_classes: int = 3, pretrained_backbone: bool = True,
                 arch: dict | None = None):
        super().__init__()
        a = dict(arch or {})
        self.num_classes = num_classes
        self.num_queries = int(a.get("num_object_queries", 100))
        self.test_topk = int(a.get("test_topk_per_image", 100))
        # detectron2 PIXEL_MEAN/STD（0-255 RGB 口径；输入为 [0,1] 故乘 255）
        self.register_buffer(
            "pixel_mean", torch.tensor([123.675, 116.280, 103.530]).view(1, 3, 1, 1))
        self.register_buffer(
            "pixel_std", torch.tensor([58.395, 57.120, 57.375]).view(1, 3, 1, 1))

        weights = (torchvision.models.ResNet50_Weights.IMAGENET1K_V2
                   if pretrained_backbone else None)
        r50 = torchvision.models.resnet50(weights=weights)
        self.stem = nn.Sequential(r50.conv1, r50.bn1, r50.relu, r50.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            r50.layer1, r50.layer2, r50.layer3, r50.layer4)

        self.pixel_decoder = PyramidPoolingModuleFPN(
            {"res3": (512, 8), "res4": (1024, 16), "res5": (2048, 32)},
            convs_dim=int(a.get("convs_dim", 256)))

        self.predictor = FastInstDecoder(
            in_channels=int(a.get("convs_dim", 256)),
            num_classes=num_classes,
            hidden_dim=int(a.get("hidden_dim", 256)),
            num_queries=self.num_queries,
            num_aux_queries=int(a.get("num_aux_queries", 8)),
            nheads=int(a.get("nheads", 8)),
            dim_feedforward=int(a.get("dim_feedforward", 1024)),
            dec_layers=int(a.get("dec_layers", 3)),
            pre_norm=bool(a.get("pre_norm", False)),
            mask_dim=int(a.get("mask_dim", 256)),
        )

        class_w = float(a.get("class_weight", 2.0))
        mask_w = float(a.get("mask_weight", 5.0))
        dice_w = float(a.get("dice_weight", 5.0))
        loc_w = float(a.get("cost_location", 1000.0))
        prop_w = float(a.get("proposal_weight", 20.0))
        no_obj_w = float(a.get("no_object_weight", 0.1))
        matcher = HungarianMatcher(
            cost_class=class_w, cost_mask=mask_w, cost_dice=dice_w,
            cost_location=loc_w,
            num_points=int(a.get("train_num_points", 12544)))
        weight_dict = {"loss_ce": class_w, "loss_mask": mask_w, "loss_dice": dice_w}
        dec_layers = int(a.get("dec_layers", 3))
        aux_weight_dict = {}
        for i in range(2 * dec_layers):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        weight_dict.update({"loss_proposal": prop_w})
        self.criterion = SetCriterion(
            num_classes, matcher=matcher, weight_dict=weight_dict,
            eos_coef=no_obj_w, losses=["labels", "masks"],
            num_points=int(a.get("train_num_points", 12544)),
            oversample_ratio=float(a.get("oversample_ratio", 3.0)),
            importance_sample_ratio=float(a.get("importance_sample_ratio", 0.75)))
        self.predictor.criterion = self.criterion

    def _backbone_features(self, x: torch.Tensor) -> dict:
        x = self.stem(x)
        x = self.layer1(x)
        f3 = self.layer2(x)
        f4 = self.layer3(f3)
        f5 = self.layer4(f4)
        return {"res3": f3, "res4": f4, "res5": f5}

    def forward(self, images: list[torch.Tensor], targets: list[dict] | None = None):
        x = torch.stack(images, 0)
        x = (x * 255.0 - self.pixel_mean) / self.pixel_std
        features = self._backbone_features(x)
        _, multi_scale_features = self.pixel_decoder.forward_features(features)

        if self.training:
            assert targets is not None, "训练模式必须提供 targets"
            new_targets = [{"labels": t["labels"], "masks": t["masks"].float()}
                           for t in targets]
            outputs = self.predictor(multi_scale_features, None, new_targets)
            losses = self.criterion(outputs, new_targets)
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] = losses[k] * self.criterion.weight_dict[k]
                else:
                    losses.pop(k)
            return losses

        # ---------------- 推理（instance_inference 官方口径） ----------------
        outputs = self.predictor(multi_scale_features, None, None)
        mask_cls_results = outputs["pred_logits"]                     # [B, Q, C+1]
        mask_pred_results = outputs["pred_masks"]                     # [B, Q, h8, w8]
        H, W = int(images[0].shape[-2]), int(images[0].shape[-1])
        mask_pred_results = F.interpolate(
            mask_pred_results, size=(H, W), mode="bilinear", align_corners=False)

        results = []
        for mask_cls_result, mask_pred_result in zip(mask_cls_results,
                                                     mask_pred_results):
            scores = F.softmax(mask_cls_result, dim=-1)[:, :-1]       # [Q, C]
            labels = torch.arange(self.num_classes, device=mask_cls_result.device) \
                .unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(
                min(self.test_topk, scores.numel()), sorted=False)
            labels_per_image = labels[topk_indices]
            topk_queries = topk_indices // self.num_classes
            mask_pred = mask_pred_result[topk_queries]                # [k, H, W]

            mask_pred_sigmoid = mask_pred.sigmoid()
            binary = (mask_pred_sigmoid > 0.5).float()
            mask_scores = (mask_pred_sigmoid.flatten(1) * binary.flatten(1)).sum(1) / \
                (binary.flatten(1).sum(1) + 1e-6)
            final_scores = scores_per_image * mask_scores

            # 清掉零面积掩码并从二值 mask 导出 xyxy 框（官方默认零框，此处为对接 COCO）
            boxes, sel = [], []
            for i in range(binary.shape[0]):
                m = binary[i] > 0.5
                ys, xs = m.nonzero(as_tuple=True)
                if xs.numel() == 0:
                    continue
                sel.append(i)
                boxes.append([float(xs.min()), float(ys.min()),
                              float(xs.max()) + 1.0, float(ys.max()) + 1.0])
            if not sel:
                results.append({
                    "boxes": torch.zeros(0, 4, device=final_scores.device),
                    "labels": torch.zeros(0, dtype=torch.int64,
                                          device=final_scores.device),
                    "scores": torch.zeros(0, device=final_scores.device),
                    "masks": torch.zeros(0, 1, H, W, device=final_scores.device)})
                continue
            sel_t = torch.tensor(sel, device=final_scores.device)
            results.append({
                "boxes": torch.tensor(boxes, device=final_scores.device),
                "labels": labels_per_image[sel_t],
                "scores": final_scores[sel_t],
                "masks": mask_pred_sigmoid[sel_t][:, None]})
        return results


def build_fastinst(num_classes: int = 3, input_size: int = 1024,
                   pretrained_backbone: bool = True, arch: dict | None = None) -> nn.Module:
    """统一构建入口（train.py 与 tools/evaluate.py 共用）。input_size 由协议固定。"""
    return FastInst(num_classes=num_classes, pretrained_backbone=pretrained_backbone,
                    arch=arch)
