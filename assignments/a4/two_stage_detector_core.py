# Core implementation adapted from University of Michigan EECS 498/598 course starter code.
# Static validation only; not executed.
import math
from typing import Dict, List, Optional, Tuple
import torch
import torchvision
from a4_helper_core import *
from common_core import class_spec_nms, get_fpn_location_coords, nms
from torch import nn
from torch.nn import functional as F
TensorDict = Dict[str, torch.Tensor]

def hello_two_stage_detector():
    print('Hello from two_stage_detector.py!')

class RPNPredictionNetwork(nn.Module):

    def __init__(self, in_channels: int, stem_channels: List[int], num_anchors: int=3):
        super().__init__()
        self.num_anchors = num_anchors
        stem_rpn = []
        for channels in stem_channels:
            conv = nn.Conv2d(in_channels, channels, 3, padding=1)
            nn.init.normal_(conv.weight, std=0.01)
            nn.init.zeros_(conv.bias)
            stem_rpn.extend((conv, nn.ReLU()))
            in_channels = channels
        self.stem_rpn = nn.Sequential(*stem_rpn)
        self.pred_obj = None
        self.pred_box = None
        self.pred_obj = nn.Conv2d(in_channels, num_anchors, 1)
        self.pred_box = nn.Conv2d(in_channels, 4 * num_anchors, 1)
        for layer in (self.pred_obj, self.pred_box):
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, feats_per_fpn_level: TensorDict) -> List[TensorDict]:
        object_logits = {}
        boxreg_deltas = {}
        for level, feats in feats_per_fpn_level.items():
            feats = self.stem_rpn(feats)
            object_logits[level] = self.pred_obj(feats).permute(0, 3, 2, 1).reshape(feats.shape[0], -1)
            boxreg_deltas[level] = self.pred_box(feats).permute(0, 3, 2, 1).reshape(feats.shape[0], -1, 4)
        return [object_logits, boxreg_deltas]

@torch.no_grad()
def generate_fpn_anchors(locations_per_fpn_level: TensorDict, strides_per_fpn_level: Dict[str, int], stride_scale: int, aspect_ratios: List[float]=[0.5, 1.0, 2.0]):
    anchors_per_fpn_level = {level_name: None for level_name, _ in locations_per_fpn_level.items()}
    for level_name, locations in locations_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]
        anchor_boxes = []
        for aspect_ratio in aspect_ratios:
            width = stride_scale * level_stride / math.sqrt(aspect_ratio)
            height = width * aspect_ratio
            half = locations.new_tensor([width, height]) / 2
            anchor_boxes.append(torch.cat((locations - half, locations + half), dim=1))
        anchor_boxes = torch.stack(anchor_boxes)
        anchor_boxes = anchor_boxes.permute(1, 0, 2).contiguous().view(-1, 4)
        anchors_per_fpn_level[level_name] = anchor_boxes
    return anchors_per_fpn_level

@torch.no_grad()
def iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    intersection = (torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:]) - torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])).clamp(min=0).prod(dim=2)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp(min=0).prod(dim=1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp(min=0).prod(dim=1)
    iou = intersection / (area1[:, None] + area2[None, :] - intersection).clamp(min=torch.finfo(intersection.dtype).tiny)
    return iou

@torch.no_grad()
def rcnn_match_anchors_to_gt(anchor_boxes: torch.Tensor, gt_boxes: torch.Tensor, iou_thresholds: Tuple[float, float]) -> TensorDict:
    gt_boxes = gt_boxes[gt_boxes[:, 4] != -1]
    if len(gt_boxes) == 0:
        fake_boxes = torch.zeros_like(anchor_boxes) - 1
        fake_class = torch.zeros_like(anchor_boxes[:, [0]]) - 1
        return torch.cat([fake_boxes, fake_class], dim=1)
    match_matrix = iou(anchor_boxes, gt_boxes[:, :4])
    match_quality, matched_idxs = match_matrix.max(dim=1)
    matched_gt_boxes = gt_boxes[matched_idxs]
    matched_gt_boxes[match_quality <= iou_thresholds[0]] = -1
    neutral_idxs = (match_quality > iou_thresholds[0]) & (match_quality < iou_thresholds[1])
    matched_gt_boxes[neutral_idxs, :] = -100000000.0
    return matched_gt_boxes

def rcnn_get_deltas_from_anchors(anchors: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    deltas = None
    size = anchors[:, 2:] - anchors[:, :2]
    gt_size = gt_boxes[:, 2:4] - gt_boxes[:, :2]
    deltas = torch.cat(((gt_boxes[:, :2] + gt_size / 2 - anchors[:, :2] - size / 2) / size, (gt_size.clamp(min=torch.finfo(size.dtype).tiny) / size).log()), dim=1)
    invalid = (gt_boxes[:, :4] == -1).all(dim=1) | (gt_boxes[:, :4] == -100000000.0).all(dim=1)
    deltas[invalid] = -100000000.0
    return deltas

def rcnn_apply_deltas_to_anchors(deltas: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    scale_clamp = math.log(224 / 8)
    output_boxes = None
    size = anchors[:, 2:] - anchors[:, :2]
    center = anchors[:, :2] + size / 2 + deltas[:, :2] * size
    size = size * deltas[:, 2:].clamp(max=scale_clamp).exp()
    output_boxes = torch.cat((center - size / 2, center + size / 2), dim=1)
    return output_boxes

@torch.no_grad()
def sample_rpn_training(gt_boxes: torch.Tensor, num_samples: int, fg_fraction: float):
    foreground = (gt_boxes[:, 4] >= 0).nonzero().squeeze(1)
    background = (gt_boxes[:, 4] == -1).nonzero().squeeze(1)
    num_fg = min(int(num_samples * fg_fraction), foreground.numel())
    num_bg = num_samples - num_fg
    perm1 = torch.randperm(foreground.numel(), device=foreground.device)[:num_fg]
    perm2 = torch.randperm(background.numel(), device=background.device)[:num_bg]
    fg_idx = foreground[perm1]
    bg_idx = background[perm2]
    return (fg_idx, bg_idx)

@torch.no_grad()
def mix_gt_with_proposals(proposals_per_fpn_level: Dict[str, List[torch.Tensor]], gt_boxes: torch.Tensor):
    for _idx, _gtb in enumerate(gt_boxes):
        _gtb = _gtb[_gtb[:, 4] != -1]
        if len(_gtb) == 0:
            continue
        _gt_area = (_gtb[:, 2] - _gtb[:, 0]) * (_gtb[:, 3] - _gtb[:, 1])
        level_assn = torch.floor(5 + torch.log2(torch.sqrt(_gt_area) / 224))
        level_assn = torch.clamp(level_assn, min=3, max=5).to(torch.int64)
        for level_name, _props in proposals_per_fpn_level.items():
            _prop = _props[_idx]
            _gt_boxes_fpn_subset = _gtb[level_assn == int(level_name[1])]
            if len(_gt_boxes_fpn_subset) > 0:
                proposals_per_fpn_level[level_name][_idx] = torch.cat([_prop, _gt_boxes_fpn_subset[:, :4]], dim=0)
    return proposals_per_fpn_level

class RPN(nn.Module):

    def __init__(self, fpn_channels: int, stem_channels: List[int], batch_size_per_image: int, anchor_stride_scale: int=8, anchor_aspect_ratios: List[int]=[0.5, 1.0, 2.0], anchor_iou_thresholds: Tuple[int, int]=(0.3, 0.6), nms_thresh: float=0.7, pre_nms_topk: int=400, post_nms_topk: int=100):
        super().__init__()
        self.pred_net = RPNPredictionNetwork(fpn_channels, stem_channels, num_anchors=len(anchor_aspect_ratios))
        self.batch_size_per_image = batch_size_per_image
        self.anchor_stride_scale = anchor_stride_scale
        self.anchor_aspect_ratios = anchor_aspect_ratios
        self.anchor_iou_thresholds = anchor_iou_thresholds
        self.nms_thresh = nms_thresh
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk

    def forward(self, feats_per_fpn_level: TensorDict, strides_per_fpn_level: TensorDict, gt_boxes: Optional[torch.Tensor]=None):
        num_images = feats_per_fpn_level['p3'].shape[0]
        pred_obj_logits, pred_boxreg_deltas, anchors_per_fpn_level = (None, None, None)
        pred_obj_logits, pred_boxreg_deltas = self.pred_net(feats_per_fpn_level)
        feats = feats_per_fpn_level['p3']
        locations = get_fpn_location_coords({k: v.shape for k, v in feats_per_fpn_level.items()}, strides_per_fpn_level, dtype=feats.dtype, device=feats.device)
        anchors_per_fpn_level = generate_fpn_anchors(locations, strides_per_fpn_level, self.anchor_stride_scale, self.anchor_aspect_ratios)
        output_dict = {}
        img_h = feats_per_fpn_level['p3'].shape[2] * strides_per_fpn_level['p3']
        img_w = feats_per_fpn_level['p3'].shape[3] * strides_per_fpn_level['p3']
        output_dict['proposals'] = self.predict_proposals(anchors_per_fpn_level, pred_obj_logits, pred_boxreg_deltas, (img_w, img_h))
        if not self.training:
            return output_dict
        anchor_boxes = self._cat_across_fpn_levels(anchors_per_fpn_level, dim=0)
        matched_gt_boxes = []
        matched_gt_boxes = [rcnn_match_anchors_to_gt(anchor_boxes, boxes, self.anchor_iou_thresholds) for boxes in gt_boxes]
        matched_gt_boxes = torch.stack(matched_gt_boxes, dim=0)
        pred_obj_logits = self._cat_across_fpn_levels(pred_obj_logits)
        pred_boxreg_deltas = self._cat_across_fpn_levels(pred_boxreg_deltas)
        if self.training:
            anchor_boxes = anchor_boxes.unsqueeze(0).repeat(num_images, 1, 1)
            anchor_boxes = anchor_boxes.contiguous().view(-1, 4)
            matched_gt_boxes = matched_gt_boxes.view(-1, 5)
            pred_obj_logits = pred_obj_logits.view(-1)
            pred_boxreg_deltas = pred_boxreg_deltas.view(-1, 4)
            loss_obj, loss_box = (None, None)
            fg, bg = sample_rpn_training(matched_gt_boxes, self.batch_size_per_image * num_images, 0.5)
            idx = torch.cat((fg, bg))
            targets = (matched_gt_boxes[idx, 4] >= 0).to(pred_obj_logits)
            loss_obj = F.binary_cross_entropy_with_logits(pred_obj_logits[idx], targets, reduction='none')
            deltas = rcnn_get_deltas_from_anchors(anchor_boxes[fg], matched_gt_boxes[fg, :4])
            loss_box = F.l1_loss(pred_boxreg_deltas[fg], deltas, reduction='none')
            total_batch_size = self.batch_size_per_image * num_images
            output_dict['loss_rpn_obj'] = loss_obj.sum() / total_batch_size
            output_dict['loss_rpn_box'] = loss_box.sum() / total_batch_size
        return output_dict

    @torch.no_grad()
    def predict_proposals(self, anchors_per_fpn_level: Dict[str, torch.Tensor], pred_obj_logits: Dict[str, torch.Tensor], pred_boxreg_deltas: Dict[str, torch.Tensor], image_size: Tuple[int, int]):
        proposals_all_levels = {level_name: None for level_name, _ in anchors_per_fpn_level.items()}
        for level_name in anchors_per_fpn_level.keys():
            level_anchors = anchors_per_fpn_level[level_name]
            level_obj_logits = pred_obj_logits[level_name]
            level_boxreg_deltas = pred_boxreg_deltas[level_name]
            level_proposals_per_image = []
            for _batch_idx in range(level_obj_logits.shape[0]):
                boxes = rcnn_apply_deltas_to_anchors(level_boxreg_deltas[_batch_idx], level_anchors)
                boxes[:, 0::2].clamp_(min=0, max=image_size[0])
                boxes[:, 1::2].clamp_(min=0, max=image_size[1])
                scores, idx = level_obj_logits[_batch_idx].topk(min(self.pre_nms_topk, len(boxes)))
                boxes = boxes[idx]
                keep = torchvision.ops.nms(boxes, scores, self.nms_thresh)[:self.post_nms_topk]
                level_proposals_per_image.append(boxes[keep])
            proposals_all_levels[level_name] = level_proposals_per_image
        return proposals_all_levels

    @staticmethod
    def _cat_across_fpn_levels(dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int=1):
        return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)

class FasterRCNN(nn.Module):

    def __init__(self, backbone: nn.Module, rpn: nn.Module, stem_channels: List[int], num_classes: int, batch_size_per_image: int, roi_size: Tuple[int, int]=(7, 7)):
        super().__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.num_classes = num_classes
        self.roi_size = roi_size
        self.batch_size_per_image = batch_size_per_image
        cls_pred = []
        in_channels = backbone.out_channels
        for channels in stem_channels:
            conv = nn.Conv2d(in_channels, channels, 3, padding=1)
            nn.init.normal_(conv.weight, std=0.01)
            nn.init.zeros_(conv.bias)
            cls_pred.extend((conv, nn.ReLU()))
            in_channels = channels
        linear = nn.Linear(in_channels * roi_size[0] * roi_size[1], num_classes + 1)
        nn.init.normal_(linear.weight, std=0.01)
        nn.init.zeros_(linear.bias)
        cls_pred.extend((nn.Flatten(), linear))
        self.cls_pred = nn.Sequential(*cls_pred)

    def forward(self, images: torch.Tensor, gt_boxes: Optional[torch.Tensor]=None, test_score_thresh: Optional[float]=None, test_nms_thresh: Optional[float]=None):
        feats_per_fpn_level = self.backbone(images)
        output_dict = self.rpn(feats_per_fpn_level, self.backbone.fpn_strides, gt_boxes)
        proposals_per_fpn_level = output_dict['proposals']
        if self.training:
            proposals_per_fpn_level = mix_gt_with_proposals(proposals_per_fpn_level, gt_boxes)
        num_images = feats_per_fpn_level['p3'].shape[0]
        roi_feats_per_fpn_level = {level_name: None for level_name in feats_per_fpn_level.keys()}
        for level_name in feats_per_fpn_level.keys():
            level_feats = feats_per_fpn_level[level_name]
            level_props = output_dict['proposals'][level_name]
            level_stride = self.backbone.fpn_strides[level_name]
            roi_feats = torchvision.ops.roi_align(level_feats, level_props, self.roi_size, spatial_scale=1 / level_stride, aligned=True)
            roi_feats_per_fpn_level[level_name] = roi_feats
        roi_feats = self._cat_across_fpn_levels(roi_feats_per_fpn_level, dim=0)
        pred_cls_logits = self.cls_pred(roi_feats)
        if not self.training:
            return self.inference(images, proposals_per_fpn_level, pred_cls_logits, test_score_thresh=test_score_thresh, test_nms_thresh=test_nms_thresh)
        matched_gt_boxes = [rcnn_match_anchors_to_gt(props[i], gt_boxes[i], (0.5, 0.5)) for props in proposals_per_fpn_level.values() for i in range(num_images)]
        matched_gt_boxes = torch.cat(matched_gt_boxes, dim=0)
        loss_cls = None
        fg, bg = sample_rpn_training(matched_gt_boxes, self.batch_size_per_image * num_images, 0.25)
        idx = torch.cat((fg, bg))
        loss_cls = F.cross_entropy(pred_cls_logits[idx], (matched_gt_boxes[idx, 4] + 1).long(), reduction='sum')
        loss_cls = loss_cls / max(idx.numel(), 1)
        return {'loss_rpn_obj': output_dict['loss_rpn_obj'], 'loss_rpn_box': output_dict['loss_rpn_box'], 'loss_cls': loss_cls}

    @staticmethod
    def _cat_across_fpn_levels(dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int=1):
        return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)

    def inference(self, images: torch.Tensor, proposals: torch.Tensor, pred_cls_logits: torch.Tensor, test_score_thresh: float, test_nms_thresh: float):
        if images.shape[0] != 1:
            raise ValueError('Faster R-CNN inference requires batch size 1')
        test_score_thresh = 0.2 if test_score_thresh is None else test_score_thresh
        test_nms_thresh = 0.5 if test_nms_thresh is None else test_nms_thresh
        proposals = {level_name: prop[0] for level_name, prop in proposals.items()}
        pred_boxes = self._cat_across_fpn_levels(proposals, dim=0)
        pred_scores, pred_classes = (None, None)
        pred_scores, pred_classes = pred_cls_logits.softmax(dim=1).max(dim=1)
        keep = (pred_classes > 0) & (pred_scores > test_score_thresh)
        pred_boxes, pred_scores, pred_classes = (pred_boxes[keep], pred_scores[keep], pred_classes[keep] - 1)
        keep = class_spec_nms(pred_boxes, pred_scores, pred_classes, iou_threshold=test_nms_thresh)
        pred_boxes = pred_boxes[keep]
        pred_classes = pred_classes[keep]
        pred_scores = pred_scores[keep]
        return (pred_boxes, pred_classes, pred_scores)
