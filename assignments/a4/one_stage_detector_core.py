# Core implementation adapted from University of Michigan EECS 498/598 course starter code.
# Static validation only; not executed.
import math
from typing import Dict, List, Optional
import torch
from a4_helper_core import *
from common_core import DetectorBackboneWithFPN, class_spec_nms, get_fpn_location_coords
from torch import nn
from torch.nn import functional as F
from torch.utils.data._utils.collate import default_collate
from torchvision.ops import sigmoid_focal_loss
TensorDict = Dict[str, torch.Tensor]

def hello_one_stage_detector():
    print('Hello from one_stage_detector.py!')

class FCOSPredictionNetwork(nn.Module):

    def __init__(self, num_classes: int, in_channels: int, stem_channels: List[int]):
        super().__init__()
        stem_cls = []
        stem_box = []
        for channels in stem_channels:
            for stem in (stem_cls, stem_box):
                conv = nn.Conv2d(in_channels, channels, 3, padding=1)
                nn.init.normal_(conv.weight, std=0.01)
                nn.init.zeros_(conv.bias)
                stem.extend((conv, nn.ReLU()))
            in_channels = channels
        self.stem_cls = nn.Sequential(*stem_cls)
        self.stem_box = nn.Sequential(*stem_box)
        self.pred_cls = None
        self.pred_box = None
        self.pred_ctr = None
        self.pred_cls = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.pred_box = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.pred_ctr = nn.Conv2d(in_channels, 1, 3, padding=1)
        for layer in (self.pred_cls, self.pred_box, self.pred_ctr):
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)
        torch.nn.init.constant_(self.pred_cls.bias, -math.log(99))

    def forward(self, feats_per_fpn_level: TensorDict) -> List[TensorDict]:
        class_logits = {}
        boxreg_deltas = {}
        centerness_logits = {}
        for level, feats in feats_per_fpn_level.items():
            cls, box = (self.stem_cls(feats), self.stem_box(feats))
            for output, pred in ((class_logits, self.pred_cls(cls)), (boxreg_deltas, self.pred_box(box)), (centerness_logits, self.pred_ctr(box))):
                output[level] = pred.permute(0, 3, 2, 1).reshape(pred.shape[0], -1, pred.shape[1])
        return [class_logits, boxreg_deltas, centerness_logits]

@torch.no_grad()
def fcos_match_locations_to_gt(locations_per_fpn_level: TensorDict, strides_per_fpn_level: Dict[str, int], gt_boxes: torch.Tensor) -> TensorDict:
    matched_gt_boxes = {level_name: None for level_name in locations_per_fpn_level.keys()}
    if gt_boxes.shape[0] == 0:
        return {level: gt_boxes.new_full((len(centers), 5), -1) for level, centers in locations_per_fpn_level.items()}
    for level_name, centers in locations_per_fpn_level.items():
        stride = strides_per_fpn_level[level_name]
        x, y = centers.unsqueeze(dim=2).unbind(dim=1)
        x0, y0, x1, y1 = gt_boxes[:, :4].unsqueeze(dim=0).unbind(dim=2)
        pairwise_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)
        pairwise_dist = pairwise_dist.permute(1, 0, 2)
        match_matrix = pairwise_dist.min(dim=2).values > 0
        pairwise_dist = pairwise_dist.max(dim=2).values
        lower_bound = stride * 4 if level_name != 'p3' else 0
        upper_bound = stride * 8 if level_name != 'p5' else float('inf')
        match_matrix &= (pairwise_dist > lower_bound) & (pairwise_dist < upper_bound)
        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        match_matrix = match_matrix.to(torch.float32)
        match_matrix *= 100000000.0 - gt_areas[:, None]
        match_quality, matched_idxs = match_matrix.max(dim=0)
        matched_idxs[match_quality < 1e-05] = -1
        matched_boxes_this_level = gt_boxes[matched_idxs.clip(min=0)]
        matched_boxes_this_level[matched_idxs < 0, :] = -1
        matched_gt_boxes[level_name] = matched_boxes_this_level
    return matched_gt_boxes

def fcos_get_deltas_from_locations(locations: torch.Tensor, gt_boxes: torch.Tensor, stride: int) -> torch.Tensor:
    deltas = None
    deltas = torch.cat((locations - gt_boxes[:, :2], gt_boxes[:, 2:4] - locations), dim=1) / stride
    deltas[(gt_boxes[:, :4] == -1).all(dim=1)] = -1
    return deltas

def fcos_apply_deltas_to_locations(deltas: torch.Tensor, locations: torch.Tensor, stride: int) -> torch.Tensor:
    deltas = deltas.clamp(min=0) * stride
    output_boxes = torch.cat((locations - deltas[:, :2], locations + deltas[:, 2:]), dim=1)
    return output_boxes

def fcos_make_centerness_targets(deltas: torch.Tensor):
    centerness = None
    numerator = torch.minimum(deltas[:, 0], deltas[:, 2]) * torch.minimum(deltas[:, 1], deltas[:, 3])
    denominator = torch.maximum(deltas[:, 0], deltas[:, 2]) * torch.maximum(deltas[:, 1], deltas[:, 3])
    centerness = (numerator / denominator.clamp(min=torch.finfo(deltas.dtype).tiny)).clamp(min=0).sqrt()
    centerness[(deltas == -1).all(dim=1)] = -1
    return centerness

class FCOS(nn.Module):

    def __init__(self, num_classes: int, fpn_channels: int, stem_channels: List[int]):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = None
        self.pred_net = None
        self.backbone = DetectorBackboneWithFPN(fpn_channels)
        self.pred_net = FCOSPredictionNetwork(num_classes, fpn_channels, stem_channels)
        self._normalizer = 150

    def forward(self, images: torch.Tensor, gt_boxes: Optional[torch.Tensor]=None, test_score_thresh: Optional[float]=None, test_nms_thresh: Optional[float]=None):
        pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits = (None, None, None)
        feats_per_fpn_level = self.backbone(images)
        pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits = self.pred_net(feats_per_fpn_level)
        locations_per_fpn_level = None
        locations_per_fpn_level = get_fpn_location_coords({k: v.shape for k, v in feats_per_fpn_level.items()}, self.backbone.fpn_strides, dtype=images.dtype, device=images.device)
        if not self.training:
            return self.inference(images, locations_per_fpn_level, pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits, test_score_thresh=test_score_thresh, test_nms_thresh=test_nms_thresh)
        matched_gt_boxes = []
        matched_gt_boxes = [fcos_match_locations_to_gt(locations_per_fpn_level, self.backbone.fpn_strides, boxes) for boxes in gt_boxes]
        matched_gt_deltas = []
        matched_gt_deltas = [{k: fcos_get_deltas_from_locations(locations_per_fpn_level[k], boxes[k], stride) for k, stride in self.backbone.fpn_strides.items()} for boxes in matched_gt_boxes]
        matched_gt_boxes = default_collate(matched_gt_boxes)
        matched_gt_deltas = default_collate(matched_gt_deltas)
        matched_gt_boxes = self._cat_across_fpn_levels(matched_gt_boxes)
        matched_gt_deltas = self._cat_across_fpn_levels(matched_gt_deltas)
        pred_cls_logits = self._cat_across_fpn_levels(pred_cls_logits)
        pred_boxreg_deltas = self._cat_across_fpn_levels(pred_boxreg_deltas)
        pred_ctr_logits = self._cat_across_fpn_levels(pred_ctr_logits)
        num_pos_locations = (matched_gt_boxes[:, :, 4] != -1).sum()
        pos_loc_per_image = num_pos_locations.item() / images.shape[0]
        self._normalizer = 0.9 * self._normalizer + 0.1 * pos_loc_per_image
        loss_cls, loss_box, loss_ctr = (None, None, None)
        fg = matched_gt_boxes[:, :, 4] >= 0
        labels = (matched_gt_boxes[:, :, 4] + 1).long()
        targets = F.one_hot(labels, self.num_classes + 1)[:, :, 1:].to(pred_cls_logits)
        loss_cls = sigmoid_focal_loss(pred_cls_logits, targets, reduction='none')
        loss_box = 0.25 * F.l1_loss(pred_boxreg_deltas[fg], matched_gt_deltas[fg], reduction='none')
        loss_ctr = F.binary_cross_entropy_with_logits(pred_ctr_logits.squeeze(-1)[fg], fcos_make_centerness_targets(matched_gt_deltas[fg]), reduction='none')
        return {'loss_cls': loss_cls.sum() / (self._normalizer * images.shape[0]), 'loss_box': loss_box.sum() / (self._normalizer * images.shape[0]), 'loss_ctr': loss_ctr.sum() / (self._normalizer * images.shape[0])}

    @staticmethod
    def _cat_across_fpn_levels(dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int=1):
        return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)

    def inference(self, images: torch.Tensor, locations_per_fpn_level: Dict[str, torch.Tensor], pred_cls_logits: Dict[str, torch.Tensor], pred_boxreg_deltas: Dict[str, torch.Tensor], pred_ctr_logits: Dict[str, torch.Tensor], test_score_thresh: float=0.3, test_nms_thresh: float=0.5):
        if images.shape[0] != 1:
            raise ValueError('FCOS inference requires batch size 1')
        test_score_thresh = 0.3 if test_score_thresh is None else test_score_thresh
        test_nms_thresh = 0.5 if test_nms_thresh is None else test_nms_thresh
        pred_boxes_all_levels = []
        pred_classes_all_levels = []
        pred_scores_all_levels = []
        for level_name in locations_per_fpn_level.keys():
            level_locations = locations_per_fpn_level[level_name]
            level_cls_logits = pred_cls_logits[level_name][0]
            level_deltas = pred_boxreg_deltas[level_name][0]
            level_ctr_logits = pred_ctr_logits[level_name][0]
            level_pred_boxes, level_pred_classes, level_pred_scores = (None, None, None)
            level_pred_scores = torch.sqrt(level_cls_logits.sigmoid() * level_ctr_logits.sigmoid())
            level_pred_scores, level_pred_classes = level_pred_scores.max(dim=1)
            keep = level_pred_scores > test_score_thresh
            level_pred_scores, level_pred_classes = (level_pred_scores[keep], level_pred_classes[keep])
            level_pred_boxes = fcos_apply_deltas_to_locations(level_deltas[keep], level_locations[keep], self.backbone.fpn_strides[level_name])
            level_pred_boxes[:, 0::2].clamp_(min=0, max=images.shape[-1])
            level_pred_boxes[:, 1::2].clamp_(min=0, max=images.shape[-2])
            pred_boxes_all_levels.append(level_pred_boxes)
            pred_classes_all_levels.append(level_pred_classes)
            pred_scores_all_levels.append(level_pred_scores)
        pred_boxes_all_levels = torch.cat(pred_boxes_all_levels)
        pred_classes_all_levels = torch.cat(pred_classes_all_levels)
        pred_scores_all_levels = torch.cat(pred_scores_all_levels)
        keep = class_spec_nms(pred_boxes_all_levels, pred_scores_all_levels, pred_classes_all_levels, iou_threshold=test_nms_thresh)
        pred_boxes_all_levels = pred_boxes_all_levels[keep]
        pred_classes_all_levels = pred_classes_all_levels[keep]
        pred_scores_all_levels = pred_scores_all_levels[keep]
        return (pred_boxes_all_levels, pred_classes_all_levels, pred_scores_all_levels)
