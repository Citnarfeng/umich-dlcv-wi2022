# Core implementation adapted from University of Michigan EECS 498/598 course starter code.
# Static validation only; not executed.
from typing import Dict, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import feature_extraction

def hello_common():
    print('Hello from common.py!')

class DetectorBackboneWithFPN(nn.Module):

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        _cnn = models.regnet_x_400mf(pretrained=True)
        self.backbone = feature_extraction.create_feature_extractor(_cnn, return_nodes={'trunk_output.block2': 'c3', 'trunk_output.block3': 'c4', 'trunk_output.block4': 'c5'})
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]
        print('For dummy input images with shape: (2, 3, 224, 224)')
        for level_name, feature_shape in dummy_out_shapes:
            print(f'Shape of {level_name} features: {feature_shape}')
        self.fpn_params = nn.ModuleDict()
        for level, shape in dummy_out_shapes:
            self.fpn_params[f'lateral_{level}'] = nn.Conv2d(shape[1], out_channels, 1)
            self.fpn_params[f'output_{level}'] = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    @property
    def fpn_strides(self):
        return {'p3': 8, 'p4': 16, 'p5': 32}

    def forward(self, images: torch.Tensor):
        backbone_feats = self.backbone(images)
        fpn_feats = {'p3': None, 'p4': None, 'p5': None}
        merged = None
        for level in ('c5', 'c4', 'c3'):
            lateral = self.fpn_params[f'lateral_{level}'](backbone_feats[level])
            merged = lateral if merged is None else lateral + F.interpolate(merged, size=lateral.shape[-2:], mode='nearest')
            fpn_feats['p' + level[1:]] = self.fpn_params[f'output_{level}'](merged)
        return fpn_feats

def get_fpn_location_coords(shape_per_fpn_level: Dict[str, Tuple], strides_per_fpn_level: Dict[str, int], dtype: torch.dtype=torch.float32, device: str='cpu') -> Dict[str, torch.Tensor]:
    location_coords = {level_name: None for level_name, _ in shape_per_fpn_level.items()}
    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]
        h, w = feat_shape[-2:]
        x, y = torch.meshgrid((torch.arange(w, dtype=dtype, device=device) + 0.5) * level_stride, (torch.arange(h, dtype=dtype, device=device) + 0.5) * level_stride, indexing='ij')
        location_coords[level_name] = torch.stack((x, y), dim=-1).reshape(-1, 2)
    return location_coords

def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float=0.5):
    if not boxes.numel() or not scores.numel():
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    keep = None
    areas = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(dim=1)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel():
        i = order[0]
        keep.append(i)
        order = order[1:]
        intersection = (torch.minimum(boxes[i, 2:], boxes[order, 2:]) - torch.maximum(boxes[i, :2], boxes[order, :2])).clamp(min=0).prod(dim=1)
        union = areas[i] + areas[order] - intersection
        iou = intersection / union.clamp(min=torch.finfo(boxes.dtype).tiny)
        order = order[iou <= iou_threshold]
    keep = torch.stack(keep)
    return keep

def class_spec_nms(boxes: torch.Tensor, scores: torch.Tensor, class_ids: torch.Tensor, iou_threshold: float=0.5):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep
