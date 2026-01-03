import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

# 用于生成共享特征图
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)  # [B, 256, H, W]
    
# Faster R-CNN的RPN过程，用于生成proposal
class RPN(nn.Module):
    def __init__(self, in_channels=256, num_anchors=9):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.cls_logits = nn.Conv2d(256, num_anchors * 2, 1)
        self.bbox_pred = nn.Conv2d(256, num_anchors * 4, 1)

    def forward(self, feature):
        t = F.relu(self.conv(feature))
        logits = self.cls_logits(t)   # object / background
        bbox = self.bbox_pred(t)      # anchor offsets
        return logits, bbox  # 得到分类信息和位置信息
# 用于将得到的不同大小proposal统一成固定特征
def extract_roi_features(feature_map, proposals, output_size=7):
    """
    proposals: [num_rois, 5] -> (batch_idx, x1, y1, x2, y2)
    """
    roi_features = roi_align(
        feature_map,
        proposals,
        output_size=(output_size, output_size),
        spatial_scale=1.0,
        sampling_ratio=-1
    )
    return roi_features  # [N, C, 7, 7]

class FastRCNNHead(nn.Module):
    def __init__(self, in_channels=256, num_classes= 20):
        super().__init__()
        self.fc1 = nn.Linear(in_channels * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 1024)

        self.cls_score = nn.Linear(1024, num_classes + 1)  # + background
        self.bbox_pred = nn.Linear(1024, num_classes * 4)  # class-specific

    def forward(self, roi_features):
        x = roi_features.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas
    
class FasterRCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = Backbone()
        self.rpn = RPN()
        self.head = FastRCNNHead(num_classes=num_classes)

    def forward(self, images, proposals):
        features = self.backbone(images)
        rpn_logits, rpn_bbox = self.rpn(features)

        roi_features = extract_roi_features(features, proposals)
        cls_scores, bbox_preds = self.head(roi_features)

        return cls_scores, bbox_preds