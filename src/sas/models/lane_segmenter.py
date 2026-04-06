import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from .builder import LANE_SEGMENTERS
from .base_model import BaseModelClass
from .segmentation.autodrive_erfnet import ERFNet
from sas.utils.preprocessing import imagenet_preprocess, yolox_preprocess

@LANE_SEGMENTERS.register()
class CityscapesERFNet(BaseModelClass):
    INPUT_SIZE = (976, 208)
    MEAN_BGR = [103.939, 116.779, 123.68]

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        # Preprocess frame for model input
        # Resize, normalize, etc.
        return preprocessed_frame
    
    def _postprocess(self, raw_output) -> tuple[np.ndarray, float, dict]:
        # Convert model output to lane segmentation mask
        return mask, confidence, {}


@LANE_SEGMENTERS.register()
class AutoDriveERFNet(BaseModelClass):
    def __init__(self, model_path, input_size=(288, 800), dataset='culane', device='cpu'):
        self.INPUT_SIZE = input_size
        self.device = device
        self.model = ERFNet(
            num_classes=5 if dataset == 'culane' else 7,    # TODO: adjust for othe datasets
            dropout_1=0.1,
            dropout_2=0.1,
            pretrained_weights=None,
            lane_exist_cfg=dict(
                name='EDLaneExist',
                num_output=4,
                flattened_size=3965,
                dropout=0.1,
                pool='max'
            )
        )
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint['state_dict']
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.to(device).eval()

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        preprocessed = imagenet_preprocess(frame, self.INPUT_SIZE)
        return torch.from_numpy(preprocessed).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def _postprocess(self, raw_output) -> tuple[np.ndarray, float, dict]:
        # Segmentation mask: argmax over class dimention -> binary lane mask
        seg_probs = F.softmax(raw_output['out'], dim=1) # (1, 5, H, W)
        class_mask = seg_probs.argmax(dim=1).squeeze(0) # (H, W) with values in {0, 1, 2, 3, 4}
        lane_mask = (class_mask > 0).cpu().numpy().astype(np.uint8) * 255

        # Confidence: mean sigmoid existence prob of ego lanes
        exist = torch.sigmoid(raw_output['lane']).squeeze(0) # (4,)
        confidence = float(exist[1:3].mean()) # average confidence of ego lane existence
        return lane_mask, confidence, {}