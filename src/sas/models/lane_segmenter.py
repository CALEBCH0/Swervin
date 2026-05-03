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
class CFLDERFNet(BaseModelClass):
    """
    ERFNet model trained on CuLane
    input resolution: 208x976 --> flattened size: 3965
    """
    INPUT_SIZE = (976, 208)
    MEAN_BGR = [103.939, 116.779, 123.68]

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        # Preprocess frame for model input
        # Resize, normalize, etc.
        return preprocessed_frame
    
    def _postprocess(self, raw_output) -> tuple[np.ndarray, float, dict]:
        # Convert model output to lane segmentation mask
        return mask, confidence


@LANE_SEGMENTERS.register()
class AutoDriveERFNet(BaseModelClass):
    """
    ERFNet model trained on CuLane
    input resolution: 288x800 --> flattened size: 4500
    """
    def __init__(self, model_path, input_size=(288, 800), dataset='culane', device='cpu'):
        self.INPUT_SIZE = input_size
        self.device = device

        # Load checkpoint first to infer architecture params
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        state_dict = checkpoint.get('model') or checkpoint.get('state_dict')
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        state_dict = {k.replace('aux_head.', 'lane_exist.', 1): v for k, v in state_dict.items()}

        # Infer flattened_size from checkpoint — avoids mismatch across training resolutions
        flattened_size = state_dict['lane_exist.linear1.weight'].shape[1]

        num_classes = 5 if dataset == 'culane' else 7
        self.model = ERFNet(
            num_classes=num_classes,
            dropout_1=0.1,
            dropout_2=0.1,
            pretrained_weights=None,
            lane_exist_cfg=dict(
                name='EDLaneExist',
                num_output=num_classes - 1,
                flattened_size=flattened_size,
                dropout=0.1,
                pool='max'
            )
        )
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
        return lane_mask, confidence