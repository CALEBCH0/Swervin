import os
from sas.models.builder import LANE_SEGMENTERS
from sas.utils.SAS_Process import SASProcessClass
import sas.models.lane_segmenter
import sas.models.optimized_models  # Import ONNX and TensorRT models

def init_sas_process(config):
    """
    Initialize SAS process with models from registry
    Args:
        config: Configuration object
    
    Returns:
        SAS process instance with loaded models
    """
    # Resolve full model path by combining base path + filename
    lane_params = vars(config.lane_segmenter).copy()
    lane_params['model_path'] = os.path.join(config.model_path.path_models, lane_params['model_path'])
    
    lane_segmenter = LANE_SEGMENTERS.from_dict(lane_params)
    sas_process = SASProcessClass(
        config=config,
        lane_segmenter=lane_segmenter
        # TODO: add other models later
    )
    print("SAS process successfully initialized")
    return sas_process