from abc import ABC, abstractmethod
import numpy as np
import torch

class BaseModelClass(ABC):
    """Base interface for SAS models"""
    @abstractmethod
    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Frame to normalized tensor ready for the model"""

    @abstractmethod
    def _postprocess(self, raw_output) -> tuple[np.ndarray, float]:
        """Raw model output to (binary mask, confidence)"""

    def infer(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """Run the full inference pipeline on a single frame"""
        tensor = self._preprocess(frame)
        with torch.no_grad():
            raw = self.model(tensor)
        return self._postprocess(raw)

    def __call__(self, *args, **kwargs) -> any:
        """Callable interface"""
        return self.infer(*args, **kwargs)
    
    def get_model_info(self):
        return {"name": self.__class__.__name__, "input_size": self.INPUT_SIZE}

    def release(self) -> None:
        """Release any resources if needed (e.g., GPU memory)"""
        del self.model