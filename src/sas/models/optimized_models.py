"""
ONNX and TensorRT model implementations for optimized inference
"""
import numpy as np
import cv2
from typing import Tuple, Dict

from .base_model import BaseModelClass
from .builder import LANE_SEGMENTERS
from sas.utils.preprocessing import imagenet_preprocess

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False


@LANE_SEGMENTERS.register()
class ONNXERFNet(BaseModelClass):
    """ONNX Runtime implementation of ERFNet for faster CPU/GPU inference"""
    
    def __init__(self, model_path, input_size=(288, 800), dataset='culane', device='cpu'):
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime not installed. Run: pip install onnxruntime-gpu")

        self.INPUT_SIZE = input_size
        self.device = device
        self.dataset = dataset
        
        # Set providers based on device
        if device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        # Create ONNX Runtime session
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get input/output info
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"✅ ONNX model loaded with providers: {self.session.get_providers()}")
        print(f"   Input names: {self.input_names}")
        print(f"   Output names: {self.output_names}")

        if device == 'cuda':
            print("   Warming up CUDA kernels (first-run compilation, may take 30-120s)...")
            dummy = np.zeros((1, 3, input_size[0], input_size[1]), dtype=np.float32)
            self.session.run(self.output_names, {self.input_names[0]: dummy})
            print("   CUDA warmup complete.")
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for ONNX model"""
        # Use same preprocessing as PyTorch version
        preprocessed = imagenet_preprocess(frame, self.INPUT_SIZE)
        # Add batch dimension and ensure float32
        preprocessed = np.expand_dims(preprocessed, axis=0).astype(np.float32)
        return preprocessed
    
    def _postprocess(self, raw_output) -> Tuple[np.ndarray, float, Dict]:
        """Process ONNX model output"""
        if isinstance(raw_output, list) and len(raw_output) >= 2:
            segmentation = raw_output[0]  # Lane segmentation
            lane_exist = raw_output[1]    # Lane existence confidence
        else:
            segmentation = raw_output[0] if isinstance(raw_output, list) else raw_output
            lane_exist = None
        
        # Remove batch dimension and get the lane mask
        seg_mask = segmentation[0]  # Shape: (num_classes, H, W)
        
        # Convert to binary mask (argmax across classes)
        binary_mask = np.argmax(seg_mask, axis=0).astype(np.uint8)
        
        if lane_exist is not None:
            lane_confidences = (1 / (1 + np.exp(-lane_exist[0]))).tolist()  # logits → probabilities
            confidence = float(max(lane_confidences))
        else:
            lane_confidences = None
            confidence = float(np.max(np.max(seg_mask, axis=(1, 2))))

        # Resize mask to original input size if needed
        if binary_mask.shape != self.INPUT_SIZE:
            binary_mask = cv2.resize(binary_mask, (self.INPUT_SIZE[1], self.INPUT_SIZE[0]))

        metadata = {
            'model_type': 'onnx',
            'num_classes': seg_mask.shape[0],
            'lane_confidences': lane_confidences,
            'dataset': self.dataset,
        }
        
        return binary_mask, confidence, metadata
    
    def infer(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Dict]:
        """Run inference on a single frame"""
        input_tensor = self._preprocess(frame)
        
        # Run ONNX inference
        raw_output = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        
        return self._postprocess(raw_output)


@LANE_SEGMENTERS.register()
class TensorRTERFNet(BaseModelClass):
    """TensorRT implementation of ERFNet for maximum GPU performance"""
    
    def __init__(self, model_path, input_size=(288, 800), dataset='culane', device='cuda'):
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT not installed. Install CUDA and TensorRT")
        
        if device != 'cuda':
            raise ValueError("TensorRT requires CUDA device")
        
        self.INPUT_SIZE = input_size
        self.device = device
        
        # Load TensorRT engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(model_path, 'rb') as f:
            engine_data = f.read()
        
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # Setup memory allocation
        self._setup_memory()
        
        print(f"✅ TensorRT engine loaded successfully")
        print(f"   Input shape: {self.input_shape}")
        print(f"   Output shapes: {self.output_shapes}")
    
    def _setup_memory(self):
        """Setup GPU memory allocation for TensorRT"""
        # Set input shape
        self.input_shape = (1, 3, self.INPUT_SIZE[0], self.INPUT_SIZE[1])
        self.context.set_binding_shape(0, self.input_shape)
        
        # Calculate memory sizes
        self.input_size = np.prod(self.input_shape) * np.dtype(np.float32).itemsize
        
        self.output_shapes = []
        self.output_sizes = []
        
        for i in range(1, self.engine.num_bindings):
            shape = self.context.get_binding_shape(i)
            self.output_shapes.append(shape)
            size = np.prod(shape) * np.dtype(np.float32).itemsize
            self.output_sizes.append(size)
        
        # Allocate GPU memory
        self.d_input = cuda.mem_alloc(self.input_size)
        self.d_outputs = [cuda.mem_alloc(size) for size in self.output_sizes]
        
        # Prepare bindings
        self.bindings = [int(self.d_input)] + [int(d_output) for d_output in self.d_outputs]
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for TensorRT model"""
        # Use same preprocessing as PyTorch version
        preprocessed = imagenet_preprocess(frame, self.INPUT_SIZE)
        # Add batch dimension and ensure float32 contiguous
        preprocessed = np.expand_dims(preprocessed, axis=0).astype(np.float32)
        return np.ascontiguousarray(preprocessed)
    
    def _postprocess(self, raw_outputs) -> Tuple[np.ndarray, float, Dict]:
        """Process TensorRT model output"""
        segmentation = raw_outputs[0]  # Lane segmentation
        
        # Reshape output
        seg_shape = self.output_shapes[0]
        segmentation = segmentation.reshape(seg_shape)[0]  # Remove batch dim
        
        # Convert to binary mask (argmax across classes)
        binary_mask = np.argmax(segmentation, axis=0).astype(np.uint8)
        
        # Calculate confidence
        confidence = float(np.max(np.max(segmentation, axis=(1, 2))))
        
        # Lane existence confidence if available
        if len(raw_outputs) > 1:
            lane_exist = raw_outputs[1].reshape(self.output_shapes[1])
            confidence = max(confidence, float(np.max(lane_exist)))
        
        metadata = {'model_type': 'tensorrt', 'num_classes': segmentation.shape[0]}
        
        return binary_mask, confidence, metadata
    
    def infer(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Dict]:
        """Run inference on a single frame"""
        input_tensor = self._preprocess(frame)
        
        # Copy input to GPU
        cuda.memcpy_htod(self.d_input, input_tensor)
        
        # Run TensorRT inference
        self.context.execute_v2(self.bindings)
        
        # Copy outputs from GPU
        outputs = []
        for i, (d_output, size, shape) in enumerate(zip(self.d_outputs, self.output_sizes, self.output_shapes)):
            output = np.empty(np.prod(shape), dtype=np.float32)
            cuda.memcpy_dtoh(output, d_output)
            outputs.append(output)
        
        return self._postprocess(outputs)
    
    def release(self):
        """Release GPU memory"""
        if hasattr(self, 'd_input'):
            self.d_input.free()
        if hasattr(self, 'd_outputs'):
            for d_output in self.d_outputs:
                d_output.free()
        super().release()