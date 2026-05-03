#!/usr/bin/env python3
"""
Convert ONNX model to TensorRT Engine for maximum GPU performance
"""
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import argparse
import os
from pathlib import Path

class ONNXToTensorRTConverter:
    def __init__(self):
        self.logger = trt.Logger(trt.Logger.WARNING)
        
    def build_engine(
        self,
        onnx_file_path: str,
        engine_file_path: str,
        precision: str = 'fp32',
        max_batch_size: int = 1,
        max_workspace_size: int = 1 << 30,  # 1GB
        input_shape: tuple = (3, 288, 800)
    ):
        """Build TensorRT engine from ONNX model"""
        
        print(f"Building TensorRT engine from: {onnx_file_path}")
        print(f"Precision: {precision}")
        print(f"Max batch size: {max_batch_size}")
        
        # Create builder and network
        builder = trt.Builder(self.logger)
        config = builder.create_builder_config()
        
        # Set precision
        if precision == 'fp16':
            if not builder.platform_has_fast_fp16:
                print("⚠️  FP16 not supported on this platform, falling back to FP32")
            else:
                config.set_flag(trt.BuilderFlag.FP16)
                print("✅ FP16 precision enabled")
        elif precision == 'int8':
            if not builder.platform_has_fast_int8:
                print("⚠️  INT8 not supported on this platform, falling back to FP32")
            else:
                config.set_flag(trt.BuilderFlag.INT8)
                print("✅ INT8 precision enabled (requires calibration)")
        
        # Set workspace size
        config.max_workspace_size = max_workspace_size
        
        # Parse ONNX model
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)
        
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("❌ Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    print(f"Error {error}: {parser.get_error(error)}")
                return None
        
        print("✅ ONNX model parsed successfully")
        
        # Configure input shapes for optimization
        input_tensor = network.get_input(0)
        print(f"Input tensor name: {input_tensor.name}")
        print(f"Input tensor shape: {input_tensor.shape}")
        
        # Create optimization profile
        profile = builder.create_optimization_profile()
        
        # Set input shape ranges (min, opt, max)
        input_name = input_tensor.name
        min_shape = (1,) + input_shape
        opt_shape = (max_batch_size,) + input_shape
        max_shape = (max_batch_size,) + input_shape
        
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        
        print(f"Input shape profile: min={min_shape}, opt={opt_shape}, max={max_shape}")
        
        # Build engine
        print("🔄 Building TensorRT engine... (this may take a few minutes)")
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            print("❌ Failed to build TensorRT engine")
            return None
        
        # Save engine to file
        with open(engine_file_path, 'wb') as f:
            f.write(serialized_engine)
        
        print(f"✅ TensorRT engine saved to: {engine_file_path}")
        return serialized_engine
    
    def test_engine(self, engine_file_path: str, input_shape: tuple = (1, 3, 288, 800)):
        """Test the built engine with dummy data"""
        print(f"\n🧪 Testing engine: {engine_file_path}")
        
        # Load engine
        with open(engine_file_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(self.logger)
        engine = runtime.deserialize_cuda_engine(engine_data)
        
        if engine is None:
            print("❌ Failed to load engine")
            return False
        
        # Create execution context
        context = engine.create_execution_context()
        
        # Set input shape
        context.set_binding_shape(0, input_shape)
        
        # Allocate memory
        input_size = np.prod(input_shape) * np.dtype(np.float32).itemsize
        output_shapes = []
        output_sizes = []
        
        for i in range(1, engine.num_bindings):
            shape = context.get_binding_shape(i)
            output_shapes.append(shape)
            size = np.prod(shape) * np.dtype(np.float32).itemsize
            output_sizes.append(size)
        
        # Allocate GPU memory
        d_input = cuda.mem_alloc(input_size)
        d_outputs = [cuda.mem_alloc(size) for size in output_sizes]
        
        # Create dummy input
        dummy_input = np.random.rand(*input_shape).astype(np.float32)
        
        # Copy input to GPU
        cuda.memcpy_htod(d_input, dummy_input)
        
        # Prepare bindings
        bindings = [int(d_input)] + [int(d_output) for d_output in d_outputs]
        
        # Run inference
        context.execute_v2(bindings)
        
        print(f"✅ Engine test successful!")
        print(f"   Input shape: {input_shape}")
        print(f"   Output shapes: {output_shapes}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Convert ONNX to TensorRT Engine')
    parser.add_argument('--input', '-i', required=True, help='Input .onnx model path')
    parser.add_argument('--output', '-o', help='Output .engine path (auto-generated if not specified)')
    parser.add_argument('--precision', default='fp32', choices=['fp32', 'fp16', 'int8'], 
                      help='Engine precision')
    parser.add_argument('--batch-size', type=int, default=1, help='Maximum batch size')
    parser.add_argument('--workspace', type=int, default=1024, help='Max workspace size in MB')
    parser.add_argument('--height', type=int, default=288, help='Input height')
    parser.add_argument('--width', type=int, default=800, help='Input width')
    parser.add_argument('--test', action='store_true', help='Test the engine after conversion')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return
    
    if args.output is None:
        # Auto-generate output path
        input_path = Path(args.input)
        suffix = f"_{args.precision}_bs{args.batch_size}.engine"
        args.output = str(input_path.with_suffix(suffix))
    
    print(f"Converting {args.input} → {args.output}")
    
    try:
        converter = ONNXToTensorRTConverter()
        
        engine = converter.build_engine(
            onnx_file_path=args.input,
            engine_file_path=args.output,
            precision=args.precision,
            max_batch_size=args.batch_size,
            max_workspace_size=args.workspace * 1024 * 1024,  # Convert MB to bytes
            input_shape=(3, args.height, args.width)
        )
        
        if engine and args.test:
            converter.test_engine(args.output, (args.batch_size, 3, args.height, args.width))
            
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install TensorRT: pip install tensorrt pycuda")
    except Exception as e:
        print(f"❌ Conversion failed: {e}")

if __name__ == '__main__':
    main()