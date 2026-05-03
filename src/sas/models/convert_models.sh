#!/bin/bash
# Convert PyTorch models to optimized formats for better performance

set -e  # Exit on any error

MODEL_NAME="ERFNet_trained"
CHECKPOINT_DIR="checkpoints"
PT_FILE="$CHECKPOINT_DIR/${MODEL_NAME}.pt"
ONNX_FILE="$CHECKPOINT_DIR/${MODEL_NAME}.onnx"
ENGINE_FILE="$CHECKPOINT_DIR/${MODEL_NAME}_fp16_bs1.engine"

echo "🚀 Swervin Model Conversion Pipeline"
echo "======================================"

# Check if PyTorch model exists
if [ ! -f "$PT_FILE" ]; then
    echo "❌ PyTorch model not found: $PT_FILE"
    echo "Please ensure your trained model is in the checkpoints directory"
    exit 1
fi

echo "📁 Found PyTorch model: $PT_FILE"

# Step 1: Convert PT to ONNX
echo ""
echo "🔄 Step 1: Converting PyTorch → ONNX..."
python convert_to_onnx.py --input "$PT_FILE" --output "$ONNX_FILE" --device cpu

if [ $? -eq 0 ]; then
    echo "✅ ONNX conversion successful: $ONNX_FILE"
else
    echo "❌ ONNX conversion failed"
    exit 1
fi

# Step 2: Convert ONNX to TensorRT (optional, requires NVIDIA GPU)
echo ""
echo "🔄 Step 2: Converting ONNX → TensorRT Engine..."
echo "⚠️  This requires NVIDIA GPU and TensorRT installation"

read -p "Do you want to convert to TensorRT? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python convert_to_tensorrt.py \
        --input "$ONNX_FILE" \
        --output "$ENGINE_FILE" \
        --precision fp16 \
        --batch-size 1 \
        --test
    
    if [ $? -eq 0 ]; then
        echo "✅ TensorRT conversion successful: $ENGINE_FILE"
    else
        echo "❌ TensorRT conversion failed (GPU/TensorRT may not be available)"
    fi
else
    echo "⏭️  Skipping TensorRT conversion"
fi

# Display results
echo ""
echo "🎉 Conversion Complete!"
echo "======================"
echo "Available models:"
echo "  PyTorch:   $PT_FILE    (baseline performance)"
echo "  ONNX:      $ONNX_FILE  (2-3x faster)"
if [ -f "$ENGINE_FILE" ]; then
    echo "  TensorRT:  $ENGINE_FILE    (5-10x faster on GPU)"
fi

echo ""
echo "📋 To use optimized models, update your config.toml:"
echo ""
echo "For ONNX (recommended):"
echo "[lane_segmenter]"
echo "name = 'ONNXERFNet'"
echo "model_path = '${MODEL_NAME}.onnx'"
echo "dataset = 'culane'"
echo "device = 'cpu'  # or 'cuda' for GPU"
echo ""
echo "For TensorRT (maximum performance):"
echo "[lane_segmenter]"
echo "name = 'TensorRTERFNet'"
echo "model_path = '${MODEL_NAME}_fp16_bs1.engine'"
echo "dataset = 'culane'"
echo "device = 'cuda'"