#!/usr/bin/env python3
"""
Convert PyTorch ERFNet model to ONNX format for optimized inference
"""
import torch
import torch.onnx
import os
import sys
import argparse
from pathlib import Path

# Add src to path so we can import models
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from sas.models.segmentation.autodrive_erfnet import ERFNet

def convert_pt_to_onnx(
    pt_model_path: str,
    onnx_output_path: str,
    input_size: tuple = (288, 800),
    dataset: str = 'culane',
    device: str = 'cpu'
):
    """Convert PyTorch model to ONNX format"""
    
    print(f"Loading PyTorch model from: {pt_model_path}")
    
    # Create model instance (same as in lane_segmenter.py)
    model = ERFNet(
        num_classes=5 if dataset == 'culane' else 7,
        dropout_1=0.1,
        dropout_2=0.1,
        pretrained_weights=None,
        lane_exist_cfg=dict(
            name='EDLaneExist',
            num_output=4,
            flattened_size=3965,
            # flattened_size=4500,  # Must match the trained model: 5 x 18 x 50 = 4500
            dropout=0.1,
            pool='max'
        )
    )
    
    # Load weights
    # erfnet cityscapes
    # checkpoint = torch.load(pt_model_path, map_location=device, weights_only=False)
    # state_dict = checkpoint['state_dict']
    # # Remove 'module.' prefix if present (from DataParallel)
    # state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    # model.load_state_dict(state_dict)
    
    # erfnet culane
    checkpoint = torch.load(pt_model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model') or checkpoint.get('state_dict')
    state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    state_dict = {k.replace('aux_head.', 'lane_exist.', 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    # Set to evaluation mode
    model.eval()
    model.to(device)
    
    # Create dummy input tensor
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, input_size[0], input_size[1]).to(device)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Converting to ONNX...")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['segmentation', 'lane_exist'],  # ERFNet has 2 outputs
        dynamic_axes={
            'input': {0: 'batch_size'},
            'segmentation': {0: 'batch_size'},
            'lane_exist': {0: 'batch_size'}
        }
    )
    
    print(f"✅ ONNX model saved to: {onnx_output_path}")
    
    # Verify the ONNX model
    import onnx
    onnx_model = onnx.load(onnx_output_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model verification passed")
    
    # Print model info
    print(f"Model input shape: {dummy_input.shape}")
    with torch.no_grad():
        output = model(dummy_input)
        for k, v in output.items():
            print(f"  {k}: {v.shape}")

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch ERFNet to ONNX')
    parser.add_argument('--input', '-i', required=True, help='Input .pt model path')
    parser.add_argument('--output', '-o', help='Output .onnx path (auto-generated if not specified)')
    parser.add_argument('--dataset', default='culane', choices=['culane', 'tusimple'], help='Dataset type')
    parser.add_argument('--device', default='cpu', help='Device to use for conversion')
    parser.add_argument('--height', type=int, default=288, help='Input height')
    parser.add_argument('--width', type=int, default=800, help='Input width')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return
    
    if args.output is None:
        # Auto-generate output path
        input_path = Path(args.input)
        args.output = str(input_path.with_suffix('.onnx'))
    
    print(f"Converting {args.input} → {args.output}")
    
    convert_pt_to_onnx(
        pt_model_path=args.input,
        onnx_output_path=args.output,
        input_size=(args.height, args.width),
        dataset=args.dataset,
        device=args.device
    )

if __name__ == '__main__':
    main()