#!/usr/bin/env python3
"""
Script to analyze all parameters in the checkpoint
"""
import torch

def analyze_checkpoint():
    """Analyze the checkpoint structure completely"""
    
    checkpoint_path = 'app/models/trained/best_cnn_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    print("=== Complete Checkpoint Analysis ===")
    print(f"Total parameters: {len(state_dict)}")
    
    # Print all parameters with their shapes
    print("\nAll parameters:")
    for i, (key, tensor) in enumerate(state_dict.items()):
        shape = tensor.shape if hasattr(tensor, 'shape') else 'N/A'
        print(f"{i+1:2d}. {key}: {shape}")
    
    # Group by layer structure
    print("\n=== Layer Structure Analysis ===")
    
    features_params = [k for k in state_dict.keys() if k.startswith('features.')]
    classifier_params = [k for k in state_dict.keys() if k.startswith('classifier.')]
    
    print(f"\nFeatures layers: {len(features_params)}")
    for param in features_params:
        shape = state_dict[param].shape if hasattr(state_dict[param], 'shape') else 'N/A'
        print(f"  {param}: {shape}")
    
    print(f"\nClassifier layers: {len(classifier_params)}")
    for param in classifier_params:
        shape = state_dict[param].shape if hasattr(state_dict[param], 'shape') else 'N/A'
        print(f"  {param}: {shape}")
    
    # Analyze the structure pattern
    print("\n=== Architecture Pattern ===")
    
    # Extract unique layer prefixes
    layer_prefixes = set()
    for key in features_params:
        # Extract pattern like "features.4", "features.5", etc.
        parts = key.split('.')
        if len(parts) >= 2:
            layer_prefixes.add(f"{parts[0]}.{parts[1]}")
    
    print("Feature layer groups:")
    for prefix in sorted(layer_prefixes):
        params_in_group = [k for k in features_params if k.startswith(prefix + '.')]
        print(f"  {prefix}: {len(params_in_group)} parameters")
        for param in params_in_group:
            shape = state_dict[param].shape if hasattr(state_dict[param], 'shape') else 'N/A'
            print(f"    {param}: {shape}")

if __name__ == "__main__":
    analyze_checkpoint()