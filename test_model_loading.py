#!/usr/bin/env python3

import sys
import os
import torch

print("=== Testing Model Loading ===")

# Add the models directory to path
sys.path.insert(0, 'app/models')

try:
    from document_cnn import DocumentCNN
    print("✓ DocumentCNN imported successfully")
    
    # Create model
    model = DocumentCNN(num_classes=4)
    print("✓ Model created successfully")
    
    # Check if checkpoint exists
    checkpoint_path = 'app/models/trained/best_cnn_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"✗ Checkpoint not found at: {checkpoint_path}")
        # Try alternative paths
        alt_paths = [
            'saved_models/best_cnn_model.pth',
            'models/best_cnn_model.pth',
            'best_cnn_model.pth'
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                print(f"✓ Found checkpoint at: {alt_path}")
                checkpoint_path = alt_path
                break
        else:
            print("✗ No checkpoint found in any location")
            sys.exit(1)
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print("✓ Checkpoint loaded successfully")
    
    # Check checkpoint keys
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("✓ Using 'state_dict' key from checkpoint")
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("✓ Using 'model_state_dict' key from checkpoint")
        else:
            state_dict = checkpoint
            print("✓ Using checkpoint directly as state_dict")
    else:
        state_dict = checkpoint
        print("✓ Using checkpoint directly as state_dict")
    
    print("\nCheckpoint keys:")
    for key in list(state_dict.keys())[:10]:  # Show first 10 keys
        print(f"  {key}: {state_dict[key].shape if hasattr(state_dict[key], 'shape') else type(state_dict[key])}")
    if len(state_dict.keys()) > 10:
        print(f"  ... and {len(state_dict.keys()) - 10} more keys")
    
    print("\nModel keys:")
    model_state = model.state_dict()
    for key in list(model_state.keys())[:10]:  # Show first 10 keys
        print(f"  {key}: {model_state[key].shape}")
    if len(model_state.keys()) > 10:
        print(f"  ... and {len(model_state.keys()) - 10} more keys")
    
    # Try to load state dict
    model.load_state_dict(state_dict)
    print("✓ Model state_dict loaded successfully!")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, 3, 224, 224)
        output = model(test_input)
        print(f"✓ Forward pass successful! Output shape: {output.shape}")
        print(f"  Output values: {output}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
