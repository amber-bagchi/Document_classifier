#!/usr/bin/env python3
"""
Test the exact architecture match
"""
import torch
from app.models.document_cnn import DocumentCNN

def test_exact_architecture():
    """Test if the architecture now matches exactly"""
    
    # Load checkpoint
    checkpoint_path = 'app/models/trained/best_cnn_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        checkpoint_state = checkpoint['model_state_dict']
    else:
        checkpoint_state = checkpoint
    
    # Create model
    model = DocumentCNN(num_classes=4)
    model_state = model.state_dict()
    
    print("=== Exact Architecture Test ===")
    print(f"Checkpoint parameters: {len(checkpoint_state)}")
    print(f"Model parameters: {len(model_state)}")
    
    # Check if all checkpoint keys exist in model
    missing_in_model = []
    extra_in_model = []
    shape_mismatches = []
    
    # Check checkpoint keys in model
    for name in checkpoint_state.keys():
        if name not in model_state:
            missing_in_model.append(name)
        else:
            checkpoint_shape = checkpoint_state[name].shape if hasattr(checkpoint_state[name], 'shape') else None
            model_shape = model_state[name].shape if hasattr(model_state[name], 'shape') else None
            
            if checkpoint_shape != model_shape:
                shape_mismatches.append((name, checkpoint_shape, model_shape))
    
    # Check model keys not in checkpoint
    for name in model_state.keys():
        if name not in checkpoint_state:
            extra_in_model.append(name)
    
    print(f"\nMissing in model: {len(missing_in_model)}")
    if missing_in_model:
        for name in missing_in_model:
            print(f"  - {name}")
    
    print(f"\nExtra in model: {len(extra_in_model)}")
    if extra_in_model:
        for name in extra_in_model:
            print(f"  - {name}")
    
    print(f"\nShape mismatches: {len(shape_mismatches)}")
    if shape_mismatches:
        for name, checkpoint_shape, model_shape in shape_mismatches:
            print(f"  - {name}: checkpoint={checkpoint_shape}, model={model_shape}")
    
    # Test loading
    print(f"\n=== Loading Test ===")
    try:
        if len(missing_in_model) == 0 and len(shape_mismatches) == 0:
            model.load_state_dict(checkpoint_state, strict=True)
            print(" Perfect match! Loaded with strict=True")
        else:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint_state, strict=False)
            print(f" Loaded with strict=False")
            print(f"Missing keys: {len(missing_keys)}")
            print(f"Unexpected keys: {len(unexpected_keys)}")
            
            if unexpected_keys:
                print("Unexpected keys:")
                for key in unexpected_keys[:5]:  # Show first 5
                    print(f"  - {key}")
                if len(unexpected_keys) > 5:
                    print(f"  ... and {len(unexpected_keys) - 5} more")
        
        # Test forward pass
        model.eval()
        test_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(test_input)
            probabilities = torch.softmax(output, dim=1)
        
        print(f" Forward pass successful!")
        print(f"Output shape: {output.shape}")
        print(f"Sample probabilities: {probabilities[0].cpu().numpy()}")
        
        categories = ['invoice', 'payslip', 'resume', 'certificate']
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        print(f"Test prediction: {categories[predicted_class]} (confidence: {confidence:.4f})")
        
        return True
        
    except Exception as e:
        print(f" Loading failed: {e}")
        return False

if __name__ == "__main__":
    success = test_exact_architecture()
    
    if success:
        print(f"\n Architecture test passed! The model should work perfectly now.")
    else:
        print(f"\n  Architecture test failed. Check the errors above.")
