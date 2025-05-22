#!/usr/bin/env python3

import sys
import os
from document_pipeline import DocumentPipeline
import torch

def test_different_class_orders():
    """Test different class order mappings to find the correct one"""
    
    # Different possible class orders
    possible_orders = [
        ['invoice', 'payslip', 'resume', 'certificate'],
        ['certificate', 'invoice', 'payslip', 'resume'],
        ['certificate', 'payslip', 'invoice', 'resume'],
        ['invoice', 'certificate', 'payslip', 'resume'],
        ['payslip', 'invoice', 'certificate', 'resume'],
        ['resume', 'payslip', 'invoice', 'certificate']
    ]
    
    if len(sys.argv) < 2:
        print("Usage: python test_classification.py <path_to_test_document>")
        print("Please provide a test document (preferably one you know the correct type)")
        return
    
    pdf_path = sys.argv[1]
    expected_type = input(f"What type of document is '{os.path.basename(pdf_path)}'? (invoice/payslip/resume/certificate): ").strip().lower()
    
    print(f"\nTesting classification for: {os.path.basename(pdf_path)}")
    print(f"Expected type: {expected_type}")
    print("=" * 50)
    
    # Create pipeline
    pipeline = DocumentPipeline(debug=False)
    
    # Convert PDF and get image
    image = pipeline.convert_pdf_to_image(pdf_path)
    if image is None:
        print("Failed to convert PDF")
        return
    
    # Get raw model output
    image_tensor = pipeline.transform(image).unsqueeze(0).to(pipeline.device)
    with torch.no_grad():
        outputs = pipeline.classifier(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)
    
    print(f"Raw model output: {outputs[0].tolist()}")
    print(f"Probabilities: {probabilities.tolist()}")
    print(f"Predicted index: {predicted_idx.item()}")
    print(f"Confidence: {confidence.item():.4f}")
    print()
    
    # Test different class orders
    for i, order in enumerate(possible_orders):
        predicted_type = order[predicted_idx.item()]
        is_correct = "✓" if predicted_type == expected_type else "✗"
        print(f"{i+1}. {order} -> {predicted_type} {is_correct}")
    
    print(f"\nIf the correct mapping is found, update the 'categories' list in classify_document method")

if __name__ == "__main__":
    test_different_class_orders()
