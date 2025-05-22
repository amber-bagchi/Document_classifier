#!/usr/bin/env python3

import sys
import os

print("=== Debug Document Pipeline ===")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path[0]}")

# Check if required files exist
required_files = [
    'document_pipeline.py',
    'app/models/document_cnn.py',
    'app/models/trained/best_cnn_model.pth'
]

print("\n=== Checking required files ===")
for file_path in required_files:
    if os.path.exists(file_path):
        print(f"✓ {file_path} exists")
    else:
        print(f"✗ {file_path} MISSING")

# Test basic imports
print("\n=== Testing imports ===")
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")

try:
    import torchvision
    print(f"✓ Torchvision version: {torchvision.__version__}")
except ImportError as e:
    print(f"✗ Torchvision import failed: {e}")

try:
    from PIL import Image
    print("✓ PIL imported successfully")
except ImportError as e:
    print(f"✗ PIL import failed: {e}")

try:
    import cv2
    print(f"✓ OpenCV version: {cv2.__version__}")
except ImportError as e:
    print(f"✗ OpenCV import failed: {e}")

try:
    import fitz
    print("✓ PyMuPDF imported successfully")
except ImportError as e:
    print(f"✗ PyMuPDF import failed: {e}")

# Test model import
print("\n=== Testing model import ===")
try:
    sys.path.insert(0, 'app/models')
    from document_cnn import DocumentCNN
    model = DocumentCNN()
    print(f"✓ DocumentCNN imported and instantiated successfully")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters())}")
except Exception as e:
    print(f"✗ DocumentCNN import/instantiation failed: {e}")

# Test pipeline import
print("\n=== Testing pipeline import ===")
try:
    from document_pipeline import DocumentPipeline
    print("✓ DocumentPipeline imported successfully")
except Exception as e:
    print(f"✗ DocumentPipeline import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Testing pipeline initialization ===")
try:
    # Try to create pipeline with debug=True
    print("Creating DocumentPipeline with debug=True...")
    pipeline = DocumentPipeline(debug=True)
    print("✓ DocumentPipeline created successfully")
except Exception as e:
    print(f"✗ DocumentPipeline creation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Debug complete ===")
