#!/usr/bin/env python3
"""
Debug field extraction issues
"""
import os
import torch
from document_pipeline import DocumentPipeline

def debug_extraction_models():
    """Debug extraction model loading"""
    print("🔍 Debugging Extraction Models...\n")
    
    # Initialize pipeline
    pipeline = DocumentPipeline(debug=True)
    
    print(f"\n{'='*50}")
    print("EXTRACTION MODELS STATUS")
    print(f"{'='*50}")
    
    # Check extraction models
    print(f"Extraction models loaded: {len(pipeline.extraction_models)}")
    
    if not pipeline.extraction_models:
        print("❌ No extraction models loaded!")
        
        # Check if extraction model files exist
        categories = ['invoice', 'payslip', 'resume', 'certificate']
        for category in categories:
            model_path = f'app/models/trained/best_{category}_extraction_model.pth'
            exists = os.path.exists(model_path)
            print(f"  {category}: {model_path} - {'✅ EXISTS' if exists else '❌ MISSING'}")
        
        return False
    
    # Test each extraction model
    for doc_type, model_info in pipeline.extraction_models.items():
        print(f"\n📋 Testing {doc_type.upper()} extraction model:")
        print(f"  Fields: {model_info['fields']}")
        
        try:
            # Test with dummy input
            test_input = torch.randn(1, 3, 224, 224).to(pipeline.device)
            model = model_info['model']
            
            print(f"  Model type: {type(model)}")
            
            with torch.no_grad():
                outputs = model(test_input)
                print(f"  ✅ Forward pass successful")
                print(f"  Output type: {type(outputs)}")
                
                # Handle different output types
                if isinstance(outputs, dict):
                    print(f"  Output is dictionary with keys: {list(outputs.keys())}")
                    for key, value in outputs.items():
                        if hasattr(value, 'shape'):
                            print(f"    {key}: shape {value.shape}")
                        else:
                            print(f"    {key}: type {type(value)}, value {value}")
                elif hasattr(outputs, 'shape'):
                    print(f"  Output shape: {outputs.shape}")
                else:
                    print(f"  Output: {outputs}")
                
                # Test field extraction
                try:
                    print(f"  Testing extract_fields method...")
                    
                    # Check if model has extract_fields method
                    if hasattr(model, 'extract_fields'):
                        print(f"  ✅ Model has extract_fields method")
                        extracted = model.extract_fields(outputs, debug=True)
                        print(f"  ✅ Field extraction successful")
                        print(f"  Extracted fields type: {type(extracted)}")
                        print(f"  Extracted fields: {extracted}")
                        
                        if isinstance(extracted, dict):
                            for field, data in extracted.items():
                                print(f"    Field '{field}':")
                                print(f"      Type: {type(data)}")
                                print(f"      Content: {data}")
                                
                                if isinstance(data, dict):
                                    if 'confidence' in data and 'values' in data:
                                        conf = data.get('confidence', [0])
                                        val = data.get('values', [''])
                                        print(f"      Confidence: {conf}")
                                        print(f"      Values: {val}")
                                        
                                        if len(conf) > 0 and len(val) > 0:
                                            print(f"      First confidence: {conf[0]}")
                                            print(f"      First value: '{val[0]}'")
                    else:
                        print(f"  ❌ Model does not have extract_fields method")
                        print(f"  Available methods: {[method for method in dir(model) if not method.startswith('_')]}")
                
                except Exception as e:
                    print(f"  ❌ Field extraction failed: {e}")
                    import traceback
                    traceback.print_exc()
                    
        except Exception as e:
            print(f"  ❌ Model test failed: {e}")
            import traceback
            traceback.print_exc()
    
    return True

def debug_specific_extraction(pdf_path):
    """Debug extraction for a specific document"""
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    print(f"\n🔍 Debugging extraction for: {pdf_path}")
    
    pipeline = DocumentPipeline(debug=True)
    
    # Convert PDF to image
    image = pipeline.convert_pdf_to_image(pdf_path)
    if image is None:
        print("❌ Failed to convert PDF to image")
        return
    
    # Classify document
    doc_type, confidence = pipeline.classify_document(image)
    print(f"\n📑 Classified as: {doc_type} (confidence: {confidence:.2%})")
    
    # Debug extraction step by step
    print(f"\n🔍 Debugging field extraction for {doc_type}:")
    
    if doc_type not in pipeline.extraction_models:
        print(f"❌ No extraction model for {doc_type}")
        print(f"Available models: {list(pipeline.extraction_models.keys())}")
        return
    
    print(f"✅ Extraction model found for {doc_type}")
    
    # Test extraction with detailed debugging
    try:
        image_tensor = pipeline.transform(image).unsqueeze(0).to(pipeline.device)
        model_info = pipeline.extraction_models[doc_type]
        model = model_info['model']
        
        print(f"Model fields: {model_info['fields']}")
        
        with torch.no_grad():
            outputs = model(image_tensor)
            print(f"Model outputs shape: {outputs.shape}")
            
            # Extract fields with debugging
            extracted = model.extract_fields(outputs, debug=True)
            print(f"Raw extraction result: {extracted}")
            
            # Apply the same filtering as the pipeline
            results = {}
            for field, data in extracted.items():
                if isinstance(data, dict) and 'values' in data and 'confidence' in data:
                    if len(data['values']) > 0 and len(data['confidence']) > 0:
                        value = data['values'][0]
                        confidence = float(data['confidence'][0])
                        
                        print(f"Field {field}: value='{value}', confidence={confidence:.4f}")
                        
                        if confidence > 0.4:  # Same threshold as pipeline
                            results[field] = str(value)
                            print(f"  ✅ Added to results (confidence > 0.4)")
                        else:
                            print(f"  ❌ Skipped (confidence <= 0.4)")
            
            print(f"\nFinal extracted fields: {results}")
            
    except Exception as e:
        print(f"❌ Extraction debugging failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main debugging function"""
    import sys
    
    print("🔧 Field Extraction Debugger\n")
    
    # Debug extraction models
    models_ok = debug_extraction_models()
    
    if not models_ok:
        print("\n💥 Extraction models are not loading properly!")
        print("Make sure the extraction model files exist in app/models/trained/")
        return
    
    # If PDF path provided, debug specific document
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        debug_specific_extraction(pdf_path)
    else:
        print("\n📖 Usage:")
        print("  python debug_extraction.py                    # Debug extraction models")
        print("  python debug_extraction.py path/to/file.pdf   # Debug specific document")

if __name__ == "__main__":
    main()