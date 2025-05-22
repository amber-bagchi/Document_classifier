#!/usr/bin/env python3
"""
Test the pipeline with lower confidence threshold
"""
import sys
from document_pipeline import DocumentPipeline

def test_extraction_with_lower_threshold(pdf_path):
    """Test extraction with lower confidence threshold"""
    
    print("🧪 Testing extraction with lower confidence threshold...")
    
    # Initialize pipeline
    pipeline = DocumentPipeline(debug=True)
    
    # Process document
    result = pipeline.process_document(pdf_path)
    
    print(f"\n{'='*60}")
    print("📊 RESULTS WITH LOWER THRESHOLD")
    print(f"{'='*60}")
    
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # Analyze the results
    if result.get('success', False):
        extracted_fields = result.get('extracted_fields', {})
        if extracted_fields:
            print(f"\n✅ SUCCESS: Extracted {len(extracted_fields)} fields!")
            for field, value in extracted_fields.items():
                print(f"  • {field}: '{value}'")
        else:
            print(f"\n⚠️  Still no fields extracted even with lower threshold")
    else:
        print(f"\n❌ Processing failed: {result.get('message', 'Unknown error')}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_with_lower_threshold.py <pdf_path>")
        print("Example: python test_with_lower_threshold.py resume.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    test_extraction_with_lower_threshold(pdf_path)