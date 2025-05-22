#!/usr/bin/env python3

import sys
import os
from document_pipeline import DocumentPipeline

def main():
    print("=== Simple Pipeline Test ===")
    
    try:
        # Create pipeline
        print("Creating DocumentPipeline...")
        pipeline = DocumentPipeline(debug=True)
        print("✓ Pipeline created successfully!")
        
        # Test with a PDF file if provided
        if len(sys.argv) > 1:
            pdf_path = sys.argv[1]
            if os.path.exists(pdf_path):
                print(f"\nProcessing document: {pdf_path}")
                result = pipeline.process_document(pdf_path)
                print("\nResult:")
                print(result)
            else:
                print(f"File not found: {pdf_path}")
        else:
            print("\nTo test with a document, run: python simple_test.py <path_to_pdf>")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
