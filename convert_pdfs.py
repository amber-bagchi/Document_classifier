import os
from PIL import Image
import fitz  # PyMuPDF
from tqdm import tqdm

def convert_pdf_to_png(pdf_path, output_path):
    """Convert first page of PDF to PNG"""
    try:
        # Open PDF
        doc = fitz.open(pdf_path)
        # Get first page
        page = doc[0]
        # Convert to image
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI
        # Save image
        pix.save(output_path)
        doc.close()
        return True
    except Exception as e:
        print(f"Error converting {pdf_path}: {str(e)}")
        return False

def process_directory(input_dir):
    """Process all PDFs in directory and its subdirectories"""
    for root, _, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Processing {os.path.basename(root)}"):
            if file.endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                png_path = pdf_path.replace('.pdf', '.png')
                convert_pdf_to_png(pdf_path, png_path)

if __name__ == "__main__":
    # Convert all PDFs in the Images directory
    print("Converting PDFs to PNGs...")
    process_directory('Images')
    print("Conversion completed!")