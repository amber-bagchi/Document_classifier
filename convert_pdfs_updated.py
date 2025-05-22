import os
from PIL import Image
import fitz  # PyMuPDF
from tqdm import tqdm
import pandas as pd

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

def process_directory(input_dir, output_base_dir='converted_images'):
    """Process all PDFs in directory and its subdirectories"""
    for root, _, files in os.walk(input_dir):
        category = os.path.basename(root)  # e.g., 'generated_certificates'
        # Skip if not a category directory
        if not category.startswith('generated_'):
            continue
        
        # Get clean category name (remove 'generated_' prefix)
        clean_category = category.replace('generated_', '')
        output_dir = os.path.join(output_base_dir, clean_category)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        for file in tqdm(files, desc=f"Processing {category}"):
            if file.endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                png_path = os.path.join(output_dir, file.replace('.pdf', '.png'))
                convert_pdf_to_png(pdf_path, png_path)

def create_ground_truth_csv():
    """Create ground truth CSV file with image paths and labels"""
    import pandas as pd
    
    data = []
    base_dir = 'converted_images'
    
    for category in ['certificates', 'invoices', 'payslips', 'resumes']:
        category_dir = os.path.join(base_dir, category)
        if os.path.exists(category_dir):
            for file in os.listdir(category_dir):
                if file.endswith('.png'):
                    data.append({
                        'file_path': os.path.join(category_dir, file),
                        'label': category
                    })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv('ground_truth_classification.csv', index=False)
    print(f"Created ground truth CSV with {len(df)} entries")

if __name__ == "__main__":
    print("Converting PDFs to PNGs...")
    process_directory('Images')
    print("Creating ground truth CSV...")
    create_ground_truth_csv()
    print("Processing completed!")