import os
import json
import pandas as pd

def read_category_data(excel_file, sheet_name):
    """Read data for a specific category from Excel sheet"""
    try:
        # Read with different parameters to handle various Excel formats
        df = pd.read_excel(excel_file, sheet_name=sheet_name, header=0)
        
        # Clean column names - remove extra spaces and standardize
        df.columns = df.columns.astype(str).str.strip()
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Reset index after dropping rows
        df = df.reset_index(drop=True)
        
        return df
    except Exception as e:
        print(f"Error reading sheet '{sheet_name}': {str(e)}")
        raise

def inspect_excel_structure(excel_file):
    """Inspect the Excel file structure to understand available data"""
    try:
        excel_data = pd.ExcelFile(excel_file)
        
        for sheet_name in excel_data.sheet_names:
            print(f"\n--- Sheet: {sheet_name} ---")
            df = pd.read_excel(excel_file, sheet_name=sheet_name, header=0)
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Show first few rows
            print("Sample data:")
            print(df.head(2).to_string())
            
    except Exception as e:
        print(f"Error inspecting Excel file: {str(e)}")

def get_category_fields():
    """Define fields and column mappings for each category"""
    return {
        'invoice': {
            'sheet_name': 'Invoice',
            'fields': [
                'company_name',
                'invoice_number',
                'date',
                'amount'
            ],
            'column_map': {
                'company_name': 'Company Name',
                'invoice_number': 'Invoice no.',
                'date': 'Date',
                'amount': 'Amount'
            }
        },
        'payslip': {
            'sheet_name': 'Payslip',
            'fields': [
                'employee_name',
                'employee_id',
                'bank',
                'amount'
            ],
            'column_map': {
                'employee_name': 'Employee Name',
                'employee_id': 'Employee ID',
                'bank': 'Bank',
                'amount': 'Amount'
            }
        },
        'certificate': {
            'sheet_name': 'Certificate',
            'fields': [
                'name',
                'course_name',
                'course_by',
                'date'
            ],
            'column_map': {
                'name': 'Name',
                'course_name': 'Course Name',
                'course_by': 'Course By',
                'date': 'Date'
            }
        },
        'resume': {
            'sheet_name': 'Resume',
            'fields': [
                'name',
                'education',
                'university',
                'date'
            ],
            'column_map': {
                'name': 'Name',
                'education': 'Education',
                'university': 'University',
                'date': 'Date'
            }
        }
    }

def prepare_category_data(df, category_info):
    """Prepare data for a specific document category"""
    fields = category_info['fields']
    column_map = category_info['column_map']
    
    train_data = []
    test_data = []
    
    # Print available columns for debugging
    print(f"Available columns: {list(df.columns)}")
    
    # Use 80% for training, 20% for testing
    train_size = int(0.8 * len(df))
    
    for idx, row in df.iterrows():
        doc_data = {
            'file_name': row['Filename'],
            'fields': {}
        }
        
        # Extract fields using column mapping
        for field in fields:
            column = column_map[field]
            value = ""
            
            # Try exact column match first
            if column in df.columns:
                value = row[column]
            else:
                # Try case-insensitive match
                matching_cols = [col for col in df.columns if col.lower() == column.lower()]
                if matching_cols:
                    value = row[matching_cols[0]]
                else:
                    # Try partial match
                    partial_matches = [col for col in df.columns if column.lower() in col.lower() or col.lower() in column.lower()]
                    if partial_matches:
                        value = row[partial_matches[0]]
                        print(f"Using partial match '{partial_matches[0]}' for field '{field}'")
                    else:
                        print(f"Warning: No matching column found for '{column}' (field: {field})")
            
            # Convert to string and handle NaN/None
            if pd.isna(value) or value is None:
                doc_data['fields'][field] = ""
            else:
                # Clean the value - remove extra whitespace and newlines
                cleaned_value = str(value).strip().replace('\n', ' ')
                doc_data['fields'][field] = cleaned_value
        
        if idx < train_size:
            train_data.append(doc_data)
        else:
            test_data.append(doc_data)
    
    return {
        'fields': fields,
        'train_data': train_data,
        'test_data': test_data
    }

def main():
    excel_file = 'ground_truth_from_pdf.xlsx'
    
    # First, let's examine the Excel file structure
    print("Examining Excel file structure...")
    inspect_excel_structure(excel_file)
    
    # Get fields for each category
    category_info = get_category_fields()
    
    # Prepare data for each category
    extraction_data = {}
    for category, info in category_info.items():
        print(f"\n{'='*50}")
        print(f"Preparing data for {category.upper()}...")
        print(f"Expected sheet: {info['sheet_name']}")
        
        try:
            # Read category data
            df = read_category_data(excel_file, info['sheet_name'])
            print(f"Successfully read {len(df)} rows from '{info['sheet_name']}' sheet")
            
            # Show sample data for debugging
            print(f"Sample data (first 3 rows):")
            for i, (idx, row) in enumerate(df.head(3).iterrows()):
                print(f"  Row {i+1}: {dict(row)}")
            
            # Prepare data
            category_data = prepare_category_data(df, info)
            extraction_data[category] = category_data
            
            # Print statistics
            num_train = len(category_data['train_data'])
            num_test = len(category_data['test_data'])
            print(f"Created {num_train} training and {num_test} test samples")
            
            # Show sample extracted data
            if num_train > 0:
                print(f"Sample training data:")
                print(f"  {category_data['train_data'][0]}")
            
        except Exception as e:
            print(f"Error processing {category}: {str(e)}")
            continue
    
    # Save to JSON file
    output_dir = 'data/extraction'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'extraction_data.json')
    with open(output_file, 'w') as f:
        json.dump(extraction_data, f, indent=2)
    
    print(f"\nExtraction data saved to {output_file}")
    print(f"Total categories processed: {len(extraction_data)}")

if __name__ == '__main__':
    main()