# Document Classifier and Extractor

A comprehensive AI-powered document processing system that classifies documents (invoices, payslips, resumes, certificates) and extracts key information using deep learning models.

## 🏗️ Project Structure

```
DOCUMENT_CLASSIFIER_AND_EXTRACTOR/
│
├── .venv/                                    # Python virtual environment
├── app/                                      # Main application package
│   ├── __pycache__/                         # Python cache files
│   ├── api/                                 # API-related modules
│   │   ├── __pycache__/                     # Python cache files
│   │   ├── routes/                          # API route definitions
│   │   └── main.py                          # FastAPI main application
│   ├── models/                              # Machine learning models
│   ├── utils/                               # Utility functions
│   └── __init__.py                          # Package initialization
├── data/                                    # Data storage directory
├── document_classifier_env/                # Alternative environment (if any)
├── Images/                                  # Input images/PDFs for processing
├── logs/                                    # Application logs
│
├── analyze_checkpoint.py                   # Model checkpoint analysis utility
├── convert_pdfs_updated.py                # Enhanced PDF to image converter
├── convert_pdfs.py                        # Basic PDF to image converter
├── debug_extraction.py                    # Field extraction debugging tool
├── debug_pipeline.py                      # Pipeline debugging utility
├── document_pipeline.py                   # Main document processing pipeline
├── document_routes.py                     # Alternative API routes (legacy)
├── examine_excel.py                       # Excel file analysis utility
├── extraction_model.py                    # Neural model for field extraction
├── ground_truth_classification.csv        # Training labels for classification
├── ground_truth_from_pdf.xlsx            # Ground truth data from PDFs
├── main.py                                # FastAPI application entry point
├── prepare_extraction_data.py            # Data preparation for extraction models
├── requirements.txt                       # Python dependencies
├── run.py                                 # Alternative application runner
├── simple_test.py                         # Simple testing script
├── test_api.sh                           # API testing shell script
├── test_classification.py                # Classification model testing
├── test_exact_match.py                   # Architecture matching tests
├── test_model_loading.py                 # Model loading verification
├── test_with_lower_threshold.py          # Threshold testing script
├── train_cnn.py                          # CNN model training script
├── train_extraction_model.py             # Extraction model training
└── train_vit.py                          # Vision Transformer training
```

## 🎯 Key Features

### Document Classification
- **CNN-based Classification**: Custom CNN architecture for document type classification
- **Vision Transformer (ViT)**: Alternative ViT implementation for improved accuracy
- **Supported Document Types**: Invoice, Payslip, Resume, Certificate
- **High Accuracy**: Advanced preprocessing and quality assessment

### Field Extraction
- **Document-Specific Extraction**: Tailored models for each document type
- **Key Fields**:
  - **Invoice**: Company name, invoice number, date, amount
  - **Payslip**: Employee name, employee ID, bank, amount
  - **Certificate**: Name, course name, course provider, date
  - **Resume**: Name, education, university, date

### API Service
- **FastAPI Backend**: RESTful API for document processing
- **Health Monitoring**: Built-in health checks and logging
- **File Upload Support**: PDF and image file processing
- **Quality Assessment**: Automatic image quality validation

## 📋 File Descriptions

### Core Components

#### `document_pipeline.py` 🔧
**Main processing pipeline** - The heart of the application
- **PDF to Image Conversion**: Converts PDF documents to images for processing
- **Quality Assessment**: Evaluates image quality (blur, contrast, resolution)
- **Document Classification**: Uses trained CNN to classify document types
- **Field Extraction**: Extracts specific fields based on document type
- **Error Handling**: Comprehensive error handling and logging

**Usage:**
```bash
python document_pipeline.py path/to/document.pdf
```

#### `main.py` 🌐
**FastAPI application** - Web API service
- **REST API Endpoints**: `/health`, `/classify-and-extract`
- **File Upload Handling**: Supports PDF, JPEG, PNG formats
- **CORS Support**: Cross-origin resource sharing enabled
- **Response Formatting**: Clean, formatted JSON responses

#### `extraction_model.py` 🧠
**Neural field extraction model**
- **Vision Transformer Architecture**: ViT-based feature extraction
- **Multi-head Architecture**: Separate heads for each field type
- **Character-level Decoding**: Converts neural outputs to text
- **Confidence Scoring**: Provides confidence scores for extractions

### Training Scripts

#### `train_cnn.py` 🏋️
**CNN model training**
- **Custom CNN Architecture**: Residual blocks and adaptive pooling
- **Data Augmentation**: Advanced augmentation pipeline
- **Training Loop**: Complete training with validation
- **Metrics**: Classification report and confusion matrix

#### `train_extraction_model.py` 🎯
**Field extraction training**
- **Category-specific Training**: Trains models for each document type
- **Character-level Loss**: CrossEntropyLoss for character prediction
- **Early Stopping**: Prevents overfitting
- **Model Checkpointing**: Saves best performing models

#### `train_vit.py` 🔬
**Vision Transformer training**
- **ViT Architecture**: Multi-head self-attention mechanism
- **Patch Embedding**: Image to patch conversion
- **Transformer Blocks**: Deep transformer architecture
- **Cosine Annealing**: Learning rate scheduling

### Data Processing

#### `convert_pdfs_updated.py` 📄
**Enhanced PDF conversion**
- **Batch Processing**: Processes entire directories
- **High-resolution Conversion**: 300 DPI conversion
- **Ground Truth Generation**: Creates classification CSV
- **Error Handling**: Robust error handling for corrupted files

#### `prepare_extraction_data.py` 📊
**Extraction data preparation**
- **Excel Processing**: Reads ground truth from Excel files
- **Data Validation**: Validates column mappings
- **Train/Test Split**: 80/20 split for training/testing
- **JSON Export**: Exports structured data for training

### Testing and Debugging

#### `debug_pipeline.py` 🔍
**Pipeline debugging tool**
- **System Verification**: Checks Python version, dependencies
- **File Existence**: Verifies required files are present
- **Import Testing**: Tests all critical imports
- **Model Loading**: Verifies model loading functionality

#### `debug_extraction.py` 🔧
**Field extraction debugging**
- **Model Status**: Checks extraction model loading
- **Field Testing**: Tests field extraction on sample documents
- **Confidence Analysis**: Analyzes extraction confidence scores
- **Step-by-step Debugging**: Detailed debugging output

#### `test_classification.py` ✅
**Classification testing**
- **Class Order Testing**: Tests different class mappings
- **Confidence Analysis**: Analyzes prediction confidence
- **Ground Truth Comparison**: Compares with expected results

### API and Deployment

#### `run.py` 🚀
**Application runner**
- **Uvicorn Server**: ASGI server configuration
- **Development Mode**: Hot reload enabled
- **Port Configuration**: Configurable port settings

#### `test_api.sh` 🧪
**API testing script**
- **Health Check**: Tests health endpoint
- **File Upload**: Tests document processing endpoint
- **Error Handling**: Tests invalid file type handling

## 🛠️ Installation and Setup

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)

### 1. Clone Repository
```bash
git clone <repository-url>
cd DOCUMENT_CLASSIFIER_AND_EXTRACTOR
```

### 2. Create Virtual Environment
```bash
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python debug_pipeline.py
```

## 🚀 Running the Application

### Method 1: Direct Pipeline Execution
Process individual documents using the pipeline directly:

```bash
# Basic usage
python document_pipeline.py path/to/document.pdf

# With debug output
python document_pipeline.py path/to/invoice.pdf

# Example with sample file
python document_pipeline.py Images/sample_invoice.pdf
```

**Expected Output:**
```json
{
  "document_type": "Invoice",
  "classification_confidence": "95.67%",
  "extracted_fields": {
    "company_name": "ABC Corporation",
    "invoice_number": "INV-2024-001",
    "date": "2024-01-15",
    "amount": "$1,250.00"
  },
  "processing_time": "2.34s",
  "success": true
}
```

### Method 2: FastAPI Web Service

#### Start the API Server
```bash
# Method 1: Using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Method 2: Using the run script
python run.py

# Method 3: Using the main script
python main.py
```

#### Test the API
```bash
# Check health status
curl -X GET "http://localhost:8000/health"

# Process a document
curl -X POST "http://localhost:8000/classify-and-extract" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/document.pdf"
```

#### API Documentation
- **Interactive Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json

### Method 3: Using Test Scripts
```bash
# Test API endpoints
chmod +x test_api.sh
./test_api.sh

# Test classification with specific document
python test_classification.py path/to/document.pdf

# Debug extraction issues
python debug_extraction.py path/to/document.pdf
```

## 🧪 Model Training

### 1. Prepare Training Data

#### Convert PDFs to Images
```bash
python convert_pdfs_updated.py
```

#### Prepare Extraction Data
```bash
python prepare_extraction_data.py
```

### 2. Train Classification Model

#### CNN Training
```bash
python train_cnn.py
```

#### Vision Transformer Training
```bash
python train_vit.py
```

### 3. Train Extraction Models
```bash
python train_extraction_model.py
```

This will train extraction models for all document types (invoice, payslip, certificate, resume).

## 📊 Data Format

### Ground Truth Structure

#### Classification Data (`ground_truth_classification.csv`)
```csv
file_path,label
Images/invoices/invoice_001.png,invoice
Images/payslips/payslip_001.png,payslip
Images/resumes/resume_001.png,resume
Images/certificates/cert_001.png,certificate
```

#### Extraction Data (`ground_truth_from_pdf.xlsx`)
**Invoice Sheet:**
- Filename, Company Name, Invoice no., Date, Amount

**Payslip Sheet:**
- Filename, Employee Name, Employee ID, Bank, Amount

**Certificate Sheet:**
- Filename, Name, Course Name, Course By, Date

**Resume Sheet:**
- Filename, Name, Education, University, Date

## 🔧 Configuration

### Quality Thresholds
Adjust quality assessment thresholds in `document_pipeline.py`:
```python
self.quality_thresholds = {
    'min_blur': 5,        # Minimum blur score
    'min_contrast': 5,    # Minimum contrast score
    'min_resolution': 5   # Minimum resolution multiplier
}
```

### Model Paths
Models are stored in:
- **Classification**: `app/models/trained/best_cnn_model.pth`
- **Extraction**: `app/models/trained/best_{category}_extraction_model.pth`

### API Configuration
Modify API settings in `main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Model Loading Errors
```bash
# Debug model loading
python test_model_loading.py

# Check model architecture
python test_exact_match.py
```

#### 2. Extraction Not Working
```bash
# Debug extraction pipeline
python debug_extraction.py

# Test with lower confidence threshold
python test_with_lower_threshold.py
```

#### 3. API Not Starting
```bash
# Check dependencies
python debug_pipeline.py

# Verify port availability
netstat -an | grep 8000
```

#### 4. Poor Extraction Results
- Check image quality using quality assessment
- Verify ground truth data format
- Retrain extraction models with more data
- Adjust confidence thresholds

### Debug Commands
```bash
# Full pipeline debug
python debug_pipeline.py

# Extraction-specific debug
python debug_extraction.py path/to/document.pdf

# Classification debug
python test_classification.py path/to/document.pdf
```

## 📈 Performance Optimization

### GPU Acceleration
The system automatically detects and uses CUDA if available:
```python
self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Batch Processing
For processing multiple documents:
```python
# Process multiple files
for pdf_file in pdf_files:
    result = pipeline.process_document(pdf_file)
    print(f"Processed {pdf_file}: {result}")
```

### Memory Management
- Models are loaded once during initialization
- Images are processed and released immediately
- Temporary files are automatically cleaned up

## 🔒 Error Handling

The system includes comprehensive error handling:

### Quality Assessment Errors
```python
try:
    quality_metrics = self.assess_image_quality(image)
except DocumentQualityError as e:
    return {"error": "Image quality check failed", "message": str(e)}
```

### Processing Errors
- File not found
- PDF conversion failures
- Model loading errors
- Extraction failures

All errors return structured JSON responses with appropriate HTTP status codes.

## 📝 Logging

The system includes detailed logging:
- **Processing Time**: Tracks time for each operation
- **Confidence Scores**: Logs prediction confidence
- **Quality Metrics**: Records image quality assessments
- **Error Details**: Comprehensive error logging

Logs are stored in the `logs/` directory and can be monitored for system performance and debugging.

## 🤝 Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly using the debug scripts
5. Submit a pull request

### Testing Checklist
- [ ] Run `python debug_pipeline.py`
- [ ] Test classification with `python test_classification.py`
- [ ] Test extraction with `python debug_extraction.py`
- [ ] Verify API with `./test_api.sh`
- [ ] Check code quality and documentation

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🆘 Support

For support and questions:
1. Check the troubleshooting section
2. Run the appropriate debug scripts
3. Review the logs in the `logs/` directory
4. Create an issue with detailed error information

---

**Note**: This system requires trained models to function properly. Ensure you have either trained the models using the provided training scripts or have pre-trained model files in the correct locations before running the application.