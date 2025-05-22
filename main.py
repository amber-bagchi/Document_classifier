from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import os
import tempfile
from PIL import Image
import time
from datetime import datetime
from document_pipeline import DocumentPipeline

app = FastAPI(
    title="Document Processing API",
    description="API for document classification and information extraction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize document pipeline
pipeline = DocumentPipeline(debug=False)  # Disable debug output in API


def clean_text(text):
    """Clean extracted text"""
    if not text:
        return ""
    # Remove special characters but keep basic punctuation
    cleaned = ''.join(c for c in text if c.isalnum() or c.isspace() or c in '.,()-')
    return cleaned.strip()


def format_date(date_str):
    """Format date string consistently"""
    if not date_str:
        return ""
    try:
        # Try different date formats
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"]:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                return date_obj.strftime("%Y-%m-%d")
            except ValueError:
                continue
        return clean_text(date_str)
    except:
        return clean_text(date_str)


def format_amount(amount_str):
    """Format amount consistently"""
    if not amount_str:
        return ""
    try:
        # Remove currency symbols and other characters
        cleaned = ''.join(c for c in amount_str if c.isdigit() or c in '.,')
        # Convert to float
        amount = float(cleaned.replace(',', ''))
        return f"${amount:.2f}"
    except:
        return clean_text(amount_str)


def format_response(result):
    """Format API response"""
    if "error" in result:
        return result
    
    # Clean and format extracted fields
    formatted_fields = {}
    for field, value in result["extracted_fields"].items():
        if 'date' in field.lower():
            formatted_value = format_date(value)
        elif 'amount' in field.lower():
            formatted_value = format_amount(value)
        else:
            formatted_value = clean_text(value)
        
        if formatted_value:  # Only include non-empty values
            formatted_fields[field] = formatted_value
    
    return {
        "document_type": result["document_type"],
        "extracted_fields": formatted_fields
    }


@app.get("/health")
async def health_check():
    """Check service health status"""
    try:
        # Verify models are loaded
        if not hasattr(pipeline, 'classifier') or not pipeline.classifier:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "message": "Classifier model not loaded"
                }
            )
        
        if not pipeline.extraction_models:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "message": "Extraction models not loaded"
                }
            )
        
        return {
            "status": "healthy",
            "models_loaded": {
                "classifier": True,
                "extractors": list(pipeline.extraction_models.keys())
            },
            "timestamp": time.time()
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": str(e)
            }
        )


@app.post("/classify-and-extract")
async def classify_and_extract(file: UploadFile = File(...)):
    """
    Process document image and extract information
    
    Args:
        file: Uploaded file (PDF or image)
    
    Returns:
        JSON with document type and extracted fields
    """
    try:
        # Validate file type
        allowed_types = {
            'application/pdf': '.pdf',
            'image/jpeg': '.jpg',
            'image/png': '.png'
        }
        
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"File type not supported. Allowed types: {', '.join(allowed_types.values())}"
            )
        
        # Read file content
        content = await file.read()
        
        # Create temporary file
        suffix = allowed_types[file.content_type]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Process document
            result = pipeline.process_document(temp_path)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            if "error" in result:
                raise HTTPException(
                    status_code=422,
                    detail=result["message"] if "message" in result else result["error"]
                )
            
            # Format and return response
            return format_response(result)
            
        finally:
            # Ensure temporary file is cleaned up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)