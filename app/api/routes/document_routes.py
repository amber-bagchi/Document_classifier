from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import io
from PIL import Image
from document_pipeline import DocumentPipeline
import time

app = FastAPI(title="Document Processing API")
pipeline = DocumentPipeline(debug=False)

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
        
        return {
            "status": "healthy",
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
        file: Uploaded image file (PDF or image)
    
    Returns:
        JSON with document type and extracted fields
    """
    try:
        # Read file content
        content = await file.read()
        
        # Check file type
        if file.filename.lower().endswith('.pdf'):
            # Save PDF temporarily and process
            pass
            
        if not processor:
            return JSONResponse(
                status_code=503,
                content={'error': 'Service not initialized'}
            )
                
        return JSONResponse(
            status_code=200,
            content={
                'supported_types': list(processor.class_mapping.values()),
                'allowed_file_extensions': list(allowed_file)
            }
        )
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )