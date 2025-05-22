#!/bin/bash

# Test API endpoints

# Test health endpoint
echo "Testing health endpoint..."
curl -X GET "http://localhost:8000/health" \
     -H "Content-Type: application/json"

echo -e "\n\n"

# Test document processing endpoint
echo "Testing document processing endpoint..."
# Note: Replace 'sample_invoice.pdf' with actual test file
curl -X POST "http://localhost:8000/classify-and-extract" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample_invoice.pdf"

echo -e "\n\n"

# Test with invalid file type
echo "Testing with invalid file type..."
curl -X POST "http://localhost:8000/classify-and-extract" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test.txt"

echo -e "\n\n"

# Test API documentation
echo "API Documentation available at: http://localhost:8000/docs"
echo "OpenAPI schema available at: http://localhost:8000/openapi.json"