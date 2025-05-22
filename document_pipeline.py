import os
import sys
import json
import time
import torch
import fitz
import io
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms
from app.models.document_cnn import DocumentCNN
from extraction_model import DocumentExtractor
from app.utils.logging_utils import DocumentLogger


class DocumentQualityError(Exception):
    """Custom exception for image quality issues"""
    def __init__(self, message, quality_metrics):
        self.message = message
        self.quality_metrics = quality_metrics
        super().__init__(self.message)


class DocumentPipeline:
    def __init__(self, debug=True):
        self.debug = debug
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.debug:
            print(f"Using device: {self.device}")
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize logger
        self.logger = DocumentLogger()
        
        # Quality thresholds - Set all minimums to 5
        self.quality_thresholds = {
            'min_blur': 5,
            'min_contrast': 5,
            'min_resolution': 5
        }
        
        # Load classifier and extraction models
        self.load_models()

    def inspect_checkpoint_architecture(self, checkpoint_path):
        """Helper function to inspect checkpoint architecture"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            keys = list(state_dict.keys())
            
            if self.debug:
                print(f"Checkpoint has {len(keys)} parameters")
                print("First 10 parameter names:")
                for key in keys[:10]:
                    print(f"  {key}")
            
            # Determine architecture
            is_resnet = any(key.startswith('resnet.') for key in keys)
            is_vgg_like = any(key.startswith('features.') for key in keys)
            
            if self.debug:
                print(f"\nArchitecture analysis:")
                print(f"  ResNet-based: {is_resnet}")
                print(f"  VGG-like: {is_vgg_like}")
            
            return keys, is_resnet, is_vgg_like
            
        except Exception as e:
            print(f"Error inspecting checkpoint: {e}")
            return None, False, False

    def load_models(self):
        """Load classifier and extraction models with improved error handling"""
        if self.debug:
            print("\nLoading models...")
        
        # Load classifier with architecture detection
        classifier_path = 'app/models/trained/best_cnn_model.pth'
        if os.path.exists(classifier_path):
            if self.debug:
                print(f"Found classifier checkpoint: {classifier_path}")
            
            # First, inspect the checkpoint to determine architecture
            keys, is_resnet, is_vgg_like = self.inspect_checkpoint_architecture(classifier_path)
            
            if keys is None:
                raise RuntimeError("Failed to inspect checkpoint architecture")
            
            # Load checkpoint
            checkpoint = torch.load(classifier_path, map_location=self.device)
            
            # Determine state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Initialize appropriate model architecture
            if is_resnet:
                try:
                    from app.models.document_cnn import DocumentCNNResNet
                    self.classifier = DocumentCNNResNet().to(self.device)
                    if self.debug:
                        print("Using ResNet-based architecture")
                except ImportError:
                    print("Warning: DocumentCNNResNet not found, using default DocumentCNN")
                    self.classifier = DocumentCNN().to(self.device)
            else:
                # Default to VGG-like architecture (which matches the checkpoint)
                self.classifier = DocumentCNN().to(self.device)
                if self.debug:
                    print("Using VGG-like architecture")
            
            # Load the state dict
            try:
                self.classifier.load_state_dict(state_dict, strict=True)
                self.classifier.eval()
                if self.debug:
                    print("✅ Successfully loaded classifier model with strict=True")
                    
            except RuntimeError as e:
                if self.debug:
                    print(f"Failed to load with strict=True: {str(e)[:100]}...")
                    print("Trying with strict=False...")
                
                # Try with strict=False
                try:
                    missing_keys, unexpected_keys = self.classifier.load_state_dict(
                        state_dict, strict=False
                    )
                    
                    if self.debug:
                        print(f"✅ Loaded with strict=False")
                        print(f"Missing keys: {len(missing_keys)}")
                        print(f"Unexpected keys: {len(unexpected_keys)}")
                        
                        if missing_keys and len(missing_keys) <= 5:
                            print("Missing keys:")
                            for key in missing_keys:
                                print(f"  - {key}")
                        elif missing_keys:
                            print(f"Missing keys (first 5 of {len(missing_keys)}):")
                            for key in missing_keys[:5]:
                                print(f"  - {key}")
                        
                        if unexpected_keys and len(unexpected_keys) <= 5:
                            print("Unexpected keys:")
                            for key in unexpected_keys:
                                print(f"  - {key}")
                        elif unexpected_keys:
                            print(f"Unexpected keys (first 5 of {len(unexpected_keys)}):")
                            for key in unexpected_keys[:5]:
                                print(f"  - {key}")
                    
                    # Check if too many keys are missing (indicates major architecture mismatch)
                    total_model_params = len(list(self.classifier.parameters()))
                    if len(missing_keys) > total_model_params * 0.5:
                        print("⚠️  Warning: More than 50% of model parameters are missing!")
                        print("This suggests a significant architecture mismatch.")
                        print("Model may not work correctly.")
                    
                    self.classifier.eval()
                    
                except Exception as e2:
                    print(f"❌ Failed to load classifier model: {e2}")
                    print("Using untrained model...")
                    # Continue with untrained model
            
            # Test the loaded model with a dummy input
            try:
                test_input = torch.randn(1, 3, 224, 224).to(self.device)
                with torch.no_grad():
                    test_output = self.classifier(test_input)
                if self.debug:
                    print(f"✅ Model forward pass test successful, output shape: {test_output.shape}")
            except Exception as e:
                print(f"⚠️  Model forward pass test failed: {e}")
                print("Model may not work correctly during inference.")
                
        else:
            raise FileNotFoundError(f"No classifier model found at {classifier_path}")
        
        # Load extraction models for each category
        self.extraction_models = {}
        category_fields = {
            'invoice': ['company_name', 'invoice_number', 'date', 'amount'],
            'payslip': ['employee_name', 'employee_id', 'bank', 'amount'],
            'certificate': ['name', 'course_name', 'course_by', 'date'],
            'resume': ['name', 'education', 'university', 'date']
        }
        
        for category in category_fields.keys():
            if self.debug:
                print(f"\nLoading {category} extraction model...")
            
            try:
                # Initialize model
                model = DocumentExtractor(
                    category=category,
                    fields=category_fields[category]
                ).to(self.device)
                
                # Load trained weights
                model_path = f'app/models/trained/best_{category}_extraction_model.pth'
                if os.path.exists(model_path):
                    if self.debug:
                        print(f"Found extraction model: {model_path}")
                    
                    checkpoint = torch.load(model_path, map_location=self.device)
                    
                    # Handle different checkpoint formats
                    if isinstance(checkpoint, dict):
                        if 'model_state_dict' in checkpoint:
                            state_dict = checkpoint['model_state_dict']
                        elif 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                        else:
                            state_dict = checkpoint
                    else:
                        state_dict = checkpoint
                    
                    # Load with error handling
                    try:
                        model.load_state_dict(state_dict, strict=True)
                        if self.debug:
                            print(f"✅ Loaded {category} model with strict=True")
                    except RuntimeError:
                        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                        if self.debug:
                            print(f"✅ Loaded {category} model with strict=False")
                            print(f"  Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
                    
                    model.eval()
                    self.extraction_models[category] = {
                        'model': model,
                        'fields': category_fields[category]
                    }
                    
                else:
                    if self.debug:
                        print(f"⚠️  No extraction model found at {model_path}")
                    
            except Exception as e:
                print(f"❌ Error loading {category} extraction model: {e}")
                
        if self.debug:
            print(f"\n📊 Loaded extraction models for: {list(self.extraction_models.keys())}")

    def assess_image_quality(self, image):
        """Assess image quality using various metrics"""
        # Convert PIL Image to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 1. Blur Detection using Laplacian variance
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        # 2. Contrast Assessment
        contrast_score = float(gray.std())
        
        # 3. Resolution Check
        height, width = cv_image.shape[:2]
        resolution_score = float(min(width, height) / 224)
        
        if self.debug:
            print("\nImage Quality Metrics:")
            print(f"Blur Score: {blur_score:.2f}")
            print(f"Contrast Score: {contrast_score:.2f}")
            print(f"Resolution Score: {resolution_score:.2f}")
        
        # Quality checks with specific feedback
        quality_issues = []
        
        if blur_score < self.quality_thresholds['min_blur']:
            quality_issues.append(f"Image is too blurry (score: {blur_score:.1f}, minimum: {self.quality_thresholds['min_blur']})")
        
        if contrast_score < self.quality_thresholds['min_contrast']:
            quality_issues.append(f"Image has poor contrast (score: {contrast_score:.1f}, minimum: {self.quality_thresholds['min_contrast']})")
        
        if resolution_score < self.quality_thresholds['min_resolution']:
            quality_issues.append(f"Image resolution is too low (score: {resolution_score:.1f}x, minimum: {self.quality_thresholds['min_resolution']}x)")
        
        # Quality metrics for reporting
        quality_metrics = {
            'blur_score': blur_score,
            'contrast_score': contrast_score,
            'resolution_score': resolution_score,
            'thresholds': self.quality_thresholds,
            'status': {
                'blur_check': blur_score >= self.quality_thresholds['min_blur'],
                'contrast_check': contrast_score >= self.quality_thresholds['min_contrast'],
                'resolution_check': resolution_score >= self.quality_thresholds['min_resolution']
            }
        }
        
        if quality_issues:
            recommendations = [
                "Please provide a clearer image with:",
                "- Good lighting and contrast",
                "- No motion blur",
                "- High resolution scan or photo",
                "- Proper document alignment"
            ]
            
            raise DocumentQualityError(
                "Image quality issues detected:\n- " + "\n- ".join(quality_issues) +
                "\n\nRecommendations:\n" + "\n".join(recommendations),
                quality_metrics
            )
        
        if self.debug:
            print("Quality Assessment: PASS")
            print("\nQuality Status:")
            for check, status in quality_metrics['status'].items():
                print(f"{check}: {'PASS' if status else 'FAIL'}")
        
        return quality_metrics

    def convert_pdf_to_image(self, pdf_path):
        """Convert first page of PDF to PIL Image"""
        if self.debug:
            print(f"\nConverting PDF: {pdf_path}")
        
        try:
            # Convert PDF to image
            doc = fitz.open(pdf_path)
            
            if len(doc) == 0:
                print("Error: PDF has no pages")
                return None
                
            page = doc[0]
            
            # Convert to PNG with higher resolution
            pix = page.get_pixmap(matrix=fitz.Matrix(4.0, 4.0))
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(img_data))
            doc.close()
            
            if self.debug:
                print(f"Successfully converted PDF to image of size {image.size}")
            
            return image
            
        except Exception as e:
            print(f"Error converting PDF to image: {str(e)}")
            return None

    def classify_document(self, image):
        """Classify document using CNN model"""
        if self.debug:
            print("\nClassifying document...")
        
        try:
            # Prepare image
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.classifier(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                confidence, predicted = torch.max(probabilities, 0)
                
                # Map class index to category
                categories = ['invoice', 'payslip', 'resume', 'certificate']
                predicted_category = categories[predicted.item()]
                
                if self.debug:
                    print(f"\nClassified as: {predicted_category}")
                    print(f"Confidence: {float(confidence):.2%}")
                    print("All probabilities:")
                    for i, cat in enumerate(categories):
                        print(f"  {cat}: {float(probabilities[i]):.2%}")
                
                return predicted_category, float(confidence)
                
        except Exception as e:
            print(f"Error during classification: {e}")
            # Return default values in case of error
            return 'unknown', 0.0

    def extract_fields(self, image, doc_type):
        """Extract fields based on document type"""
        if doc_type not in self.extraction_models:
            if self.debug:
                print(f"⚠️  No extraction model available for document type: {doc_type}")
                print(f"Available models: {list(self.extraction_models.keys())}")
            return {}
        
        if self.debug:
            print(f"\nExtracting fields for {doc_type} document...")
        
        try:
            # Prepare image
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get model and fields
            model_info = self.extraction_models[doc_type]
            model = model_info['model']
            expected_fields = model_info['fields']
            
            if self.debug:
                print(f"Expected fields: {expected_fields}")
            
            # Extract fields
            with torch.no_grad():
                outputs = model(image_tensor)
                if self.debug:
                    print(f"Model forward pass completed, output shape: {outputs.shape}")
                
                # Check if model has extract_fields method
                if not hasattr(model, 'extract_fields'):
                    if self.debug:
                        print("❌ Model does not have extract_fields method")
                    return {}
                
                extracted = model.extract_fields(outputs, debug=self.debug)
                if self.debug:
                    print(f"Field extraction completed")
                    print(f"Raw extraction result: {extracted}")
            
            # Format results
            results = {}
            
            if not extracted:
                if self.debug:
                    print("⚠️  No fields extracted from model")
                return {}
            
            for field, data in extracted.items():
                if self.debug:
                    print(f"Processing field '{field}': {data}")
                
                if isinstance(data, dict) and 'values' in data and 'confidence' in data:
                    if len(data['values']) > 0 and len(data['confidence']) > 0:
                        value = data['values'][0]
                        confidence = float(data['confidence'][0])
                        
                        if self.debug:
                            print(f"  {field}: value='{value}', confidence={confidence:.4f}")
                        
                        # Only include high confidence predictions - Lower threshold for debugging
                        if confidence > 0.1:  # Lowered from 0.4 to 0.1 for debugging
                            # Clean and format the value
                            original_value = str(value)
                            
                            if 'date' in field.lower():
                                # Try to format date consistently
                                try:
                                    from datetime import datetime
                                    # Try different date formats
                                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']:
                                        try:
                                            date_obj = datetime.strptime(str(value), fmt)
                                            value = date_obj.strftime('%Y-%m-%d')
                                            break
                                        except:
                                            continue
                                except:
                                    pass
                            elif 'amount' in field.lower():
                                # Try to format amount consistently
                                try:
                                    # Remove common currency symbols and formatting
                                    clean_value = str(value).replace(',', '').replace('$', '').replace('€', '').replace('£', '').replace('₹', '').replace('¥', '').replace('₩', '').replace('₽', '').replace('₦', '').replace('₡', '').replace('₪', '').replace('₫', '').replace('₭', '').replace('₮', '').replace('₯', '').replace('₰', '').replace('₱', '').replace('₲', '').replace('₳', '').replace('₴', '').replace('₵', '').replace('₶', '').replace('₷', '').replace('₸', '').replace('₺', '').replace('₻', '').replace('₼', '').replace('₾', '').replace('₿', '').replace('%', '').replace(' ', '').strip()
                                    
                                    # Handle negative values
                                    is_negative = clean_value.startswith('-') or (clean_value.startswith('(') and clean_value.endswith(')'))
                                    clean_value = clean_value.replace('-', '').replace('(', '').replace(')', '')
                                    
                                    # Try to convert to float
                                    amount = float(clean_value)
                                    
                                    # Apply negative sign if needed
                                    if is_negative:
                                        amount = -amount
                                    
                                    # Format with appropriate currency symbol (defaulting to $)
                                    if amount >= 0:
                                        value = f"${amount:.2f}"
                                    else:
                                        value = f"-${abs(amount):.2f}"
                                        
                                except (ValueError, TypeError):
                                    # If conversion fails, keep original value
                                    pass
                            
                            results[field] = str(value)
                            if self.debug:
                                print(f"  ✅ Added '{field}': '{original_value}' -> '{value}'")
                        else:
                            if self.debug:
                                print(f"  ❌ Skipped '{field}' (confidence {confidence:.4f} <= 0.4)")
                    else:
                        if self.debug:
                            print(f"  ❌ Invalid data structure for field '{field}'")
                else:
                    if self.debug:
                        print(f"  ❌ Unexpected data format for field '{field}': {type(data)}")
            
            if self.debug:
                print(f"Final extracted fields: {results}")
            
            return results
            
        except Exception as e:
            if self.debug:
                print(f"❌ Error during field extraction: {e}")
                import traceback
                traceback.print_exc()
            else:
                print(f"Error during field extraction: {e}")
            return {}

    def process_document(self, pdf_path):
        """Process document through the pipeline"""
        start_time = time.time()
        
        if self.debug:
            print(f"\n{'='*50}")
            print("🚀 Starting document processing...")
            print(f"{'='*50}")
        
        try:
            # Validate input file
            if not os.path.exists(pdf_path):
                error_result = {
                    "error": "File not found",
                    "message": f"The file {pdf_path} does not exist"
                }
                self.logger.log_prediction(
                    filename=os.path.basename(pdf_path),
                    result=error_result,
                    processing_time=time.time() - start_time
                )
                return error_result
            
            # 1. Convert PDF to image
            image = self.convert_pdf_to_image(pdf_path)
            if image is None:
                error_result = {
                    "error": "Failed to convert PDF to image",
                    "message": "Please ensure the PDF file is valid and not corrupted"
                }
                self.logger.log_prediction(
                    filename=os.path.basename(pdf_path),
                    result=error_result,
                    processing_time=time.time() - start_time
                )
                return error_result
            
            # 2. Assess image quality
            try:
                quality_metrics = self.assess_image_quality(image)
            except DocumentQualityError as e:
                error_result = {
                    "error": "Image quality check failed",
                    "message": str(e),
                    "quality_metrics": e.quality_metrics
                }
                self.logger.log_prediction(
                    filename=os.path.basename(pdf_path),
                    result=error_result,
                    processing_time=time.time() - start_time
                )
                return error_result
            
            # 3. Classify document
            doc_type, confidence = self.classify_document(image)
            
            # 4. Extract fields
            extracted_fields = self.extract_fields(image, doc_type)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Log detailed results
            detailed_result = {
                "document_type": doc_type.capitalize(),
                "classification_confidence": confidence,
                "extracted_fields": extracted_fields,
                "quality_metrics": quality_metrics,
                "processing_time": processing_time,
                "success": True
            }
            
            self.logger.log_prediction(
                filename=os.path.basename(pdf_path),
                result=detailed_result,
                processing_time=processing_time
            )
            
            # Return simplified result for user
            result = {
                "document_type": doc_type.capitalize(),
                "classification_confidence": f"{confidence:.2%}",
                "extracted_fields": extracted_fields,
                "processing_time": f"{processing_time:.2f}s",
                "success": True
            }
            
            if self.debug:
                print(f"\n{'='*50}")
                print("✅ Processing completed successfully!")
                print(f"{'='*50}")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_result = {
                "error": "Processing failed",
                "message": str(e),
                "processing_time": f"{processing_time:.2f}s",
                "success": False
            }
            
            self.logger.log_prediction(
                filename=os.path.basename(pdf_path),
                result=error_result,
                processing_time=processing_time
            )
            
            if self.debug:
                print(f"\n{'='*50}")
                print("❌ Processing failed!")
                print(f"Error: {str(e)}")
                print(f"{'='*50}")
            
            return error_result


def main():
    """Main function to run the document pipeline"""
    if len(sys.argv) < 2:
        print("Usage: python document_pipeline.py <pdf_path>")
        print("Example: python document_pipeline.py sample_document.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    print("🔧 Initializing Document Processing Pipeline...")
    
    try:
        # Initialize pipeline with debug mode
        pipeline = DocumentPipeline(debug=True)
        
        print(f"\n📄 Processing document: {pdf_path}")
        
        # Process document
        result = pipeline.process_document(pdf_path)
        
        # Print results
        print(f"\n{'='*50}")
        print("📊 PROCESSING RESULTS")
        print(f"{'='*50}")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Exit with appropriate code
        if result.get("success", False):
            print(f"\n🎉 Document processed successfully!")
            sys.exit(0)
        else:
            print(f"\n💥 Document processing failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()