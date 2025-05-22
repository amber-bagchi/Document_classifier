# Document Information Extraction Model Training Documentation

## Overview
This documentation covers the training process and architecture of our document information extraction models, which are designed to extract specific fields from different types of documents after classification.

## Model Architecture

### Base Design
- Architecture: Custom CNN with attention mechanisms
- Input: Document images (224x224 pixels, RGB)
- Output: Multiple text fields with confidence scores
- Separate models for each document type

### Document-Specific Models

1. **Invoice Extraction Model:**
   - Fields:
     * company_name (text)
     * invoice_number (alphanumeric)
     * date (formatted date)
     * amount (numeric)
   - Special handling for currency and date formats

2. **Payslip Extraction Model:**
   - Fields:
     * employee_name (text)
     * employee_id (alphanumeric)
     * bank (text)
     * amount (numeric)
   - Focus on table structure recognition

3. **Certificate Extraction Model:**
   - Fields:
     * name (text)
     * course_name (text)
     * course_by (text)
     * date (formatted date)
   - Emphasis on layout understanding

4. **Resume Extraction Model:**
   - Fields:
     * name (text)
     * education (text)
     * university (text)
     * date (formatted date)
   - Section-based extraction approach

## Training Process

### Data Preparation

1. **Image Processing:**
   - High-resolution scanning (300 DPI)
   - Automatic deskewing
   - Contrast enhancement
   - Noise reduction

2. **Ground Truth Data:**
   - JSON format annotations
   - Field-specific bounding boxes
   - Text content for each field
   - Format specifications

3. **Augmentation Techniques:**
   - Random brightness/contrast
   - Slight rotations (±5°)
   - Gaussian noise
   - Random erasing

### Training Configuration

1. **Common Parameters:**
   - Batch Size: 16
   - Learning Rate: 0.0001
   - Optimizer: AdamW
   - Epochs: 50

2. **Loss Functions:**
   - Text Recognition: CTC Loss
   - Layout Detection: IoU Loss
   - Confidence: MSE Loss

3. **Training Strategy:**
   - Progressive learning (easy to hard samples)
   - Multi-task learning approach