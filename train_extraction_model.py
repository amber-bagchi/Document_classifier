import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
from extraction_model import DocumentExtractor
from tqdm import tqdm

class DocumentExtractionDataset(Dataset):
    """Dataset for document field extraction"""
    
    def __init__(self, data_file, image_dir, category, split='train', max_seq_length=50):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        self.image_dir = image_dir
        self.category = category
        self.max_seq_length = max_seq_length
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Character set for encoding/decoding
        self.chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        
        # Get category data
        self.category_data = self.data[category]
        self.fields = self.category_data['fields']
        
        # Select train or test data
        self.documents = (
            self.category_data['train_data'] if split == 'train'
            else self.category_data['test_data']
        )
    
    def __len__(self):
        return len(self.documents)
    
    def _encode_text(self, text):
        """Convert text to character indices (not one-hot)"""
        # Create indices tensor [seq_length]
        indices = torch.full((self.max_seq_length,), -1, dtype=torch.long)
        
        # Fill in character positions
        for i, c in enumerate(str(text)[:self.max_seq_length]):
            if c in self.char_to_idx:
                indices[i] = self.char_to_idx[c]
        
        return indices
    
    def _get_image_path(self, filename):
        """Get the correct image path, preferring PNG over PDF"""
        base_path = os.path.join(self.image_dir, f'generated_{self.category}s')
        
        # Try PNG first
        png_path = os.path.join(base_path, filename.replace('.pdf', '.png'))
        if os.path.exists(png_path):
            return png_path
        
        # Then try original filename
        orig_path = os.path.join(base_path, filename)
        if os.path.exists(orig_path):
            return orig_path
        
        raise FileNotFoundError(f"No image file found for {filename}")
    
    def __getitem__(self, idx):
        # Get document data
        doc = self.documents[idx]
        
        try:
            # Get image path
            image_path = self._get_image_path(doc['file_name'])
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
            
            # Encode field values
            field_values = {}
            for field in self.fields:
                field_values[field] = self._encode_text(doc['fields'][field])
            
            return {
                'image': image_tensor,
                'field_values': field_values,
                'raw_values': {field: doc['fields'][field] for field in self.fields}
            }
        
        except Exception as e:
            print(f"Error loading {doc['file_name']}: {str(e)}")
            raise

def train_extraction_model(train_loader, val_loader, model, device, category, num_epochs=10):
    """Train the extraction model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Changed to CrossEntropyLoss with ignore_index
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_samples = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in pbar:
            # Move data to device
            images = batch['image'].to(device)
            field_values = {
                k: v.to(device) for k, v in batch['field_values'].items()
            }
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss for each field
            batch_loss = 0
            for field_name in outputs:
                # Get logits and target values
                logits = outputs[field_name]['logits']  # [batch_size, max_seq_length, vocab_size]
                targets = field_values[field_name]      # [batch_size, max_seq_length]
                
                # Reshape for CrossEntropyLoss: [batch_size * max_seq_length, vocab_size] and [batch_size * max_seq_length]
                batch_size, seq_len, vocab_size = logits.shape
                logits_flat = logits.view(-1, vocab_size)
                targets_flat = targets.view(-1)
                
                # Calculate CrossEntropy loss
                field_loss = criterion(logits_flat, targets_flat)
                batch_loss += field_loss
            
            # Backward pass
            optimizer.zero_grad()
            batch_loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update metrics
            train_loss += batch_loss.item()
            train_samples += images.size(0)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{batch_loss.item():.4f}'})
        
        avg_train_loss = train_loss / train_samples
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_samples = 0
        field_accuracies = {field: [] for field in model.fields}
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch in pbar:
                # Move data to device
                images = batch['image'].to(device)
                field_values = {
                    k: v.to(device) for k, v in batch['field_values'].items()
                }
                raw_values = batch['raw_values']
                
                # Forward pass
                outputs = model(images)
                
                # Calculate loss and accuracy for each field
                batch_loss = 0
                for field_name in outputs:
                    # Get logits and target values
                    logits = outputs[field_name]['logits']  # [batch_size, max_seq_length, vocab_size]
                    targets = field_values[field_name]      # [batch_size, max_seq_length]
                    
                    # Reshape for loss calculation
                    batch_size, seq_len, vocab_size = logits.shape
                    logits_flat = logits.view(-1, vocab_size)
                    targets_flat = targets.view(-1)
                    
                    # Calculate loss
                    field_loss = criterion(logits_flat, targets_flat)
                    batch_loss += field_loss
                    
                    # Calculate character-level accuracy
                    pred_chars = logits.argmax(dim=-1)  # [batch_size, max_seq_length]
                    
                    # Only count accuracy for valid characters (not padding)
                    valid_mask = targets != -1
                    if valid_mask.sum() > 0:
                        accuracy = (pred_chars == targets)[valid_mask].float().mean().item()
                        field_accuracies[field_name].append(accuracy)
                
                # Update metrics
                val_loss += batch_loss.item()
                val_samples += images.size(0)
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{batch_loss.item():.4f}'})
        
        avg_val_loss = val_loss / val_samples
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Print epoch results
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
        print('\nField-wise Character Accuracy:')
        for field, accuracies in field_accuracies.items():
            if accuracies:  # Only print if there are accuracies calculated
                avg_acc = np.mean(accuracies) * 100
                print(f'{field}: {avg_acc:.2f}%')
        
        # Save best model and check early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss,
            }, f'app/models/trained/best_{category}_extraction_model.pth')
            print('\nSaved best model checkpoint')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping after {epoch + 1} epochs')
                break
        
        print()

def train_category_model(category, data_file, image_dir):
    """Train model for a specific document category"""
    print(f"\nTraining model for {category} documents...")
    
    # Create datasets
    train_dataset = DocumentExtractionDataset(
        data_file, image_dir, category, split='train'
    )
    val_dataset = DocumentExtractionDataset(
        data_file, image_dir, category, split='test'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2
    )
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DocumentExtractor(
        category=category,
        fields=train_dataset.fields
    ).to(device)
    
    # Print dataset info
    print(f"Dataset size: {len(train_dataset)} training, {len(val_dataset)} validation")
    print(f"Training on device: {device}")
    print(f"Fields to extract: {train_dataset.fields}")
    
    # Train model with 10 epochs
    train_extraction_model(train_loader, val_loader, model, device, category, num_epochs=10)

def main():
    # Create directories
    os.makedirs('app/models/trained', exist_ok=True)
    
    # Training settings
    data_file = 'data/extraction/extraction_data.json'
    image_dir = 'images'
    categories = ['invoice', 'payslip', 'certificate', 'resume']
    
    # Train model for each category
    for category in categories:
        train_category_model(category, data_file, image_dir)

if __name__ == '__main__':
    main()