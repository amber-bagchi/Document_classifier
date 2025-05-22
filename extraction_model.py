import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, num_heads)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
    
    def forward(self, x):
        x = x + self.self_attention(self.ln_1(x), self.ln_1(x), self.ln_1(x))[0]
        x = x + self.mlp(self.ln_2(x))
        return x


class ViT(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        # Patch embedding
        self.class_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.conv_proj = nn.Conv2d(3, hidden_size, kernel_size=16, stride=16)
        self.pos_embed = nn.Parameter(torch.randn(1, 197, hidden_size))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            ViTBlock(hidden_size) for _ in range(12)
        ])
        
        # Layer norm and heads
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.head1 = nn.Linear(hidden_size, hidden_size)
        self.head2 = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        # Patch embedding
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add class token
        batch_size = x.shape[0]
        cls_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Apply layer norm
        x = self.layer_norm(x)
        
        # Get class token output and apply heads
        x = x[:, 0]  # Take CLS token
        x = self.head1(x)
        x = self.head2(x)
        
        return x


class DocumentExtractor(nn.Module):
    def __init__(self, category, fields, hidden_size=768, max_seq_length=50, vocab_size=95, image_size=224):
        super().__init__()
        
        self.category = category
        self.fields = fields
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        self.image_size = image_size
        
        # Vision Transformer backbone
        self.vit = ViT(hidden_size=hidden_size)
        
        # Field-specific heads
        self.field_heads = nn.ModuleDict()
        self.field_decoders = nn.ModuleDict()
        
        for field in fields:
            # Field extraction head
            self.field_heads[field] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            )
            
            # Field decoder
            self.field_decoders[field] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, vocab_size * max_seq_length)
            )
        
        # Character set for decoding
        self.chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
    
    def forward(self, x):
        # Get features from ViT
        features = self.vit(x)
        
        # Extract fields
        outputs = {}
        for field_name in self.fields:
            # Get field-specific features
            field_features = self.field_heads[field_name](features)
            
            # Decode field value
            logits = self.field_decoders[field_name](field_features)
            
            # Reshape logits to [batch_size, max_seq_length, vocab_size]
            batch_size = logits.size(0)
            logits = logits.view(batch_size, self.max_seq_length, self.vocab_size)
            
            outputs[field_name] = {
                'logits': logits,
                'features': field_features
            }
        
        return outputs
    
    def extract_fields(self, outputs, debug=False):
        """Convert model outputs to text fields"""
        extracted_fields = {}
        
        for field_name, field_output in outputs.items():
            # Get probabilities for each character position
            char_probs = F.softmax(field_output['logits'], dim=-1)
            
            # Extract text and confidence for each sample in batch
            texts = []
            confidences = []
            
            for sample_probs in char_probs:
                text, conf = self._decode_text(sample_probs)
                texts.append(text)
                confidences.append(conf)
            
            if debug:
                print(f"\n{field_name}:")
                print(f"Raw text: '{text}'")
                print(f"Confidence: {conf:.4f}")
            
            extracted_fields[field_name] = {
                'values': texts,
                'confidence': confidences,
                'features': field_output['features'].detach().cpu().numpy()
            }
        
        return extracted_fields
    
    def _decode_text(self, char_probs, min_confidence=0.3):
        """Decode character probabilities to text with confidence"""
        # Get most likely characters and their probabilities
        probs, indices = char_probs.max(dim=-1)
        
        # Convert to text while tracking confidence
        text = []
        conf_sum = 0
        conf_count = 0
        
        for idx, prob in zip(indices, probs):
            if prob > min_confidence:
                char = self.idx_to_char[idx.item()]
                text.append(char)
                conf_sum += prob.item()
                conf_count += 1
        
        # Calculate average confidence
        avg_conf = conf_sum / max(conf_count, 1)
        text = ''.join(text).strip()
        
        return text, avg_conf


def test_model(model, image_tensor, debug=True):
    """Test model's field extraction on a single image"""
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor.unsqueeze(0))
        extracted = model.extract_fields(outputs, debug=debug)
        
        if debug:
            print("\nExtracted Fields:")
            for field, data in extracted.items():
                print(f"\n{field}:")
                print(f"Value: '{data['values'][0]}'")
                print(f"Confidence: {data['confidence'][0]:.4f}")
        
        return extracted