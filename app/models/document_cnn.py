import torch
import torch.nn as nn
import torchvision.models as models


class DocumentCNN(nn.Module):
    """CNN model for document classification based on ResNet50"""
    
    def __init__(self, num_classes=4):
        super(DocumentCNN, self).__init__()
        
        # Load ResNet50 without pretrained weights
        self.resnet = models.resnet50(pretrained=False)
        
        # Replace final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)