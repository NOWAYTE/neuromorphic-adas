import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class ThermalProcessor:
    def __init__(self, input_size=(224, 224), normalize=True):
        """
        Initialize thermal image processor
        
        Args:
            input_size (tuple): Target size for resizing images (height, width)
            normalize (bool): Whether to normalize thermal values
        """
        self.input_size = input_size
        self.normalize = normalize
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ])
    
    def preprocess(self, thermal_image):
        """
        Preprocess a single thermal image
        
        Args:
            thermal_image (numpy.ndarray): Input thermal image (single channel)
            
        Returns:
            torch.Tensor: Preprocessed thermal image tensor
        """
        # Convert to float32 if needed
        if thermal_image.dtype != np.float32:
            thermal_image = thermal_image.astype(np.float32)
            
        # Normalize to [0, 1] if not already
        if thermal_image.max() > 1.0:
            thermal_image = thermal_image / thermal_image.max()
            
        # Apply transformations
        tensor = self.transform(thermal_image)
        
        # Add batch dimension if needed
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
            
        return tensor
    
    def add_thermal_noise(self, thermal_image, noise_level=0.1):
        """Add realistic thermal noise to the image"""
        noise = np.random.normal(0, noise_level, thermal_image.shape).astype(np.float32)
        return np.clip(thermal_image + noise, 0, 1)
    
    def adjust_thermal_contrast(self, thermal_image, alpha=1.0, beta=0.0):
        """Adjust contrast of thermal image"""
        return np.clip(alpha * thermal_image + beta, 0, 1)


class ThermalEncoder(nn.Module):
    def __init__(self, input_channels=1, base_channels=32, output_dim=128):
        """
        CNN-based thermal feature extractor
        """
        super().__init__()
        self.features = nn.Sequential(
            # Initial conv block
            nn.Conv2d(input_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Residual blocks
            self._make_residual_block(base_channels, base_channels * 2, 2),
            self._make_residual_block(base_channels * 2, base_channels * 4, 2),
            self._make_residual_block(base_channels * 4, base_channels * 8, 2),
            
            # Final pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            
            # Projection head
            nn.Linear(base_channels * 8, output_dim)
        )
    
    def _make_residual_block(self, in_channels, out_channels, stride):
        """Create a residual block with skip connection"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.features(x)
