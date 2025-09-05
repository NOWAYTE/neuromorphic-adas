import torch
import torch.nn as nn
import torch.nn.functional as F
from .thermal_processor import ThermalEncoder

class HybridFusionModel(nn.Module):
    def __init__(self, num_classes=3, thermal_feat_dim=128):
        """
        Hybrid Neuromorphic-Acoustic-Thermal Fusion Model
        num_classes: 0=normal, 1=siren, 2=hazard
        thermal_feat_dim: Dimension of thermal features
        """
        super().__init__()
        
        # Neuromorphic Processing Branch
        self.event_encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 65 * 86, 256)
        )
        
        # Acoustic Processing Branch
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 128)
        )
        
        # Thermal Processing Branch
        self.thermal_encoder = ThermalEncoder(output_dim=thermal_feat_dim)
        
        # Feature dimensions for attention
        self.event_dim = 256
        self.audio_dim = 128
        self.thermal_dim = thermal_feat_dim
        
        # Attention-based Fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=self.event_dim + self.audio_dim + self.thermal_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.event_dim + self.audio_dim + self.thermal_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Confidence Head
        self.confidence = nn.Sequential(
            nn.Linear(self.event_dim + self.audio_dim + self.thermal_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, event_input, audio_input, thermal_input=None):
        # Process event data
        batch_size, seq_len = event_input.shape[0], event_input.shape[1]
        event_input = event_input.reshape(-1, *event_input.shape[2:])
        event_features = self.event_encoder(event_input)
        event_features = event_features.reshape(batch_size, seq_len, -1).mean(dim=1)
        
        # Process audio data
        audio_features = self.audio_encoder(audio_input.unsqueeze(1))
        
        # Process thermal data if provided
        if thermal_input is not None:
            thermal_features = self.thermal_encoder(thermal_input)
            combined = torch.cat([event_features, audio_features, thermal_features], dim=1)
        else:
            combined = torch.cat([event_features, audio_features], dim=1)
        
        # Apply attention
        attn_output, _ = self.attention(
            combined.unsqueeze(1),
            combined.unsqueeze(1),
            combined.unsqueeze(1)
        )
        fused = attn_output.squeeze(1)
        
        # Outputs
        classification = self.classifier(fused)
        confidence = self.confidence(fused)
        
        return classification, confidence


class EventProcessing(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3d = nn.Conv3d(2, 16, kernel_size=(3,3,3), padding=1)
        self.pool = nn.MaxPool3d((1,2,2))
        
    def forward(self, x):  # x: [B, T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        return self.pool(F.relu(self.conv3d(x)))

audio_net = nn.Sequential(
    nn.Conv1d(64, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool1d(2),
    nn.Conv1d(32, 64, kernel_size=3, padding=1),
    nn.AdaptiveAvgPool1d(1),
    nn.Flatten()
)

def add_fog(image, severity=0.7):
    fog = np.ones_like(image) * 255 * severity
    return cv2.addWeighted(image, 1-severity, fog, severity, 0)

class HybridFusion(nn.Module):
    def __init__(self, feat_dims=[256, 128, 64]):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=sum(feat_dims), 
            num_heads=4,
            batch_first=True
        )
        
    def forward(self, event_feats, audio_feats, thermal_feats):
        combined = torch.cat([event_feats, audio_feats, thermal_feats], dim=-1)
        attn_out, _ = self.attn(combined, combined, combined)
        return attn_out

confidence = 1 - torch.exp(-torch.norm(features, dim=1))