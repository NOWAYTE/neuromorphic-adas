import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from tqdm import tqdm
import kaggle
from zipfile import ZipFile
import shutil

from hybrid_model import HybridFusionModel
from neuromorphic_loader import EventDataLoader
from thermal_processor import ThermalProcessor, ThermalEncoder
from config import MODEL_PATHS, DATA_PATHS, MODEL_CONFIG

class MultiModalDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        """
        Dataset class for multimodal data (events, audio, thermal)
        
        Args:
            data_dir (str): Base directory containing the dataset
            split (str): 'train', 'val', or 'test'
            transform: Optional transform to be applied on a sample
        """
        self.data_dir = os.path.join(data_dir, split)
        self.split = split
        self.transform = transform
        
        # Load metadata
        self.metadata = pd.read_csv(os.path.join(self.data_dir, 'metadata.csv'))
        
        # Initialize data loaders
        self.event_loader = EventDataLoader(
            time_window=MODEL_CONFIG['event']['time_window'],
            height=MODEL_CONFIG['event']['height'],
            width=MODEL_CONFIG['event']['width']
        )
        
        self.thermal_processor = ThermalProcessor(
            input_size=MODEL_CONFIG['thermal']['input_size'],
            normalize=MODEL_CONFIG['thermal']['normalize']
        )
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Get sample metadata
        sample = self.metadata.iloc[idx]
        
        # Load event data
        event_data = self._load_events(sample['event_path'])
        
        # Load audio data
        audio_data = self._load_audio(sample['audio_path'])
        
        # Load thermal data
        thermal_data = self._load_thermal(sample['thermal_path'])
        
        # Get label
        label = sample['label']
        
        if self.transform:
            event_data, audio_data, thermal_data = self.transform(
                event_data, audio_data, thermal_data
            )
        
        return {
            'events': event_data,
            'audio': audio_data,
            'thermal': thermal_data,
            'label': label
        }
    
    def _load_events(self, event_path):
        """Load and process event data"""
        events = self.event_loader.load_events(event_path)
        return self.event_loader.events_to_tensor(events)
    
    def _load_audio(self, audio_path):
        """Load and preprocess audio data"""
        # Implement audio loading logic
        # Return preprocessed audio features
        pass
    
    def _load_thermal(self, thermal_path):
        """Load and preprocess thermal image"""
        # Implement thermal image loading
        # Return preprocessed thermal tensor
        pass

def train_model():
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = HybridFusionModel(
        num_classes=MODEL_CONFIG['num_classes'],
        thermal_feat_dim=MODEL_CONFIG['thermal']['feature_dim']
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # TensorBoard writer
    writer = SummaryWriter('runs/experiment_1')
    
    # Data loaders
    train_dataset = MultiModalDataset(DATA_PATHS['base'], split='train')
    val_dataset = MultiModalDataset(DATA_PATHS['base'], split='val')
    
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False, num_workers=4
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(100):  # or any number of epochs
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            # Move data to device
            events = batch['events'].to(device)
            audio = batch['audio'].to(device)
            thermal = batch['thermal'].to(device)
            labels = batch['label'].to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = model(events, audio, thermal)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        
        # Validation phase
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATHS['weights'])
            print(f'New best model saved with val_loss: {val_loss:.4f}')
        
        print(f'Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
    
    writer.close()

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            events = batch['events'].to(device)
            audio = batch['audio'].to(device)
            thermal = batch['thermal'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs, _ = model(events, audio, thermal)
            loss = criterion(outputs, labels)
            
            # Calculate metrics
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss /= len(val_loader)
    accuracy = correct / total
    
    return val_loss, accuracy

def setup_kaggle():
    """Setup Kaggle API and download dataset"""
    # Install kaggle if not already installed
    try:
        import kaggle
    except ImportError:
        !pip install -q kaggle
    
    # Create necessary directories
    os.makedirs('/root/.kaggle', exist_ok=True)
    
    # Upload kaggle.json (API key) and move to correct location
    if not os.path.exists('/root/.kaggle/kaggle.json'):
        from google.colab import files
        uploaded = files.upload()
        
        # Move the uploaded file
        for fn in uploaded.keys():
            shutil.move(fn, '/root/.kaggle/')
        
        # Set permissions
        os.chmod('/root/.kaggle/kaggle.json', 0o600)
    
    # Download dataset
    !kaggle datasets download -d YOUR_DATASET_NAME
    
    # Extract dataset
    with ZipFile('YOUR_DATASET_NAME.zip', 'r') as zip_ref:
        zip_ref.extractall(DATA_PATHS['base'])

if __name__ == '__main__':
    # Setup Kaggle and download data
    setup_kaggle()
    
    # Start training
    train_model()