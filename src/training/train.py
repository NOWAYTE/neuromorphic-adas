%%writefile training/train_utils.py
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import cv2

# Import the necessary processors
from utils.neuromorphic_loader import EventDataLoader
from utils.audio_processor import AudioProcessor
from utils.thermal_processor import ThermalProcessor

class MultiModalDataset(Dataset):
    def __init__(self, metadata_path, data_root, transform=None):
        """
        Dataset class for multimodal data (events, audio, thermal)
        """
        # Read metadata with error handling
        if isinstance(metadata_path, str):
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            with open(metadata_path, 'r') as f:
                self.metadata = pd.read_csv(f)
        elif isinstance(metadata_path, pd.DataFrame):
            self.metadata = metadata_path
        else:
            raise ValueError("metadata_path must be a string path or DataFrame")
            
        self.data_root = data_root
        self.transform = transform
        
        # Initialize processors
        self.event_loader = EventDataLoader()
        self.audio_processor = AudioProcessor()
        self.thermal_processor = ThermalProcessor()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Get sample metadata
        sample = self.metadata.iloc[idx]

        try:
            # Load event data
            event_path = os.path.join(self.data_root, sample['event_path'])
            events = self.event_loader.load_events(event_path)
            event_data = self.event_loader.events_to_tensor(events)

            # Load audio data
            audio_path = os.path.join(self.data_root, sample['audio_path'])
            audio = self.audio_processor.load_audio(audio_path)
            audio_data = self.audio_processor.extract_features(audio)

            # Load thermal data
            thermal_path = os.path.join(self.data_root, sample['thermal_path'])
            thermal_image = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
            if thermal_image is None:
                raise ValueError(f"Could not load thermal image: {thermal_path}")
            thermal_data = self.thermal_processor.preprocess(thermal_image)

            # Get label
            label = sample['label']

            return {
                'events': event_data,
                'audio': audio_data,
                'thermal': thermal_data,
                'label': torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            print(f"Error loading sample {idx} ({sample['event_path']}): {e}")
            # Return a dummy sample if there's an error
            return self._create_dummy_sample(label=sample['label'])

    def _create_dummy_sample(self, label=0):
        """Create a dummy sample for debugging"""
        return {
            'events': torch.randn(10, 2, 260, 346),
            'audio': torch.randn(64, 64),
            'thermal': torch.randn(1, 224, 224),
            'label': torch.tensor(label, dtype=torch.long)
        }

def create_data_loaders(metadata_path, data_root, batch_size=8, validation_split=0.2):
    """Create training and validation data loaders"""
    # Create dataset
    dataset = MultiModalDataset(metadata_path, data_root)
    
    # Split into train and validation
    train_size = int((1 - validation_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, val_loader