"""
Neuromorphic-ADAS Training Script for Google Colab

This script handles:
1. Setting up the Colab environment
2. Installing dependencies
3. Downloading and preparing the Kaggle dataset
4. Training the hybrid model
5. Saving the trained model
"""

def setup_colab():
    """Set up the Colab environment and install dependencies"""
    print("Setting up Colab environment...")
    
    # Install required packages
    !pip install -q kaggle torchinfo tensorboardX
    
    # Mount Google Drive
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Create necessary directories
    !mkdir -p /content/dataset
    !mkdir -p /root/.kaggle
    
    print("Colab environment setup complete!")

def setup_kaggle(kaggle_json_path):
    """Set up Kaggle API with credentials"""
    print("Setting up Kaggle API...")
    
    # Copy kaggle.json to the right location
    !cp "{kaggle_json_path}" /root/.kaggle/
    !chmod 600 /root/.kaggle/kaggle.json
    
    # Test the Kaggle API
    !kaggle datasets list
    
    print("Kaggle API setup complete!")

def download_dataset(dataset_name, target_dir):
    """Download and extract the Kaggle dataset"""
    print(f"Downloading dataset: {dataset_name}")
    
    # Create target directory
    !mkdir -p "{target_dir}"
    
    # Download dataset
    !kaggle datasets download -d {dataset_name} -p "{target_dir}" --unzip
    
    print("Dataset download and extraction complete!")

def train():
    """Main training function"""
    import os
    import sys
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
    import shutil
    
    # Add src to path
    sys.path.append('/content/drive/MyDrive/neuromorphic-adas/src')
    
    # Import our modules
    from colab.hybrid_model import HybridFusionModel
    from colab.train import MultiModalDataset, validate
    from config import MODEL_PATHS, DATA_PATHS, MODEL_CONFIG
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = HybridFusionModel(
        num_classes=MODEL_CONFIG['num_classes'],
        thermal_feat_dim=MODEL_CONFIG['thermal']['feature_dim']
    ).to(device)
    
    # Print model summary
    from torchinfo import summary
    batch_size = 4
    summary(
        model, 
        input_size=[
            (batch_size, 10, 2, MODEL_CONFIG['event']['height'], MODEL_CONFIG['event']['width']),  # events
            (batch_size, 64, 128),  # audio
            (batch_size, 1, *MODEL_CONFIG['thermal']['input_size'])  # thermal
        ]
    )
    
    # Setup data loaders
    data_dir = os.path.join('/content/dataset')
    train_dataset = MultiModalDataset(data_dir, split='train')
    val_dataset = MultiModalDataset(data_dir, split='val')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=8, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # TensorBoard
    log_dir = '/content/drive/MyDrive/neuromorphic-adas/logs/experiment_1'
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)
    
    # Training loop
    num_epochs = 50
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in progress_bar:
            # Move data to device
            events = batch['events'].to(device, non_blocking=True)
            audio = batch['audio'].to(device, non_blocking=True)
            thermal = batch['thermal'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = model(events, audio, thermal)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
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
            model_path = os.path.join('/content/drive/MyDrive/neuromorphic-adas/models', 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
            }, model_path)
            print(f'New best model saved with val_loss: {val_loss:.4f}')
        
        print(f'Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
    
    # Save final model
    final_model_path = os.path.join('/content/drive/MyDrive/neuromorphic-adas/models', 'final_model.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
    }, final_model_path)
    
    writer.close()
    print("Training complete!")

if __name__ == "__main__":
    # Set these variables before running
    KAGGLE_JSON_PATH = "/content/drive/MyDrive/kaggle.json"  # Path to your kaggle.json in Google Drive
    DATASET_NAME = "your-username/your-dataset"  # Kaggle dataset name
    
    # Setup environment
    setup_colab()
    setup_kaggle(KAGGLE_JSON_PATH)
    
    # Download dataset
    download_dataset(DATASET_NAME, "/content/dataset")
    
    # Start training
    train()
