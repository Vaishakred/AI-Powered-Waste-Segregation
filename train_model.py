import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import argparse

# Same CNN model as in the main app
class WasteClassificationCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(WasteClassificationCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First Conv Block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second Conv Block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third Conv Block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth Conv Block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class WasteDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['plastic', 'paper', 'metal', 'organic']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        self.load_samples()
        
    def load_samples(self):
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.exists(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        filepath = os.path.join(class_dir, filename)
                        self.samples.append((filepath, self.class_to_idx[class_name]))
        
        print(f"Loaded {len(self.samples)} samples")
        
        # Print class distribution
        class_counts = {}
        for _, label in self.samples:
            class_name = self.classes[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print("Class distribution:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        image = Image.open(filepath).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms():
    """Get data transforms for training and validation"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """Train the CNN model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    print(f"Training on {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f'  New best validation accuracy: {val_acc:.2f}%')
        
        scheduler.step()
        print(f'  Learning rate: {scheduler.get_last_lr()[0]:.6f}')
        print('-' * 50)
    
    # Load best model state
    model.load_state_dict(best_model_state)
    
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_val_accuracy': best_val_acc
    }
    
    return model, training_history

def evaluate_model(model, test_loader, classes):
    """Evaluate model performance"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    report = classification_report(all_targets, all_predictions, 
                                 target_names=classes, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    return accuracy, report, cm

def plot_training_history(history, save_path='training_history.png'):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['val_losses'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_accuracies'], label='Train Accuracy')
    ax2.plot(history['val_accuracies'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")

def plot_confusion_matrix(cm, classes, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Train Waste Classification CNN')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for saving model and plots')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = WasteDataset(
        os.path.join(args.data_dir, 'train'), 
        transform=train_transform
    )
    val_dataset = WasteDataset(
        os.path.join(args.data_dir, 'val'), 
        transform=val_transform
    )
    test_dataset = WasteDataset(
        os.path.join(args.data_dir, 'test'), 
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=4)
    
    # Initialize model
    print("Initializing model...")
    model = WasteClassificationCNN(num_classes=4)
    
    # Train model
    print("Starting training...")
    model, history = train_model(model, train_loader, val_loader, 
                                args.epochs, args.lr)
    
    # Save model
    model_path = os.path.join(args.output_dir, 'waste_classification_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_accuracy, test_report, cm = evaluate_model(model, test_loader, train_dataset.classes)
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report([], [], target_names=train_dataset.classes, 
                               labels=list(range(len(train_dataset.classes)))))
    
    # Save results
    results = {
        'test_accuracy': test_accuracy,
        'classification_report': test_report,
        'training_history': history,
        'model_parameters': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr
        },
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = os.path.join(args.output_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    # Plot training history
    plot_training_history(history, 
                         os.path.join(args.output_dir, 'training_history.png'))
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, train_dataset.classes,
                         os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    print("Training completed successfully!")

if __name__ == '__main__':
    main()