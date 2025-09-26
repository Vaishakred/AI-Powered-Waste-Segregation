import os
import sqlite3
import torch
import torch.nn as nn
import numpy as np
import cv2

# Same CNN model as in app.py
class WasteClassificationCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(WasteClassificationCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
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

def create_directories():
    """Create folders"""
    folders = ['uploads', 'test_images']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    print("✅ Folders created")

def create_database():
    """Create database"""
    conn = sqlite3.connect('waste_classifications.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS classifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            filename TEXT,
            processing_time REAL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS statistics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE DEFAULT CURRENT_DATE,
            plastic_count INTEGER DEFAULT 0,
            paper_count INTEGER DEFAULT 0,
            metal_count INTEGER DEFAULT 0,
            organic_count INTEGER DEFAULT 0,
            total_classifications INTEGER DEFAULT 0
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database created")

def create_demo_model():
    """Create demo model"""
    model = WasteClassificationCNN(num_classes=4)
    torch.save(model.state_dict(), 'waste_classification_model.pth')
    print("✅ Demo model created")

def create_test_images():
    """Create test images"""
    classes = ['plastic', 'paper', 'metal', 'organic']
    colors = {
        'plastic': (255, 100, 100),  # Red
        'paper': (100, 255, 100),    # Green
        'metal': (100, 100, 255),    # Blue
        'organic': (255, 255, 100)   # Yellow
    }
    
    for class_name in classes:
        img = np.full((224, 224, 3), colors[class_name], dtype=np.uint8)
        
        cv2.putText(img, class_name.upper(), (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        cv2.putText(img, "TEST", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        cv2.imwrite(f'test_images/{class_name}_test.jpg', img)
    
    print("✅ Test images created")

def main():
    print("🚀 Setting up Waste Classification Backend...")
    
    create_directories()
    create_database()
    create_demo_model()
    create_test_images()
    
    print("\n✅ Setup completed!")
    print("\nNext steps:")
    print("1. Install requirements: pip install -r requirements.txt")
    print("2. Run server: python app.py")
    print("3. Test: Open http://localhost:5000/health in browser")

if __name__ == '__main__':
    main()