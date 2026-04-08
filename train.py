import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset path
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "dataset")
print("Dataset path:", data_dir)

# Separate transforms for train and validation
# Training: with augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Validation: no augmentation
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load full dataset once to get the split indices
full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
print(f"Classes: {full_dataset.classes}")
print(f"Class to index: {full_dataset.class_to_idx}")
print(f"Total images: {len(full_dataset)}")

# Split indices
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset_raw = random_split(full_dataset, [train_size, val_size])

# Create validation dataset with proper (non-augmented) transforms
val_dataset_proper = datasets.ImageFolder(data_dir, transform=val_transform)
val_dataset = Subset(val_dataset_proper, val_dataset_raw.indices)

print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# Load MobileNetV2 with modern weights API
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# Freeze early layers, unfreeze later layers for fine-tuning
# MobileNetV2 has 19 feature blocks (0-18). Unfreeze the last 5 blocks.
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last 5 blocks of the feature extractor
for i in range(14, 19):
    for param in model.features[i].parameters():
        param.requires_grad = True

# Modify classifier with dropout for regularization
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.last_channel, 2)
)

model = model.to(device)

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params:,} / {total_params:,}")

# Loss and optimizer - use different learning rates for different parts
classifier_params = list(model.classifier.parameters())
feature_params = [p for p in model.features.parameters() if p.requires_grad]

optimizer = optim.Adam([
    {'params': feature_params, 'lr': 1e-4},      # Lower LR for pretrained features
    {'params': classifier_params, 'lr': 1e-3},    # Higher LR for new classifier
], weight_decay=1e-4)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)

# Loss function
criterion = nn.CrossEntropyLoss()

# Training loop
epochs = 25
best_val_acc = 0.0
best_model_path = os.path.join(base_dir, "yawn_model.pth")

for epoch in range(epochs):
    # ---- Training ----
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # ---- Validation ----
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data).item()
            val_total += labels.size(0)

    val_loss = val_loss / val_total
    val_acc = val_correct / val_total

    # Step scheduler
    scheduler.step(val_acc)

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} "
          f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"  -> Best model saved (Val Acc: {val_acc:.4f})")

print(f"\nTraining complete. Best Val Accuracy: {best_val_acc:.4f}")
print(f"Model saved as {best_model_path}")