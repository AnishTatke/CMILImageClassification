import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from dataset import GlomeruliDataset, load_data
from model import SimpleCNN
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import v2
from PIL import Image

# Define the training function
def train(epoch, model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(loader):
        images, labels = images.float().to(device), labels.float().to(device)
        
        outputs = model(images)
        outputs = outputs.squeeze()
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = torch.round(F.sigmoid(outputs))
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    running_loss /= len(loader)
    accuracy = correct / total * 100
    print(f"Train Epoch {epoch}: Loss {running_loss:.3f},\t Accuracy: {accuracy:.2f}%")

# Define the validation function
@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(loader):
        images, labels = images.float().to(device), labels.float().to(device)
        outputs = model(images)
        outputs.squeeze_()
        loss = criterion(outputs, labels)

        running_loss += loss.item()

        predicted = torch.round(F.sigmoid(outputs))
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    running_loss /= len(loader)
    accuracy = correct / total * 100
    print(f"Validation: Loss {running_loss:.3f},\t Accuracy: {accuracy:.2f}%")

# Set the device, batch size, learning rate, and number of epochs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
LEARNING_RATE = 2e-3
NUM_EPOCHS = 10
print(f"Running on {device}")

# Load the data
if not os.path.exists('data/train.csv') and not os.path.exists('data/validate.csv') and not os.path.exists('data/rescaling_config.json'):
    load_data('public/public.csv')

train_df = pd.read_csv('data/train.csv')
validate_df = pd.read_csv('data/validate.csv')
config = json.load(open('data/rescaling_config.json', 'r'))

# Define the transformations
transforms = v2.Compose([
    v2.ToTensor(),
    v2.Resize((config['mean_width'], config['mean_height'])),
    v2.Normalize(mean = [0.5 for _ in range(4)], std = [0.5 for _ in range(4)]),
    v2.ToTensor(),
])

# Create the datasets
train_dataset = GlomeruliDataset(train_df, transform=transforms)
validate_dataset = GlomeruliDataset(validate_df, transform=transforms)

labels = train_dataset.labels
class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
class_weight = 100. / class_sample_count
weights = np.array([class_weight[t] for t in labels])
sampler = WeightedRandomSampler(weights=torch.from_numpy(weights), num_samples=len(train_dataset), replacement=True)

# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
validate_loader = DataLoader(validate_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize the model, loss function, and optimizer
model = SimpleCNN().to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weight[1], dtype=torch.float64))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

torch.cuda.empty_cache()

# Train the model
for epoch in range(NUM_EPOCHS):
    train(epoch, model, train_loader, criterion, optimizer)
validate(model, validate_loader, criterion)

# Save the model
torch.save(model.state_dict(), 'models/model.pth')


