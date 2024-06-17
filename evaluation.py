import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from PIL import Image
from dataset import GlomeruliDataset
from model import SimpleCNN
from torchvision.transforms import v2
from torch.utils.data import DataLoader

# Define the load_data function
def load_data(path):
    img_files = os.listdir(path)
    paths = [os.path.join(path, img_file) for img_file in img_files]
    df = pd.DataFrame({'name': img_files, 'path': paths})
    return df

# Define the TestGlomeruliDataset class
class TestGlomeruliDataset(GlomeruliDataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __getitem__(self, idx):
        name = self.df.iloc[idx, 0]
        path = self.df.iloc[idx, 1]
        image = np.asarray(Image.open(path))
        if self.transform:
            image = self.transform(image)
        return image, name

# Define the evaluate function
@torch.no_grad()
def evaluate(model, loader):
    df = loader.dataset.df
    model.eval()
    for i, (images, img_names) in enumerate(loader):
        images = images.float().to(device)
        outputs = model(images)
        outputs.squeeze_(1)
        predicted = torch.round(F.sigmoid(outputs))
        for j, name in enumerate(img_names):
            df.loc[df['name'] == name, 'prediction'] = int(predicted[j].item())
    return df

# Define the main function
if __name__ == '__main__':
    path = sys.argv[1]
    if os.path.isdir(path):
        # Load the test data
        test = load_data(path)
    else:
        print("Invalid path")

    # Define the device and batch size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 32
    config = json.load(open('data/rescaling_config.json', 'r'))

    # Define the transformations
    transforms = v2.Compose([
        v2.ToTensor(),
        v2.Resize((config['mean_width'], config['mean_height'])),
        v2.Normalize(mean = [0.5 for _ in range(4)], std = [0.5 for _ in range(4)]),
        v2.ToTensor(),
    ])

    # Create the test dataset
    test_dataset = TestGlomeruliDataset(test, transform=transforms)

    # Create the test loader
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load the model
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load('models/model.pth'))

    # Evaluate the model
    test = evaluate(model, test_loader)
    test.drop('path', axis=1, inplace=True)

    # Save the outputs in csv
    test.to_csv('evaluation.csv', index=False)
