from torchvision.transforms import v2
from torch.utils.data import Dataset
from PIL import Image
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the train_validate_split function
def train_validate_split(df, split = 0.8, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    split_end = int(split * m)
    train = df.iloc[perm[:split_end]]
    validate = df.iloc[perm[split_end:]]
    train.reset_index(inplace=True)
    train.drop('index', axis=1, inplace=True)
    validate.reset_index(inplace=True)
    validate.drop('index', axis=1, inplace=True)
    return train, validate

# Define the load_data function
def load_data(path):
    df = pd.read_csv(path)
    folderClasses = ['non_globally_sclerotic_glomeruli', 'globally_sclerotic_glomeruli']
    df['path'] = df.apply(lambda x: f"public/{folderClasses[x['ground truth']]}/{x['name']}", axis=1)

    sizes = None
    for path in df['path']:
        img = np.asarray(plt.imread(path))
        if sizes is None:
            sizes = [img.shape]
        else:
            sizes.append(img.shape)

    config = {
        'mean_width': round(np.mean([w for h, w, c in sizes])),
        'mean_height': round(np.mean([h for h, w, c in sizes]))
    }
    json.dump(config, open('data/rescaling_config.json', 'w'))

    train, validate = train_validate_split(df, seed=42)
    train.to_csv('data/train.csv', index=False)
    validate.to_csv('data/validate.csv', index=False)


# Define the GlomeruliDataset class
class GlomeruliDataset(Dataset):
    folderClasses = ['non_globally_sclerotic_glomeruli', 'globally_sclerotic_glomeruli']
    def __init__(self, df, format=None, transform=None):
        self.df = df
        self.format = format
        self.labels = df['ground truth'].to_numpy().astype(np.int32)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.labels[idx]
        path = self.df.iloc[idx, 2]
        if self.format == "RGB":
            image = np.asarray(Image.open(path).convert('RGB'))
        else:
            image = np.asarray(Image.open(path))

        if self.transform:
            image = self.transform(image)

        return image, label