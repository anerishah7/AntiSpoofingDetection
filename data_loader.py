import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class SpoofDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = int(self.data.iloc[idx, 2])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label

# Merge multiple CSVs
csv_file = './data/combined_images.csv'

merged_df = pd.read_csv(csv_file)

# Create dataset and dataloader
dataset = SpoofDataset(merged_df)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)