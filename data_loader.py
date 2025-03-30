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
csv_files = [
    'test_data/shailly_image_data/shailly_images.csv',
    'test_data/ricky_test/ricky_images.csv',
    'test_data/Aneri/aneri_images.csv'
]

dataframes = [pd.read_csv(f) for f in csv_files]
merged_df = pd.concat(dataframes, ignore_index=True)

# Create dataset and dataloader
dataset = SpoofDataset(merged_df)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
