import os
import json
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ClassificationDataSet(Dataset):
    def _init_(self, dataframe, label_col=2, transform=None, label_map_file=None):
        self.data = dataframe.reset_index(drop=True)
        self.label_col = label_col
        self.label_map_file = label_map_file

        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        raw_labels = self.data.iloc[:, self.label_col].astype(str)  # convert all labels to str for consistent mapping

        # Load or create label mapping
        if self.label_map_file and os.path.exists(self.label_map_file):
            with open(self.label_map_file, 'r') as f:
                self.label_mapping = json.load(f)
        else:
            unique_labels = sorted(raw_labels.unique())
            self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            if self.label_map_file:
                with open(self.label_map_file, 'w') as f:
                    json.dump(self.label_mapping, f)

        # Validate labels
        if not set(raw_labels.unique()).issubset(set(self.label_mapping.keys())):
            missing = set(raw_labels.unique()) - set(self.label_mapping.keys())
            raise ValueError(f"CSV contains labels not present in label mapping: {missing}")

        # Store encoded labels in the dataframe
        self.data['encoded_label'] = raw_labels.map(self.label_mapping)

    def _len_(self):
        return len(self.data)

    def _getitem_(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = int(self.data.iloc[idx]['encoded_label'])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label
