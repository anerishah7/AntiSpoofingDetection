'''
1. Use the data frame from data_loader
2. Use sklearn to split the data into train.csv and test.csv
'''
from sklearn.model_selection import train_test_split
import pandas as pd
from data_loader import SpoofDataset

import data_loader
merged_df = data_loader.merged_df

# Split into 80% train, 20% test
train_df, test_df = train_test_split(merged_df, test_size=0.2, random_state=42, stratify=merged_df['label'])

# Save to CSVs
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

print("Train/Test split complete.")