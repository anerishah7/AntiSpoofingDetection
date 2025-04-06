'''
1. Use the data frame from data_loader
2. Use sklearn to split the data into train.csv and test.csv

New obj:
make 70/30 split
Split 1: People in training split should not be in test split
Split 2: People in training split can be in test split
'''
from sklearn.model_selection import train_test_split
import pandas as pd
from data_loader import SpoofDataset
import data_loader

merged_df = data_loader.merged_df

unique_people = merged_df['person_name'].unique()

train_people, test_people = train_test_split(unique_people, test_size=0.3, random_state=42)

# Split 1: People in training split should not be in test split
train_df, test_df = train_test_split(merged_df, test_size=0.3, random_state=42, stratify=merged_df['label'])
split1_train_df = merged_df[merged_df['person_name'].isin(train_people)]
split1_test_df = merged_df[merged_df['person_name'].isin(test_people)]

# Save Split 1 to CSVs
split1_train_df.to_csv('split1_train.csv', index=False)
split1_test_df.to_csv('split1_test.csv', index=False)
print("Split 1 complete (no overlapping people).")

# Split 2: People in training split can be in test split
split2_train_df, split2_test_df = train_test_split(merged_df, test_size=0.3, random_state=42, stratify=merged_df['label'])

# Save split 2 to CSV's
split2_train_df.to_csv('split2_train.csv', index=False)
split2_test_df.to_csv('split2_test.csv', index=False)

print("Split 2 complete (people may overlap).")