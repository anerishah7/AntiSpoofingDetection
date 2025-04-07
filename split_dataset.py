from sklearn.model_selection import train_test_split
import pandas as pd
from data_loader import SpoofDataset
import data_loader

merged_df = data_loader.merged_df

print("Merged DataFrame columns:", merged_df.columns)
print("Sample data:\n", merged_df.head())

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

'''
Task to do
spit out 
file 1: data/spoof_nn/
    train1.csv (people in train cant be in test) 70% train2.csv
    test1.csv 30% test2.csv
file 2: data/detect_nn/
    train1.csv (5 images)
    test1.csv  (1 image)
    doesnt matter how you split real or fake
'''