from sklearn.model_selection import train_test_split
import pandas as pd
# from data_loader import SpoofDataset
import data_loader

csv_file = './data/combined_images.csv'
merged_df = pd.read_csv(csv_file)

# create identity so all pics of x person falls under one
merged_df['person_id'] = merged_df['first_name'] + '_' + merged_df['last_name']
unique_people = merged_df['person_id'].unique()

# Split 1: People in training split should not be in test split
train_people, test_people = train_test_split(unique_people, test_size=0.3, random_state=42)
split1_train_df = merged_df[merged_df['person_id'].isin(train_people)]
split1_test_df = merged_df[merged_df['person_id'].isin(test_people)]

# Save Split 1 to CSVs
split1_train_df.to_csv('./data/spoof_model/split1_train.csv', index=False)
split1_test_df.to_csv('./data/spoof_model/split1_test.csv', index=False)
print("Split 1 complete (no overlapping people).")

# Split 2: People in training split can be in test split
split2_train_df, split2_test_df = train_test_split(merged_df, test_size=0.3, random_state=42, stratify=merged_df['label'])

# Save split 2 to CSV's
split2_train_df.to_csv('./data/spoof_model/split2_train.csv', index=False)
split2_test_df.to_csv('./data/spoof_model/split2_test.csv', index=False)
print("Split 2 complete (people may overlap).")

# Split 3: 1 image per person in test, rest in train
train_split3 = []
test_split3 = []

for person in merged_df['person_id'].unique():
    person_df = merged_df[merged_df['person_id'] == person]
    if len(person_df) >= 2:
        sampled = person_df.sample(frac=1, random_state=42)
        test_split3.append(sampled.iloc[0])
        train_split3.append(sampled.iloc[1:])
    else:
        print(f"Skipping {person} (not enough images)")

split3_train_df = pd.concat(train_split3)
split3_test_df = pd.DataFrame(test_split3)

split3_train_df.to_csv('./data/face_detection_model/train1.csv', index=False)
split3_test_df.to_csv('./data/face_detection_model/test1.csv', index=False)
print("Split 3 complete (1 image per person in test, rest in train).")