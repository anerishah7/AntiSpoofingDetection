import os
import pandas as pd

# Path to your folder of test images
folder_path = "./data/images"

# List all PNG files
image_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]

# Build rows for DataFrame
data = []
for index, file in enumerate(image_files):
    name_parts = file.split("_")
    
    first_name = name_parts[0]  # First Name
    last_name = name_parts[1] # Last Name
    image_id = int(name_parts[2])  # Number id of the image of the person
    label = 0 if "Real" in file else 1  # 0 = real, 1 = spoof

    print(f"Index: {index} File:{file}")

    data.append({
        "image_name": os.path.join(folder_path, file),
        "first_name": first_name,
        "last_name": last_name,
        "image_id": image_id,
        "label": label
    })

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert to csv
# df.to_csv('./data/combined_images.csv', index=False)
# Show the result
print(df)
# print(df.groupby(['first_name', 'last_name']).size().reset_index(name='counts'))
# aneri_df = df.groupby(['first_name', 'last_name']).size().reset_index(name='counts')
aneri_list = ['Shah', 'Solanki', 'Rathod', 'Joshi', 'Bhatt', 'Bhadreshwara']
# train_df = df[df['image_id'] < 3 and df['last_name'] in aneri_list]
train_df = df[(df['image_id'] < 3) & (df['last_name'].isin(aneri_list))]
test_df = df[df['image_id'] == 3]
train_df.to_csv('./data/face_detection_model/train.csv', index=False)
test_df.to_csv('./data/face_detection_model/test.csv', index=False)
# renmae this convert_csv.py