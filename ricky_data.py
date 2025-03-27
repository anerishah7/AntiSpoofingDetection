import os
import pandas as pd

# Path to your folder of test images
folder_path = "./test_data/ricky_test/images/"

# List all PNG files
image_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]

# Build rows for DataFrame
data = []
for index, file in enumerate(image_files):
    name_parts = file.split("_")
    
    person_name = f"{name_parts[0]}_{name_parts[1]}"  
    label = 0 if "Real" in file else 1  # 0 = real, 1 = spoof (you can adjust logic)

    print(f"Index: {index} File:{file}")

    data.append({
        "image_name": os.path.join(folder_path, file),
        "person_name": person_name,
        "label": label
    })

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert to csv
df.to_csv('./test_data/ricky_test/ricky_images.csv', index=False)
# Show the result
print(df)