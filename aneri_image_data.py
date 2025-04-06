import os
import pandas as pd

training_data_path = './test_data/Aneri/images/'
training_data_dir = os.listdir(training_data_path)

training_list = []
for image in training_data_dir:
    if image.endswith('Real.png'):
        training_list.append([training_data_path + image, image[:-11],0])
    elif image.endswith('Spoof.png'):
        training_list.append([training_data_path + image, image[:-12],1])

train_data = pd.DataFrame(training_list, columns=['image_name', 'person_name', 'label'])
# print(train_data)

# Umcomment to store the images in .csv fomat
train_data.to_csv('./test_data/Aneri/aneri_images.csv', index=False)
                                                                                                                                                                                                                         
