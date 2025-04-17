# ImageSpoofDetection
##############################################################################################
Authors : Aneri Shah, Ricky Lee, Shailly Bhati
##############################################################################################



VGG_16 Pre-trained model:
- 

Place VGG_FACE.t7 trained model from http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_torch.tar.gz ([1] O. M. Parkhi, A. Vedaldi, A. Zisserman, Deep Face Recognition, British Machine Vision Conference, 2015). Ensure that you place the VGG_FACE.t7 file in the root directory of this project.


Files and Description:
- 
VGG_16.py - VGG16 model architecture 

convert_csv.py - converts all images in data folder to csv file.

split_dataset.py - split csv file in train and test split csv files.

spoof_train.py, face_detection_model_train.py - train and test spoof and face detction models.

real_time_testing.py - main file with real time testing of models.


Dataset Description:
- 
We built 28 people face dataset with 3 real and 3 spoof images for each person. So, we hvae total 168 images data.


Output example from our testing:
- 
https://www.youtube.com/watch?v=06R9K7RN9Uk


To run this:
- 
1. Clone the repository:

git clone https://github.com/anerishah7/AntiSpoofingDetection.git

2. Install dependencies:

pip install -r requirements.txt

3. Download VGG_FACE.t7 from torch version of VGG models hosted by oxford:

http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_torch.tar.gz
Place it in project root folder.

4. Run training model files.

python3 spoof_train.py
python3 face_detection_model_train.py

5. For real-time testing, run the following file:

python3 real_time_testing.py


##############################################################################################
17th April, 2025