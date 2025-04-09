import json
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from VGG_16 import VGG_16

SPOOF_MODEL_PATH = 'vgg_face_spoof_finetuned.pth'
FACE_MODEL_PATH = 'vgg_face_detection_finetuned.pth'
NUM_PEOPLE = 28
PERSON_LABEL_MAP = './person_labels_map.json'
spoof_label_map = {'0': 'Real', '1': 'Spoof'}

spoof_model = VGG_16()
face_model = VGG_16()

# Freeze all layers
for param in face_model.parameters():
    param.requires_grad = False

# Replace and unfreeze fc8 layer for 24-class classification (Total new persons)
face_model.fc8 = nn.Linear(4096, NUM_PEOPLE)
for param in face_model.fc8.parameters():
    param.requires_grad = True
face_model.load_state_dict(torch.load(FACE_MODEL_PATH))
face_model.eval()

# Freeze all layers
for param in spoof_model.parameters():
    param.requires_grad = False

# Replace and unfreeze fc8 layer for 24-class classification (Total new persons)
spoof_model.fc8 = nn.Linear(4096, 2)
for param in spoof_model.fc8.parameters():
    param.requires_grad = True
spoof_model.load_state_dict(torch.load(SPOOF_MODEL_PATH))
spoof_model.eval()

# Open webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2
    roi = frame[cy-300:cy+300, cx-300:cx+300]  # Crop 600x600 square from center

    input_tensor = transform(roi).unsqueeze(0)  # Add batch dimension
    person_file = open(PERSON_LABEL_MAP, 'r')
    person_label_map = json.load(person_file)

    with torch.no_grad():
        spoof_output = spoof_model(input_tensor)
        spoof_pred = torch.argmax(spoof_output, dim=1).item()
        spoof_predicted_label = spoof_label_map[str(spoof_pred)]
        
        face_output = face_model(input_tensor)
        face_pred = torch.argmax(face_output, dim=1).item()
        face_predicted_label = person_label_map[str(face_pred)]

    # Display prediction
    (text2_w, text2_h), _ = cv2.getTextSize(f"Person: {face_predicted_label}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

    cv2.rectangle(frame, (cx-300, cy-300), (cx+300, cy+300), (0, 255, 255), 2)
    
    cv2.rectangle(frame, (cx-300, cy+300), (cx-300 + text2_w + 10, cy+355 + text2_h + 10), (255, 255, 255), -1)

    cv2.putText(frame, f"Prediction: {spoof_predicted_label}", (cx-295, cy+325), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (128, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Person: {face_predicted_label}", (cx-295, cy+355), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (128, 0, 0), 2, cv2.LINE_AA)


    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
