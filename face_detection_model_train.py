import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data_loader import ClassificationDataSet
from VGG_16 import VGG_16
import torch.nn.functional as F

TRAIN_CSV = './data/face_detection_model/train1.csv'
TEST_CSV = './data/face_detection_model/test1.csv'
WEIGHTS_PATH = './VGG_FACE.t7'
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = './vgg_face_detection_finetuned.pth'
LABEL_MAP_FILE = './person_labels_map.json'
NUM_PEOPLE = 28

transform = transforms.Compose([
 	transforms.Resize((224, 224)),
 	transforms.ToTensor(),
])
 
def imLoad(image_name):
    image = Image.open(image_name).convert('RGB')
    image = transform(image).unsqueeze(0)  
    return image

# -----------------------------
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

# -----------------------------
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item() * images.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_script(model, criterion, optimizer, device):
    # Load CSVs
    train_df = pd.read_csv(TRAIN_CSV)
    # Dataset and Dataloaders
    train_dataset = ClassificationDataSet(train_df, label_col=3, transform=transform, label_map_file=LABEL_MAP_FILE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | ")

    # Save the model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
def test_script(model, criterion, device):
    test_df = pd.read_csv(TEST_CSV)
    test_dataset = ClassificationDataSet(test_df)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
	# Training loop
    for epoch in range(NUM_EPOCHS):
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
    

# -----------------------------
def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and weights
    model = VGG_16()
    model.load_weights(WEIGHTS_PATH)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace and unfreeze fc8 layer for 24-class classification (Total new persons)
    model.fc8 = nn.Linear(4096, NUM_PEOPLE)
    for param in model.fc8.parameters():
        param.requires_grad = True
        
    model.to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc8.parameters(), lr=LEARNING_RATE)
    
	# Training Script
    # train_script(model, criterion, optimizer, device)
    # Testing Script
    # test_script(model, criterion, device)
    
    
    #############################################################
    # Uncomment to Test single image
    #############################################################
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    x=imLoad('./data/images/Aneri_Shah_3_Real.png')
    probabilities = F.softmax(model(x), dim=1)  # if x is batched
    predicted_class = probabilities.argmax(dim=1)
    print(predicted_class)
    

if __name__ == "__main__":
    main()