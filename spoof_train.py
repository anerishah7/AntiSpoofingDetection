import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from data_loader import ClassificationDataSet
from predict_faces_vgg import VGG_16

# -----------------------------
# Config
TRAIN_CSV = 'test_data/Aneri/aneri_images.csv'
TEST_CSV = 'test_data/ricky_test/ricky_images.csv'
WEIGHTS_PATH = 'VGG_FACE.t7'
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = 'vgg_face_spoof_finetuned.pth'

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

# -----------------------------
def main():
    # Load CSVs
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # Dataset and Dataloaders
    train_dataset = ClassificationDataSet(train_df, label_col=2, label_map_file="label_map.json")
    test_dataset = ClassificationDataSet(test_df, label_col=2, label_map_file="label_map.json")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and weights
    model = VGG_16()
    model.load_weights(WEIGHTS_PATH)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace and unfreeze fc8 layer for 2-class classification
    model.fc8 = nn.Linear(4096, 2)
    for param in model.fc8.parameters():
        param.requires_grad = True

    model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc8.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    # Save the model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
