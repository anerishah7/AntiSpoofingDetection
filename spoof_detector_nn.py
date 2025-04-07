# import torch
# import torchfile
# import torch.nn as nn
# from torchvision import transforms
# import torch.nn.functional as F
# from PIL import Image
# from torch.autograd import Variable
# from torchvision import datasets, models, transforms
# import os

# class VGG_16(nn.Module):
# 	def __init__(self):
# 		"""
# 		Constructor
# 		"""
# 		super().__init__()
# 		self.block_size = [2, 2, 3, 3, 3]
# 		self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
# 		self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
# 		self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
# 		self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
# 		self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
# 		self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
# 		self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
# 		self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
# 		self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
# 		self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
# 		self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
# 		self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
# 		self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
# 		self.fc6 = nn.Linear(512 * 7 * 7, 4096)
# 		self.fc7 = nn.Linear(4096, 4096)
# 		self.fc8 = nn.Linear(4096, 2622)
# 	def load_weights(self, path):
# 		model = torchfile.load(path)
# 		counter = 1
# 		block = 1
# 		for i, layer in enumerate(model.modules):
# 			if layer.weight is not None:
# 				if block <= 5:
# 					self_layer = getattr(self, "conv_%d_%d" % (block, counter))
# 					counter += 1
# 					if counter > self.block_size[block - 1]:
# 						counter = 1
# 						block += 1
# 					self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
# 					self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
# 				else:
# 					self_layer = getattr(self, "fc%d" % (block))
# 					block += 1
# 					self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
# 					self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
# 	def forward(self, x):
# 		x = F.relu(self.conv_1_1(x))
# 		x = F.relu(self.conv_1_2(x))
# 		x = F.max_pool2d(x, 2, 2)
# 		x = F.relu(self.conv_2_1(x))
# 		x = F.relu(self.conv_2_2(x))
# 		x = F.max_pool2d(x, 2, 2)
# 		x = F.relu(self.conv_3_1(x))
# 		x = F.relu(self.conv_3_2(x))
# 		x = F.relu(self.conv_3_3(x))
# 		x = F.max_pool2d(x, 2, 2)
# 		x = F.relu(self.conv_4_1(x))
# 		x = F.relu(self.conv_4_2(x))
# 		x = F.relu(self.conv_4_3(x))
# 		x = F.max_pool2d(x, 2, 2)
# 		x = F.relu(self.conv_5_1(x))
# 		x = F.relu(self.conv_5_2(x))
# 		x = F.relu(self.conv_5_3(x))
# 		x = F.max_pool2d(x, 2, 2)
# 		x = x.view(x.size(0), -1)
# 		x = F.relu(self.fc6(x))
# 		x = F.dropout(x, 0.5, self.training)
# 		x = F.relu(self.fc7(x))
# 		x = F.dropout(x, 0.5, self.training)
# 		return self.fc8(x)

# preprocess = transforms.Compose([
#     # transforms.ToTensor(),
#     # transforms.Lambda(lambda x: x * 255),  # Scale to [0, 255]
#     # transforms.Normalize(mean=mean, std=[1.0, 1.0, 1.0])
# 	transforms.Resize((224, 224)),
# 	transforms.ToTensor(),
# 	#transforms.Normalize(mean = [129.1863, 104.7624, 93.5940])
# ])

# TRAIN = 'train'
# TEST = 'test'
# image_dir = './data/Aneri/images/'

# data_transforms = {
# 	TRAIN: transforms.Compose([
# 		transforms.Resize((224, 224)),
# 		transforms.ToTensor(),
# 		#transforms.Normalize(mean = [129.1863, 104.7624, 93.5940])
# 	]),
# 	TEST: transforms.Compose([
# 		transforms.Resize((224, 224)),
# 		transforms.ToTensor(),
# 		#transforms.Normalize(mean = [129.1863, 104.7624, 93.5940])
# 	])
# }

# def imLoad(image_name):
# 	image = Image.open(image_name).convert('RGB')
# 	# image = data_transforms[TEST](image)
# 	image = preprocess(image)
# 	image = image.unsqueeze(0)
# 	# temp_image = transforms.ToPILImage()(image)
# 	# temp_image.save('processed_image.png')
	
# 	return image

# use_gpu = torch.cuda.is_available()

# img_datasets = { x: datasets.ImageFolder(os.path.join(image_dir, x), transform=data_transforms[x]) for x in [TRAIN, TEST] }
# img_dataloaders = { x: torch.utils.data.DataLoader(img_datasets[x], batch_size=8, shuffle=True, num_workers=0)  for x in [TRAIN, TEST]  }

# # Load the model and weights
# torch_model = VGG_16()
# torch_model.load_weights("VGG_FACE.t7")

# ##########################################################################
# # 								CUSTOM LAYERS							 #
# ##########################################################################
# # Freeze weights
# for param in torch_model.parameters():
# 	param.requires_grad=False
# num_features = torch_model._modules['fc8'].in_features
# torch_model._modules['fc8']=nn.Linear(num_features, 1)

# if use_gpu:
# 	torch_model.cuda() 
	
# # criterion = nn.BCEWithLogitsLoss()
# # optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.001)

# ##########################################################################
# # 								TRAINING!!!!							 #
# ##########################################################################
# # 100 epochs
# # for epoch in range(20):
# # 	running_loss = 0.0
# # 	for i, data in enumerate(img_dataloaders[TRAIN]):
# # 		torch_model.train()
# # 		inputs, labels = data
# # 		if use_gpu:
# # 			inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
# # 		optimizer.zero_grad(set_to_none=True)
# # 		outputs = torch_model(inputs).squeeze()
# # 		loss = criterion(outputs, labels.float())
# # 		loss.backward()
# # 		optimizer.step()

# # 		running_loss += loss.item()
# # 		print('[%d, %5d] loss: %.3f' %
# # 				(epoch + 1, i + 1, running_loss / 2))
# # 		running_loss = 0.0

# # print('Finished Training Model')

# # torch.save(torch_model.state_dict(), "./Spoof_Detection_VGG.pth")


# ##########################################################################
# # 								LOAD SAVED								 #
# ##########################################################################

# torch_model.load_state_dict(torch.load("./Spoof_Detection_VGG.pth"))

# torch_model.train(False)
# for param in torch_model.parameters():
# 	param.requires_grad=False

# correct = 0
# total = 0
# with torch.no_grad():
# 	for i, data in enumerate(img_dataloaders[TEST]):
# 		images, labels = data
# 		if use_gpu:
# 			images, labels = Variable(images.cuda()), Variable(labels.cuda())
# 		outputs = torch_model(images)
# 		_, predicted = torch.max(outputs.data, 1)
# 		#print(dataloaders[TEST].dataset.samples[i], predicted)
# 		total += labels.size(0)
# 		correct += (predicted == labels).sum().item()

# print('Accuracy of the network on the images: %d %%' % (100 * correct / total))


# # x=imLoad('./test_data/akki.png')
# # # probabilities = F.softmax(torch_model(x))
# # probabilities = F.softmax(torch_model(x), dim=1)  # if x is batched
# # # print(probabilities)
# # predicted_class = probabilities.argmax(dim=1)
# # print(predicted_class)


import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_loader import SpoofDataset
from VGG_16 import VGG_16, imLoad

# -----------------------------
# Config
TRAIN_CSV = './data/face_detection_model/train.csv'
TEST_CSV = './data/face_detection_model/test.csv'
WEIGHTS_PATH = './VGG_FACE.t7'
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = './vgg_face_detection_finetuned.pth'

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
    # train_df = pd.read_csv(TRAIN_CSV)
    # test_df = pd.read_csv(TEST_CSV)

    # Dataset and Dataloaders
    # train_dataset = SpoofDataset(train_df)
    # test_dataset = SpoofDataset(test_df)
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and weights
    model = VGG_16()
    model.load_weights(WEIGHTS_PATH)

    # # Freeze all layers
    # for param in model.parameters():
    #     param.requires_grad = False

    # # Replace and unfreeze fc8 layer for 24-class classification (Total new persons)
    model.fc8 = nn.Linear(4096, 24)
    # for param in model.fc8.parameters():
    #     param.requires_grad = True

    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.to(device)
    x=imLoad('./data/images/Aneri_Shah_3_Real.png')
	# probabilities = F.softmax(torch_model(x))
    probabilities = F.softmax(model(x), dim=1)  # if x is batched
	# print(probabilities)
    predicted_class = probabilities.argmax(dim=1)
    print(predicted_class)

    # Loss and Optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.fc8.parameters(), lr=LEARNING_RATE)

    # Training loop
    # for epoch in range(NUM_EPOCHS):
    #     # train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    #     test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    #     print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
    #         #   f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | ")
    #           f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    # Save the model
    # torch.save(model.state_dict(), MODEL_SAVE_PATH)
    # print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
