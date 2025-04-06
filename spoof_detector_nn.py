import torch
import torchfile
import torch.nn as nn
import torch.nn.functional as F

class SpoofNeuralNet(nn.Module):
	def __init__(self, in_channels=3, out_channels=64, num_classes=2):
		super(SpoofNeuralNet, self).__init__()
		# 1 Convolution Layer
		self.conv_1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
		# 1 Fully Connected Layer
		self.fc = nn.Linear(out_channels * 224 * 224, num_classes)
	def forward(self, x):
		x = F.relu(self.conv_1_1(x))
		x = torch.flatten(x, 1)
		x = self.fc(x)
		return x

if __name__ == "__main__":
    model = SpoofNeuralNet()
    
    # (batch_size=1, channels=1, height=28, width=28)
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(output)
    print("Output shape:", output.shape) 
