
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, input_shape):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv_output_size = self._calculate_conv_output_size(input_shape)

        self.fc1 = nn.Linear(32 * self.conv_output_size, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * self.conv_output_size)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

    def _calculate_conv_output_size(self, input_shape):
        # create a dummy tensor to get the output size of the
        # convolutional layers
        x = torch.zeros((1, 1, *input_shape))
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        return x.view(1, -1).size(1) // 32
