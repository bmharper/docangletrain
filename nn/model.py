import torch
import torch.nn as nn
import torch.nn.functional as F


class TextAngleClassifier(nn.Module):

    def __init__(self):
        super(TextAngleClassifier, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, groups=4)
        self.bn2 = nn.BatchNorm2d(64)
        # Max pooling layer: 2x2 kernel, stride 2
        self.pool = nn.MaxPool2d(2, stride=2)
        # Global average pooling to reduce to 1x1 spatial size
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully connected layer: 4 output classes
        self.fc = nn.Linear(64, 4)

    def forward(self, x):
        # Input shape: (batch_size, 1, 32, 32)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Output: (batch_size, 16, 16, 16)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Output: (batch_size, 64, 8, 8)
        x = self.global_pool(x)  # Output: (batch_size, 64, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 64)
        x = self.fc(x)  # Output: (batch_size, 4)
        return x  # Returns logits for cross-entropy loss
