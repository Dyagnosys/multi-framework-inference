# model_architectures.py

import torch
import torch.nn as nn
import torchvision.models as models
import logging

logger = logging.getLogger(__name__)

class ResNet50(nn.Module):
    def __init__(self, num_classes=7, channels=3):
        super(ResNet50, self).__init__()
        # Define layers directly without wrapping in 'resnet'
        self.conv_layer_s2_same = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Load pre-trained ResNet50 model
        resnet = models.resnet50(pretrained=True)

        # Extract layers
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # Fully connected layers
        self.fc1 = nn.Linear(resnet.fc.in_features, num_classes)
        # If your model has additional fully connected layers, define them here
        # Example:
        # self.fc2 = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        x = self.conv_layer_s2_same(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # If additional fully connected layers are defined, pass x through them
        # x = self.fc2(x)
        return x

    def extract_features(self, x):
        x = self.conv_layer_s2_same(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class LSTMPyTorch(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMPyTorch, self).__init__()
        self.hidden_size = hidden_size

        # Define separate LSTM layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0_1 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0_1 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out1, _ = self.lstm1(x, (h0_1, c0_1))

        h0_2 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0_2 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out2, _ = self.lstm2(out1, (h0_2, c0_2))

        out = self.fc(out2[:, -1, :])
        return out
