import torch
import torch.nn as nn
import torch.nn.functional as F

class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=4, stride=1, padding =1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv21 = nn.Conv1d(32, 64, kernel_size=4, stride=1, padding =2)
        self.bn21 = nn.BatchNorm1d(64)
        self.relu = nn.Sigmoid()
        self.maxpool = nn.AvgPool1d(stride=2, kernel_size=2)


    def forward(self, x, is_deconv=False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.maxpool(self.relu(self.bn21(self.conv21(x))))
        return x

class Predictor(nn.Module):
    def __init__(self, prob=0.5):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(64*300, 1000)
        self.bn1_fc = nn.BatchNorm1d(1000)
        self.fc3 = nn.Linear(1000, 3)
        self.bn_fc3 = nn.BatchNorm1d(3)
        self.relu = nn.ReLU()
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        x = x.view(x.size(0), 64*300)
        x = F.dropout(x, training=self.training, p=self.prob)
        x = self.relu(self.bn1_fc(self.fc1(x)))
        x = self.fc3(x)
        return x
