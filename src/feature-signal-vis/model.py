import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomModel(nn.Module):
  """
  4 layer MLP Model with Batch Normalization
  """
  def __init__(self, num_classes=10):
    super(CustomModel, self).__init__()
    self.fc1 = nn.Linear(32 * 32 * 3, 128)
    self.bn1 = nn.BatchNorm1d(128)
    self.fc2 = nn.Linear(128, 64)
    self.bn2 = nn.BatchNorm1d(64)
    self.fc3 = nn.Linear(64, 32)
    self.bn3 = nn.BatchNorm1d(32)
    self.fc4 = nn.Linear(32, num_classes)
    # self.drop = nn.Dropout(0.5)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    x = x.view(x.size(0), -1)
    x = F.relu(self.bn1(self.fc1(x)))
    x = F.relu(self.bn2(self.fc2(x)))
    x = F.relu(self.bn3(self.fc3(x)))
    x = self.fc4(x)
    x = self.softmax(x)
    return x
