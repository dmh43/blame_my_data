import torch
import torch.nn as nn

class TorchLogisticRegression(nn.Module):
  def __init__(self, num_features):
    super().__init__()
    self.lin = nn.Linear(num_features, 1)
    self.lin.weight.data.fill_(0)
    self.lin.bias.data.fill_(0)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    return self.sigmoid(self.lin(x)).squeeze()
