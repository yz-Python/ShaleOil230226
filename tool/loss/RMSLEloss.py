import torch
import torch.nn as nn
class RMSLEloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(abs(pred + 1)), torch.log(abs(actual + 1))))