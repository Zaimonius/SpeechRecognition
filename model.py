import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets


#have a big batch size


class SpeechRecognition(nn.Module):
    def __init__(self, input_size=81, dropout=0.1, num_classes=29, hidden_size=1024, num_layers=1):
        super().__init__()
        #define layers
        #in out ?

        #first cnn layer
        #CNNs are used to look for features of data - in my case letters
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, input_size, 10, 2, padding=10//2),
        )
        #dense layers
        self.dense = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
