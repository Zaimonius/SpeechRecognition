import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets


#have a big batch size


class SpeechRecognition(nn.Module):
    def __init__(self,input):
        super().__init__()
        #define layers
        #in out ?

        #first cnn layer
        conv = nn.Conv1d(in_channel, out_channel, groups=1, bias=True, kernel_size=2, padding=0, stride=1)
        self.cnn = nn.Sequential(
            nn.Conv1d()
        )
        
