import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets

#have a big batch size

#input:
#n_features - features for the network to look for i.e. letters
#dropout - rate of randomly zeroing out some of the elements ... creates more errors for the network to find solutions for
#num_classes - number of classes
#hidden_size - size of hidden layer
# num_layers - number of lstm layers

class SpeechRecognition(nn.Module):
    def __init__(self, n_features=64, dropout=0.1, num_classes=29, hidden_size=1024, num_layers=1, num_channels=1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        #define layer
        #------------------------------------------------
        #first cnn layer
        #CNNs are used to look for features of data - in my case letters
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, num_channels, 10, 2, padding=10//2),
        )
        #dense layers
        self.dense = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        #RNN layer
        #RNNs are used to process sequence of inputs
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=num_layers, dropout=0.0, bidirectional=False)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.final_fc = nn.Linear(hidden_size, num_classes)

    def _init_hidden(self, batch_size):
        n, hs = self.num_layers, self.hidden_size
        return (torch.zeros(n*1, batch_size, hs),
                torch.zeros(n*1, batch_size, hs))
    
    def forward(self, x, hidden):
        #
        # x = x.squeeze(1)  # batch, feature, time
        #TODO: fix data order in collate/pad function
        x = self.cnn(x) # batch, time, feature
        x = self.dense(x) # batch, time, feature
        x = x.transpose(0, 1) # time, batch, feature
        out, (hn, cn) = self.lstm(x, hidden)
        x = self.dropout2(F.gelu(self.layer_norm2(out)))  # (time, batch, n_class)
        return self.final_fc(x), (hn, cn)