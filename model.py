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
    def __init__(self, num_input=1, num_output=128, dropout=0.1, hidden_size=1024, num_layers=1, num_channels=32):
        super(SpeechRecognition, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = num_output
        #define layer
        #------------------------------------------------
        #first cnn layer
        #CNNs are used to look for features of data - in my case letters
        self.cnn1 = nn.Conv1d(num_input, num_output, kernel_size=80, stride=16)
        self.bn1 = nn.BatchNorm1d(num_output)
        self.pool1 = nn.MaxPool1d(4)

        self.cnn2 = nn.Conv1d(num_output, num_output, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(num_output)
        self.pool2 = nn.MaxPool1d(4)

        self.cnn3 = nn.Conv1d(num_output, 2 * num_output, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * num_output)
        self.pool3 = nn.MaxPool1d(4)

        self.cnn4 = nn.Conv1d(2 * num_output, 2 * num_output, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * num_output)
        self.pool4 = nn.MaxPool1d(4)

        self.fc1 = nn.Linear(2 * num_output, num_output)
        #BRNN layer
        #BRNNs are used to process sequence of inputs
        #self.lstm = nn.LSTM(input_size=num_output, hidden_size=hidden_size, num_layers=num_layers, dropout=0.0, bidirectional=True, batch_first=True)
        #self.final_fc = nn.Linear(2 * hidden_size, num_output)

    def _init_hidden(self, batch_size):
        n, hs = self.num_layers, self.hidden_size
        return (torch.zeros(n*1, batch_size, hs),
                torch.zeros(n*1, batch_size, hs))
    
    def forward(self, x, hidden):
        #TODO: fix data order in collate/pad function
        #CNNs
        x = self.cnn1(x) # batch, feature, sequence length
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.cnn2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.cnn3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.cnn4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        #BRNNs
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.contiguous().transpose(1,2) #TODO: fix continguous copy stuff
        x = self.fc1(x)
        #print(x.shape)
        #out, _states = self.lstm(x) #batch size, sequence length, features
        #print(x.shape)
        #print(out.shape)
        #out = self.final_fc(out)
        #print(out.shape)
        return F.log_softmax(x, dim=2)#F.log_softmax(out, dim=2)
        # out, (hn, cn) = self.lstm(x, hidden)
        # x = self.dropout2(F.gelu(self.layer_norm2(out)))  # (time, batch, n_class)
        # return self.final_fc(x), (hn, cn)