import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets

class CNNLayerNorm(nn.Module):
    """CNN layer normalization"""
    def __init__(self, num_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(num_feats)

    def forward(self, x):
        x = x.transpose(2, 3).contiguous() #batch, channel, time, feature
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() #batch, channel, feature, time

class CNN(nn.Module):
    """CNN layer"""
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, num_feats):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(num_feats)
        self.layer_norm2 = CNNLayerNorm(num_feats)

    def forward(self, x):
        residual = x  #batch, channel, feature, time
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x #batch, channel, feature, time

class BiLSTM(nn.Module):
    """RNN Bidirectional LSTM layer"""
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BiLSTM, self).__init__()
        self.BiLSTM = nn.LSTM(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiLSTM(x)
        x = self.dropout(x)
        return x



#have a big batch size

#input:
#n_features - features for the network to look for i.e. letters
#dropout - rate of randomly zeroing out some of the elements ... creates more errors for the network to find solutions for
#num_classes - number of classes
#hidden_size - size of hidden layer
# num_layers - number of lstm layers

class SpeechRecognition(nn.Module):
    def __init__(self, num_cnn_layers=3, num_rnn_layers=5, rnn_dim=512, num_class=29, num_feats=128, stride=2, dropout=0.1):
        super(SpeechRecognition, self).__init__()
        num_feats = num_feats//2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features

        # number of cnn layers with filter size of 32
        self.cnn_layers = nn.Sequential(*[
            CNN(32, 32, kernel=3, stride=1, dropout=dropout, num_feats=num_feats) 
            for _ in range(num_cnn_layers)
        ])
        self.fully_connected = nn.Linear(num_feats*32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BiLSTM(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(num_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2

        self.cnn4 = nn.Conv1d(2 * num_output, 2 * num_output, kernel_size=3)
            nn.Linear(rnn_dim, num_class)
        self.pool4 = nn.MaxPool1d(4)

    def forward(self, x):
        x = self.cnn(x)
        x = self.cnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x
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