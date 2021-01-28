import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torchvision import datasets
import utils
import model

textprocess = utils.TextProcess()

class Trainer: 
    def __init__(self, file_path):
        print("Cuda : " + str(torch.cuda.is_available()))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.file_path = file_path
        if file_path is not None and io.exists(file_path):
            self.load(file_path)
            self.net.to(self.device)
            self.net.eval()
    
    def pad_sequence(batch):
        # Make all tensor in a batch the same length by padding with zeros
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch.permute(0, 2, 1)

    #------------------------------------------------
    # data order
    # waveform, sample_rate, dic
    #------------------------------------------------
    #collate functions helps the dataloader set labels and data
    def collate_fn(batch):
        data_list, target_list = [], []
        for _data, _, _target in batch:
            data_list.append(_data) #input
            target_list.append(textprocess.text_to_int_sequence(_target)) #target output
        #format data in the end
        return pad_sequence(data_list), torch.stack(target_list)
    
    def load(self):
        #loading neural net
        starter = torch.load(self.file_path)
        self.net = Net(starter['input']) #inputsize
        self.net.load_state_dict(starter['dict']) # dictionary
        self.net.eval()
    
    def save(self):
        #save the neural network
        if self.file_path is not None:
            obj = {'dict': self.net.state_dict(), 'input': self.net.n_features}
            torch.save(obj, self.file_path) #save input size and dictionary

    def train(self, batch_size=64, epochs):
        #setup net
        self.net = SpeechRecognition()
        loss = nn.CTCLoss(blank=28, zero_infinity=True)
        optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        for epoch in epochs:
            print("Epoch #", epoch)
            #setup dataset
            #TODO: set path
            train = torchaudio.datasets.COMMONVOICE(root: pathtodir,version: str = 'cv-corpus-4-2019-12-10', download: bool = False)
            trainset = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            for data in trainset:
                X, y = data  # X is the batch of features, y is the batch of targets.
                self.net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
                output = self.net(X.view(-1, size*size))  # pass in the reshaped batch (recall they are 28x28 atm)
                lossoutput = loss(output, labels, input_lengths, label_lengths)  # calc and grab the loss value #TODO:fix params
                lossoutput.backward()  # apply this loss backwards thru the network's parameters
                optimizer.step()  # attempt to optimize weights to account for loss/gradients
        #after train test



        #setup dataset
        #TODO: set path
        test = torchaudio.datasets.COMMONVOICE(root: pathtodir,version: str = 'cv-corpus-4-2019-12-10', download: bool = False)
        testset = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
