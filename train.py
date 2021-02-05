import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torchvision import datasets
import os.path
from os import path
import utils
import model
from itertools import repeat
from tqdm import tqdm

textprocess = utils.TextProcess()

class Trainer: 
    def __init__(self, file_path, epochs, batch_size=16):
        print("Cuda : " + str(torch.cuda.is_available()))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #set cpu or gpu
        self.file_path = file_path
        self.net = model.SpeechRecognition()
        self.net.train()
        if file_path is not None and path.exists(file_path):
            self.load()
            self.net.to(self.device)
            self.net.train()
        self.train(epochs=epochs, batch_size=batch_size)

    #------------------------------------------------
    # data order
    # waveform, sample_rate, dic[sentence, person id, etc.]
    #------------------------------------------------
    #collate functions helps the dataloader set labels and data
    def collate_fn(self, batch):
        data_list, target_list = [], []
        for _data, _, _target in batch:
            #data = _data.t().tolist()
            data_list.append(torch.Tensor(_data.t().tolist())) #input
            text = textprocess.text_to_int_sequence(_target['sentence'].lower())
            text = text + list(repeat(0, 128 - len(text)))
            text = text[:128]
            target_list.append(torch.Tensor(text)) #target output
        #format data in the end
        return torch.nn.utils.rnn.pad_sequence(data_list, batch_first=True, padding_value=0.).transpose(1,2), torch.nn.utils.rnn.pad_sequence(target_list, batch_first=True, padding_value=0.)
    
    def load(self):
        #loading neural net
        self.net.load_state_dict(torch.load(self.file_path)) # dictionary
        print("loaded neural network file: " + self.file_path)
    
    def save(self):
        #save the neural network
        print("saved file")
        if self.file_path is not None:
            torch.save(self.net.state_dict(), self.file_path) #save input size and dictionary

    def train(self, epochs, batch_size=16):
        #setup net
        if torch.cuda.is_available():
            self.net.cuda()
        optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10
        for epoch in range(0,epochs):
            #setup dataset
            train = torchaudio.datasets.COMMONVOICE(root='/media/gussim/SlaveDisk/MCV',version= 'cv-corpus-6.1-2020-12-11', download = False)
            trainset = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
            trainset_len = len(trainset)
            i = 0
            j = 1
            for data, target in tqdm(trainset, desc="Epoch #"+str(epoch)):
                data, target = data.to(self.device), target.to(self.device) # data is the batch of features, target is the batch of targets.
                self.net.zero_grad()                                        # sets gradients to 0 before loss calc. You will do this likely every step.
                hidden = self.net._init_hidden(batch_size=batch_size)
                output = self.net(data,hidden).squeeze()                           # pass in the reshaped batch
                target = target.long()
                if i > trainset_len-2:
                    break
                loss = F.nll_loss(output, target)
                loss.backward()                                             # apply this loss backwards thru the network's parameters
                optimizer.step()                                            # attempt to optimize weights to account for loss/gradients
                i = i + 1
                if((i/trainset_len)*100) > j:
                    j = j + 1
                    self.save()
            #save here every epoch
            self.save()
            #after train test the model
            self.test(batch_size=batch_size)

    def test(self,batch_size=16):
        self.net.eval()
        correct = 0
        test = torchaudio.datasets.COMMONVOICE(root='/media/gussim/SlaveDisk/MCV',version= 'cv-corpus-6.1-2020-12-11', download = False)
        testset = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        for data, target in testset:
            data, target = data.to(self.device), target.to(self.device) # data is the batch of features, target is the batch of targets.
            self.net.zero_grad()                                        # sets gradients to 0 before loss calc. You will do this likely every step.
            hidden = self.net._init_hidden(batch_size=batch_size)
            output = self.net(data,hidden).squeeze()                           # pass in the reshaped batch
            target = target.long()
            data = output.tolist()
            data2 = output.data[1]
            correct += self.num_correct(output, target)

            #pred = get_likely_index(output)

    def num_correct(self, pred, target):
        pred.eval()
        