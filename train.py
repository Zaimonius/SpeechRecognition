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

textprocess = utils.TextProcess()

class Trainer: 
    def __init__(self, file_path, epochs, batch_size=64):
        print("Cuda : " + str(torch.cuda.is_available()))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.file_path = file_path
        if file_path is not None and path.exists(file_path):
            self.load(file_path)
            self.net.to(self.device)
            self.net.eval()
        self.train(epochs=epochs, batch_size=batch_size)
    
    def pad_sequence(self, batch):
        # Make all tensor in a batch the same length by padding with zeros
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch.permute(0, 2, 1)

    #------------------------------------------------
    # data order
    # waveform, sample_rate, dic
    #------------------------------------------------
    #collate functions helps the dataloader set labels and data
    def collate_fn(self, batch):
        data_list, target_list = [], []
        for _data, _, _target in batch:
            data_list.append(_data.t()) #input
            target_list.append(torch.Tensor(textprocess.text_to_int_sequence(_target['sentence'].lower()))) #target output
        #format data in the end
        return self.pad_sequence(data_list), torch.nn.utils.rnn.pad_sequence(target_list, batch_first=True, padding_value=0.)
    
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

    def train(self, epochs, batch_size=64):
        #setup net
        self.net = model.SpeechRecognition()
        optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        for epoch in range(0,epochs):
            print("Epoch #", epoch)
            #setup dataset
            train = torchaudio.datasets.COMMONVOICE(root='/media/gussim/SlaveDisk/MCV',version= 'cv-corpus-6.1-2020-12-11', download = False)
            trainset = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
            for data, target in trainset:
                data, target = data.to(self.device), target.to(self.device) # data is the batch of features, target is the batch of targets.
                self.net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
                hidden = self.net._init_hidden(batch_size=batch_size)
                output = self.net(data,hidden)  # pass in the reshaped batch 
                loss = F.nll_loss(output, target)
                loss.backward()  # apply this loss backwards thru the network's parameters
                optimizer.step()  # attempt to optimize weights to account for loss/gradients
            #save here every epoch
            self.save()
        #after train test
        #setup dataset
        #test = torchaudio.datasets.COMMONVOICE(root="/media/gussim/SlaveDisk/MCV/cv-corpus-6.1-2020-12-11", version='cv-corpus-4-2019-12-10', download=False)
        #testset = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)





# def train(args, model, device, train_loader, optimizer, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))
#             if args.dry_run:
#                 break


# def test(model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)

#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))
