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
    def __init__(self, file_path, epochs, batch_size=4):
        print("Cuda : " + str(torch.cuda.is_available()))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #set cpu or gpu
        self.file_path = file_path
        self.net = model.SpeechRecognition()
        self.criterion = nn.CTCLoss(blank=28).to(self.device)
        if torch.cuda.is_available():
            self.net.cuda()
        else:
            self.net.cpu()
        if file_path is not None and path.exists(file_path):
            self.load()
            self.net.to(self.device)
        #set training waveform data transformer
        self.train_audio_transforms = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
        torchaudio.transforms.TimeMasking(time_mask_param=100)
        )
        #set testing waveform data transformer
        self.valid_audio_transforms = torchaudio.transforms.MelSpectrogram()
        #train
        self.train(epochs=epochs, batch_size=batch_size)

    #------------------------------------------------
    # data order
    # waveform, sample_rate, dic[sentence, person id, etc.]
    #------------------------------------------------
    #collate functions helps the dataloader set labels and data
    def collate_fn(self, data, data_type="train"):
        spectrograms = []
        labels = []
        input_lengths = []
        label_lengths = []
        for (waveform, _, utterance, _, _, _) in data:
            if data_type == 'train':
                spec = self.train_audio_transforms(waveform).squeeze(0).transpose(0, 1) #for training
            else:
                spec = self.valid_audio_transforms(waveform).squeeze(0).transpose(0, 1) #for testing
            spectrograms.append(spec)
            label = torch.Tensor(textprocess.text_to_int_sequence(utterance.lower()))
            labels.append(label)
            input_lengths.append(spec.shape[0]//2)
            label_lengths.append(len(label))
        spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
        return spectrograms, labels, input_lengths, label_lengths
    
    def load(self):
        #loading neural net
        self.net.load_state_dict(torch.load(self.file_path)) # dictionary
        print("loaded neural network file: " + self.file_path)
    
    def save(self):
        #save the neural network
        print("saved file")
        if self.file_path is not None:
            torch.save(self.net.state_dict(), self.file_path) #save input size and dictionary

    def train(self, epochs, batch_size=4):
        #setup net and loss
        self.net.train()
        for epoch in range(0,epochs):
            #setup dataset
            #train = torchaudio.datasets.COMMONVOICE(root='/media/gussim/SlaveDisk/MCV',version= 'cv-corpus-6.1-2020-12-11', download = False)
            train = torchaudio.datasets.LIBRISPEECH("./", url="train-clean-100", download=False)
            trainset = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
            trainset_len = len(trainset)
            i = 0
            j = 1
            optimizer = optim.AdamW(self.net.parameters(), lr=5e-4)
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-4, steps_per_epoch=int(trainset_len), epochs=epochs, anneal_strategy='linear')
            for data in tqdm(trainset, desc="Epoch #"+str(epoch)):

                spectrogramdata, labels, input_lengths, label_lengths = data 
                spectrogramdata, labels = spectrogramdata.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                output = self.net(spectrogramdata).contiguous()  # batch, time, num_class
                outp = output.tolist()
                output = F.log_softmax(output,dim=2)
                outp2 = output.tolist()
                output = output.transpose(0, 1) # time, batch, num_class

                loss = self.criterion(output, labels, input_lengths, label_lengths)
                #print(loss.item())
                loss.backward()                                             # apply this loss backwards thru the network's parameters

                optimizer.step()                                            # attempt to optimize weights to account for loss/gradients
                scheduler.step()
                #save
                i = i + 1
                if((i/trainset_len)*100) > j:
                    #self.test(batch_size=batch_size)
                    self.net.train()
                    j = j + 1
                    self.save()
                if i > trainset_len-2:
                    break
            #after train test the model
            #save here every epoch
            self.save()

    def test(self,batch_size=4):
        self.net.eval()
        #test = torchaudio.datasets.COMMONVOICE(root='/media/gussim/SlaveDisk/MCV',version= 'cv-corpus-6.1-2020-12-11', download = False)
        test = torchaudio.datasets.LIBRISPEECH("./", url="test-clean", download=False)
        testset = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        test_loss = 0
        test_cer, test_wer = [], []
        with torch.no_grad():
            i = 0
            for data in testset:
                spectrograms, labels, input_lengths, label_lengths = data 
                spectrograms, labels = spectrograms.to(self.device), labels.to(self.device)
                output = self.net(spectrograms)  # batch, time, num_class
                output = F.log_softmax(output, dim=2)
                output = output.transpose(0, 1) # time, batch, num_class
                outptlst = output.tolist()
                loss = self.criterion(output, labels, input_lengths, label_lengths)
                test_loss += loss.item() / len(testset)
                decoded_preds, decoded_targets = textprocess.greedy_decoder(output.transpose(0, 1), labels, label_lengths)
                for j in range(len(decoded_preds)):
                    test_cer.append(utils.cer(decoded_targets[j], decoded_preds[j]))
                    test_wer.append(utils.wer(decoded_targets[j], decoded_preds[j]))
                avg_cer = sum(test_cer)/len(test_cer)
                avg_wer = sum(test_wer)/len(test_wer)
                print(avg_cer)
                print(avg_wer)
                i = i + 1
                if i == 10:
                    break
        print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))