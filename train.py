import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torchvision import datasets
import utils

textprocess = utils.TextProcess()

class Trainer:
    def __init__(self, batch_size=64):
        train = torchaudio.datasets.COMMONVOICE(root: pathtodir,version: str = 'cv-corpus-4-2019-12-10', download: bool = False)
        test = torchaudio.datasets.COMMONVOICE(root: pathtodir,version: str = 'cv-corpus-4-2019-12-10', download: bool = False)

        trainset = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        testset = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

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
        data_list, label_list = [], []
        for _data, _, _label in batch:
            data_list.append(_data) #input
            label_list.append(textprocess.text_to_int_sequence(_label)) #target output
        #format data in the end
        return pad_sequence(data_list), torch.stack(label_list)

    