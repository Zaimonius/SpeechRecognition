import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torchvision import datasets


class Trainer:
    def __init__(self,params):
        train = torchaudio.datasets.commonvoiceCOMMONVOICE(root: str, tsv: str = 'train.tsv', url: str = 'english', folder_in_archive: str = 'CommonVoice', version: str = 'cv-corpus-4-2019-12-10', download: bool = False)
        test = torchaudio.datasets.commonvoiceCOMMONVOICE(root: str, tsv: str = 'train.tsv', url: str = 'english', folder_in_archive: str = 'CommonVoice', version: str = 'cv-corpus-4-2019-12-10', download: bool = False)

        trainset = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
        testset = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)

