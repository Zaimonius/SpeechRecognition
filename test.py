import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import utils
import train

#REQUIREMENTS:
#PyTorch
#Pip
#Python
#tqdm

#------------------------------------------------

# print(torch.cuda.is_available())

#------------------------------------------------

# texp = utils.TextProcess()

# i = texp.text_to_int_sequence("hej jeg ar simme")

# w = texp.int_to_text_sequence(i)

# print("word: " + w + " int: " + str(i))

#------------------------------------------------

# m = nn.Dropout(p=0.2)
# output = torch.randn(20)
# output = m(output)
# print(output)

#------------------------------------------------
#Train
trainer = train.Trainer(file_path="model_dict.txt", epochs=10)

#------------------------------------------------
#Test
# trainer = train.Trainer(file_path="model_dict.txt", epochs=0)
# trainer.test()
