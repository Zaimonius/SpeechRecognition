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

mode = 1


#------------------------------------------------
#Train
if mode == 1:
    trainer = train.Trainer(file_path="newtest.txt", epochs=1)
    trainer.test()

#------------------------------------------------
#Test
if mode == 2:
    trainer = train.Trainer(file_path="newtest.txt", epochs=0)
    trainer.test()

# #Target are to be padded
# T = 50      # Input sequence 
# C = 20      # Number of classes (including blank)
# N = 16      # Batch size
# S = 30      # Target sequence length of longest target in batch (padding length)
# S_min = 10  # Minimum target length, for demonstration purposes

# # Initialize random batch of input vectors, for *size = (T,N,C)
# input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()

# # Initialize random batch of targets (0 = blank, 1:C = classes)
# target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)

# input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
# target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
# ctc_loss = nn.CTCLoss()
# loss = ctc_loss(input, target, input_lengths, target_lengths)
# loss.backward()