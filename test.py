import torch
import torchvision
import torchaudio
import utils
print(torch.cuda.is_available())

texp = utils.TextProcess()

i = texp.text_to_int_sequence("ree")

w = texp.int_to_text_sequence(i)

print("word:" + w + " int: " + str(i))