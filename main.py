from torch.utils.data import DataLoader
from ECAPAModel import ECAPAModel
from dataset import SoundTrainValidDataset
import torch
import random
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed) # cpu vars
    torch.cuda.manual_seed(seed)  # gpu vars
    torch.backends.cudnn.deterministic = True  # cudnn
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms

set_seed(0)

s = ECAPAModel(lr=0.002, lr_decay=0.95, C=1024, n_class=251, m=0.2, s=30, test_step=1)
train_data = SoundTrainValidDataset("./LibriSpeech-SI/train", "train", segement_length=16000*2+240)
valid_data = SoundTrainValidDataset("./LibriSpeech-SI/train", "valid", segement_length=16000*2+240)
train_loader = DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=4)
valid_loader = DataLoader(dataset=valid_data, batch_size=128, shuffle=False, num_workers=4)

for epoch in range(100):
    s.train_network(epoch, train_loader)

    if epoch % 1 == 0:
        s.save_parameters("./saved_model"+"/model_%04d.model"%epoch)
        s.eval_network(epoch, valid_loader)
