from torch.utils.data import DataLoader
from ECAPAModel import ECAPAModel
from dataset import SoundTrainDataset, SoundValidDataset, split_train_valid
import torch
import random
import numpy as np
import time

def set_seed(seed):
    torch.manual_seed(seed) # cpu vars
    torch.cuda.manual_seed(seed)  # gpu vars
    torch.backends.cudnn.deterministic = True  # cudnn
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms

set_seed(0)

s = ECAPAModel(lr=0.002, lr_decay=0.95, C=1024, n_class=251, m=0.2, s=30, test_step=1)
train_list, valid_list, valid_pair = split_train_valid("./LibriSpeech-SI/train")
train_data = SoundTrainDataset("./LibriSpeech-SI/train", train_list)
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
valid_data = SoundValidDataset("./LibriSpeech-SI/train", valid_list)
valid_loader = DataLoader(dataset=valid_data, batch_size=64, shuffle=False)

EERs = []
minDCFs = []
score_file = open("./saved/score_file.log", "a+")

for epoch in range(100):
	## Training for one epoch
	loss, lr, acc = s.train_network(epoch = epoch, loader = train_loader)

	## Evaluation every [test_step] epochs
	if epoch % 2 == 0:
		s.save_parameters("./saved/model" + "/model_%04d.model"%epoch)
		eer, minDCF = s.eval_network("./LibriSpeech-SI/train", valid_pair)
		EERs.append(eer)
		minDCFs.append(minDCF)
		print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%, minDCF bestEER %2.2f%%"%(epoch, acc, EERs[-1], min(EERs), minDCFs[-1]))
		score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%, minDCF bestEER %2.2f%%\n"%(epoch, lr, loss, acc, EERs[-1], min(EERs), minDCFs[-1]))
		score_file.flush()
