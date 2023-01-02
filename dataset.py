import os
import random
import torch
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from itertools import  permutations

def split_train_valid(data_dir, valid_avg=6):
    train_list = []
    valid_list = []
    valid_pair = []
    for _, _, filelist in os.walk(data_dir):
        random.shuffle(filelist)
        valid_num = random.randint(valid_avg-2,valid_avg+2)
        valid_list.append(filelist[:valid_num])
        train_list.append(filelist[valid_num:])
        valid_pair.append(random.sample(list(permutations(filelist[:valid_num], 2))), valid_num)
    return train_list, valid_list, valid_pair

def add_noisy(data, segement_length):
    filelist = os.listdir("./LibriSpeech-SI/noise")
    filename = os.path.join("./LibriSpeech-SI/noise", random.sample(filelist, 1)[0])
    noise, _ = sf.read(filename)
    PS = np.sum(data**2)/len(data)
    PN = np.sum(noise**2)/len(noise)
    noise = noise * np.sqrt(PS/(31.6*PN))
    if len(noise) < segement_length:
        noise = np.pad(noise, (0, segement_length - len(noise)), 'constant')
    else:
        start = random.randint(0, len(noise)-segement_length)
        noise = noise[start:start+segement_length]
    return data + noise

class SoundValidDataset(Dataset):
    def __init__(self, data_dir, valid_list, aug=True, segement_length=16000*2):
        self.data_dir = data_dir
        self.aug = aug
        self.valid_list = valid_list
        self.segement_length = segement_length

    def __len__(self):
        return len(self.valid_list)

    def __getitem__(self, index):
        return self.read_data(self.valid_list, index)

    def read_data(self, data_list, index):
        label = int(self.data_list[index].split('_')[0][3:])
        filename = os.path.join(
            self.data_dir, 
            "spk{:03}".format(label), 
            data_list[index])
        data, _ = sf.read(filename)
        if len(data) < self.segement_length:
            data = np.pad(data, (0, self.segement_length - len(data)), 'constant')  
        else:
            start = random.randint(0, len(data)-self.segement_length)
            data = data[start:start+self.segement_length]
        if self.aug:
            data = add_noisy(data, self.segement_length)
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.int64)


class SoundTrainDataset(Dataset):
    def __init__(self, data_dir, train_list, aug=True, segement_length=16000*2):
        self.aug = aug
        self.data_dir = data_dir
        self.train_list = train_list
        self.segement_length = segement_length
    
    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, index):
        label = int(self.train_list[index].split('_')[0][3:])
        filename = os.path.join(
            self.data_dir, 
            "spk{:03}".format(label), 
            self.train_list[index])
        data, _ = sf.read(filename)
        if len(data) < self.segement_length:
            data = np.pad(data, (0, self.segement_length - len(data)), 'constant')  
        else:
            start = random.randint(0, len(data)-self.segement_length)
            data = data[start:start+self.segement_length]
        if self.aug:
            data = add_noisy(data, self.segement_length)
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.int64)


if __name__ == "__main__":
    train_list, valid_list, valid_pair = split_train_valid("./LibriSpeech-SI/train")
    train_data = SoundTrainDataset("./LibriSpeech-SI/train", train_list)
    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=10)
    valid_data = SoundValidDataset("./LibriSpeech-SI/train", valid_list)
    valid_loader = DataLoader(dataset=valid_data, batch_size=64, shuffle=False, num_workers=10)
    print(list(train_loader))