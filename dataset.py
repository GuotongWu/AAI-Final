import os
import random
import torch
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def split_train_valid(data_dir, valid_ratio=0.08):
    dicktkey = dict()
    idx = 0
    for _, _, filelist in os.walk(data_dir):
        for filename in filelist:
            dicktkey[idx] = filename
            idx += 1
    train_list, valid_list, _, _ = train_test_split(
        list(dicktkey.keys()), 
        list(dicktkey.values()),
        test_size=valid_ratio)
    return dicktkey, train_list, valid_list

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
    def __init__(self, data_dir, dicktkey, valid_list, aug=True, segement_length=16000*2):
        self.data_dir = data_dir
        self.aug = aug
        self.dicktkey = dicktkey
        self.valid_list = valid_list
        self.new_list = np.random.permutation(valid_list)
        self.segement_length = segement_length

    def __len__(self):
        return len(self.valid_list)

    def __getitem__(self, index):
        x1, y1 = self.read_data(self.valid_list, index)
        x2, y2 = self.read_data(self.new_list, index)
        return x1, y1, x2, y2

    def read_data(self, data_list, index):
        index = data_list[index]
        label = int(self.dicktkey[index].split('_')[0][3:])
        filename = os.path.join(
            self.data_dir, 
            "spk{:03}".format(label), 
            self.dicktkey[index])
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
    def __init__(self, data_dir, dicktkey, train_list, aug=True, segement_length=16000*2):
        self.data_dir = data_dir
        self.aug = aug
        self.dicktkey = dicktkey
        self.train_list = train_list
        self.segement_length = segement_length
    
    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, index):
        index = self.train_list[index]
        label = int(self.dicktkey[index].split('_')[0][3:])
        filename = os.path.join(
            self.data_dir, 
            "spk{:03}".format(label), 
            self.dicktkey[index])
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
    dicktkey, train_list, valid_list = split_train_valid("./LibriSpeech-SI/train")
    train_data = SoundTrainDataset("./LibriSpeech-SI/train", dicktkey, train_list)
    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=10)
    valid_data = SoundValidDataset("./LibriSpeech-SI/train", dicktkey, valid_list)
    valid_loader = DataLoader(dataset=valid_data, batch_size=64, shuffle=False, num_workers=10)
    print(list(train_loader))