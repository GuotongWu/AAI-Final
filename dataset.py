import os
import random
import torch
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class SoundTrainValidDataset(Dataset):
    def __init__(self, data_dir, type, segement_length=16000*2, valid_ratio=0.08):
        self.data_dir = data_dir
        self.segement_length = segement_length
        self.dicktkey = dict()
        idx = 0
        for _, _, filelist in os.walk(data_dir):
            for filename in filelist:
                self.dicktkey[idx] = filename
                idx += 1
        train_list, valid_list, _, _ = train_test_split(
            list(self.dicktkey.keys()), 
            list(self.dicktkey.values()),
            test_size=valid_ratio)
        if type == "train":
            self.data_list = train_list
        else:
            self.data_list = valid_list
        print("Successfully load {} data, {} mode".format(len(self.data_list), type))
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        index = self.data_list[index]
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
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.int64)

    
class SoundTestDataset(Dataset):
    def __init__(self, data_dir, segement_length=16000*2):
        self.data_dir = data_dir
        self.segement_length = segement_length
        self.data_list = list()
        for _, _, filelist in os.walk(data_dir):
            for filename in filelist:
                index = int(filename.split('.')[0])
                self.data_list.append(index)
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        index = self.data_list[index]
        filename = os.path.join(self.data_dir, "{0:03}.flac".format(index))
        data, _ = sf.read(filename)

if __name__ == "__main__":
    train_data = SoundTrainValidDataset("./LibriSpeech-SI/train", "train")
    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=10)
    valid_data = SoundTrainValidDataset("./LibriSpeech-SI/train", "valid")
    valid_loader = DataLoader(dataset=valid_data, batch_size=64, shuffle=False, num_workers=10)
    print(list(train_loader)[0][0].shape)