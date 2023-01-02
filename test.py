import torch 
import numpy as np
import random
from dataset import split_train_valid, SoundValidDataset, SoundTestDataset
from torch.utils.data import DataLoader
from ECAPAModel import ECAPAModel

def set_seed(seed):
    torch.manual_seed(seed) # cpu vars
    torch.cuda.manual_seed(seed)  # gpu vars
    torch.backends.cudnn.deterministic = True  # cudnn
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms

set_seed(0)

clean_test_data = SoundTestDataset(data_dir="./LibriSpeech-SI/test", aug=True)
clean_test_loader = DataLoader(dataset=clean_test_data, batch_size=512, shuffle=False)
noisy_test_data = SoundTestDataset(data_dir="./LibriSpeech-SI/test-noisy", aug=False)
noisy_test_loader = DataLoader(dataset=noisy_test_data, batch_size=512, shuffle=False)

s = ECAPAModel(lr=0.002, lr_decay=0.95, C=1024, n_class=251, m=0.2, s=30, test_step=1)
s.load_parameters("./saved/model/model_0014.model")
s = s.to("cpu")
clean_pred = []
noisy_pred = []

for x in clean_test_loader:
    with torch.no_grad():
        speaker_embedding = s.speaker_encoder.forward(x)
        output       = s.speaker_loss.forward(speaker_embedding)
        pred = torch.argmax(output, dim=1)
        clean_pred += pred.tolist()
        print(clean_pred)
        
for x in noisy_test_loader:
    with torch.no_grad():
        speaker_embedding = s.speaker_encoder.forward(x)
        output       = s.speaker_loss.forward(speaker_embedding)
        pred = torch.argmax(output, dim=1)
        noisy_pred += pred.tolist()
        print(noisy_pred)
        
def write_data(fileName, data, list):
    list, data = np.array(list), np.array(data)
    index = np.argsort(list)
    data = data[index]
    with open(fileName, 'w') as f:
        for label in data:
            f.write(str(label)+'\n')
            
write_data("./saved/test.txt", clean_pred, clean_test_data.datalist)
write_data("./saved/test-noisy.txt", noisy_pred, noisy_test_data.datalist)