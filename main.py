from torch.utils.data import DataLoader
from ECAPAModel import ECAPAModel
from dataset import SoundTrainValidDataset

s = ECAPAModel(lr=0.001, lr_decay=0.97, C=1024, n_class=251, m=0.2, s=30, test_step=1)
train_data = SoundTrainValidDataset("./LibriSpeech-SI/train", "train")
valid_data = SoundTrainValidDataset("./LibriSpeech-SI/train", "valid")
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=64, shuffle=False)

for epoch in range(10):
    s.train_network(epoch, train_loader)
