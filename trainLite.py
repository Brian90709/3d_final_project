import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from pathlib import Path
import numpy as np
import multiprocessing
import math
from tqdm import tqdm
import matplotlib.pyplot as plt


# check if cuda is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# define simple logging functionality
log_fw = open(f"./logLite.txt", 'w') # open log file to save log outputs
def log(text):     # define a logging function to trace the training process
    print(text)
    log_fw.write(str(text)+'\n')
    log_fw.flush()


v = 0     # model version
in_c = 2  # number of input channels
num_c = 1 # number of classes to predict

model = EfficientNet.from_pretrained(f'efficientnet-b{v}', in_channels=in_c, num_classes=num_c)
model.to(device);

# directory with the optical flow images
of_dir = '../opical-flow-estimation/opticalLite'
# labels as txt file
labels_f = '../speedchallenge/data/train.txt'

class OFDataset(Dataset):
    def __init__(self, of_dir, label_f):
        self.len = len(list(Path(of_dir).glob('*.npy')))
        self.of_dir = of_dir
        self.label_file = open(label_f).readlines()
    def __len__(self): return self.len
    def __getitem__(self, idx):
        of_array = np.load(Path(self.of_dir)/f'{idx}.npy')
        of_tensor = torch.squeeze(torch.Tensor(of_array))
        label = float(self.label_file[idx].split()[0])
        return [of_tensor, label]


ds = OFDataset(of_dir, labels_f)


# 80% of data for training
# 20% of data for validation
train_split = .8

ds_size = len(ds)
indices = list(range(ds_size))
split = int(np.floor(train_split * ds_size))
train_idx, val_idx = indices[:split], indices[split:]

sample = ds[3]
assert type(sample[0]) == torch.Tensor
assert type(sample[1]) == float

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

cpu_cores = multiprocessing.cpu_count()


train_dl = DataLoader(ds, batch_size=8, sampler=train_sampler, num_workers=0)
val_dl = DataLoader(ds, batch_size=8, sampler=val_sampler, num_workers=0)

def plot(train_loss,val_loss,title):
    N = len(train_loss)
    plt.plot(range(N),train_loss,label = 'train_loss')
    plt.plot(range(N),val_loss, label = 'val_loss')
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.legend()
    plt.savefig("./loss/resultLite.png")
    plt.show()


epochs = 25 
log_train_steps = 100

criterion = nn.MSELoss()
opt = optim.Adam(model.parameters())

history_train_loss = []
history_val_loss = []
best_loss = math.inf
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    train_losses = []
    for i, sample in enumerate(tqdm(train_dl)):
        of_tensor = sample[0].cuda()
        label = sample[1].float().cuda()
        opt.zero_grad()
        pred = torch.squeeze(model(of_tensor))
        loss = criterion(pred, label)
        train_losses.append(loss)
        loss.backward()
        opt.step()
    mean_train_loss = sum(train_losses)/len(train_losses)
    history_train_loss.append(mean_train_loss.cpu())
    print(f'{epoch}t: {mean_train_loss}')
    # validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for j, val_sample in enumerate(tqdm(val_dl)):
            of_tensor = val_sample[0].cuda()
            label = val_sample[1].float().cuda()
            pred = torch.squeeze(model(of_tensor))
            loss = criterion(pred, label)
            val_losses.append(loss)
        mean_val_loss = sum(val_losses)/len(val_losses)
        torch.save(model.state_dict(), f'./Lite{epoch}.pth')
        if(mean_val_loss < best_loss):
            torch.save(model.state_dict(), f'./Lite{epoch}_best.pth')
            best_loss = mean_val_loss
        history_val_loss.append(mean_val_loss.cpu())
        print(f'{epoch}: {mean_val_loss}')


plot(history_train_loss,history_val_loss,"efficientnetb0")