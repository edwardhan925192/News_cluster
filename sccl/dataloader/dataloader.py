import os
import pandas as pd
import torch.utils.data as util_data
from torch.utils.data import Dataset

class VirtualAugSamples(Dataset):
    def __init__(self, train_x):
        self.train_x = train_x

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, idx):
        return {'text': self.train_x[idx]}

    
class ExplitAugSamples(Dataset):
    def __init__(self, train_x, train_x1, train_x2):
        assert len(train_x) == len(train_x1) == len(train_x2)
        self.train_x = train_x
        self.train_x1 = train_x1
        self.train_x2 = train_x2
        
    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, idx):
        return {'text': self.train_x[idx], 'augmentation_1': self.train_x1[idx], 'augmentation_2': self.train_x2[idx]}
       

def explict_augmentation_loader(args):
    train_data = pd.read_csv(os.path.join(args.datapath, args.dataname+".csv"))
    train_text = train_data[args.text].fillna('.').values
    train_text1 = train_data[args.augmentation_1].fillna('.').values
    train_text2 = train_data[args.augmentation_2].fillna('.').values

    train_dataset = ExplitAugSamples(train_text, train_text1, train_text2)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    return train_loader


def virtual_augmentation_loader(args):
    train_data = pd.read_csv(os.path.join(args.datapath, args.dataname+".csv"))
    train_text = train_data[args.text].fillna('.').values

    train_dataset = VirtualAugSamples(train_text)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)   
    return train_loader


def unshuffle_loader(args):
    train_data = pd.read_csv(os.path.join(args.datapath, args.dataname+".csv"))
    train_text = train_data[args.text].fillna('.').values

    train_dataset = VirtualAugSamples(train_text)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)   
    return train_loader
