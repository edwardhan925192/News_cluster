import argparse
import os
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from nlpaug.util import Action

def set_global_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def contextual_augment(args):
    ### contextual augmentation
    print(f"\n-----transformer_augment-----\n")
    augmenter1 = naw.ContextualWordEmbsAug(
        model_path=args.model_path1, action="substitute", aug_min=1, aug_p=args.aug_p, device=args.device1)

    augmenter2 = naw.ContextualWordEmbsAug(
        model_path=args.model_path2, action="substitute", aug_min=1, aug_p=args.aug_p, device=args.device2)

    train_data = pd.read_csv(args.data_path)
    train_text = train_data[args.textcol].fillna('.').astype(str).values
    print("train_text:", len(train_text), type(train_text[0]))

    auglist1, auglist2 = [], []
    for txt in train_text:
        atxt1 = augmenter1.augment(txt)
        atxt2 = augmenter2.augment(txt)
        auglist1.append(str(atxt1))
        auglist2.append(str(atxt2))

    train_data[args.textcol + "1"] = pd.Series(auglist1)
    train_data[args.textcol + "2"] = pd.Series(auglist2)
    train_data.to_csv(args.output_path, index=False)

    for o, a1, a2 in zip(train_text[:5], auglist1[:5], auglist2[:5]):
        print("-----Original Text: \n", o)
        print("-----Augmented Text1: \n", a1)
        print("-----Augmented Text2: \n", a2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Contextual augmentation using transformer models.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input data')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the augmented data')
    parser.add_argument('--textcol', type=str, default="text", help='Column name containing the text data')
    parser.add_argument('--aug_p', type=float, default=0.2, help='Augmentation probability')
    parser.add_argument('--device1', type=str, default="cuda", help='Device for the first transformer model')
    parser.add_argument('--device2', type=str, default="cuda", help='Device for the second transformer model')
    parser.add_argument('--model_path1', type=str, default='roberta-base', help='Path to the first transformer model')
    parser.add_argument('--model_path2', type=str, default='bert-base-uncased', help='Path to the second transformer model')

    args = parser.parse_args()
    contextual_augment(args)
