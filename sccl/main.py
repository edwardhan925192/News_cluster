import os
import sys
sys.path.append( './' )
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import joblib
import torch
import argparse
from models.Transformers import SCCLBert
import dataloader.dataloader as dataloader
from training import SCCLvTrainer
from utils.kmeans import get_kmeans_centers, get_batch_token, get_embeddings
from utils.logger import setup_path, set_global_random_seed
from utils.assign_center import assign_to_closest_centers
from utils.optimizer import get_optimizer, get_bert
import numpy as np


def run(args):
    args.resPath, args.tensorboard = setup_path(args)
    set_global_random_seed(args.seed)

    # dataset loader
    train_loader = dataloader.explict_augmentation_loader(args) if args.augtype == "explicit" else dataloader.virtual_augmentation_loader(args)

    # model
    torch.cuda.set_device(args.gpuid[0])
    bert, tokenizer = get_bert(args)
    
    # initialize cluster centers
    # CLS token passed 
    cluster_centers = get_kmeans_centers(bert, tokenizer, train_loader, args.num_classes, args.max_length, args.use_cls)

    # CLS token passed 
    model = SCCLBert(bert, tokenizer, cluster_centers=cluster_centers, alpha=args.alpha, use_cls=args.use_cls) 

    model = model.cuda()

    # optimizer 
    optimizer = get_optimizer(model, args)
    
    trainer = SCCLvTrainer(model, tokenizer, optimizer, train_loader, args)
    trainer.train()  

    # ================== 09/22 ==================== #
    # Store all embeddings from training data  
    all_embeddings = []  
    
    # Iterate through the training loader to get the embeddings  
    with torch.no_grad():
        for batch in train_loader:  
            text = batch['text']
            tokenized_features = get_batch_token(tokenizer, text, args.max_length)
            input_ids, attention_mask = tokenized_features['input_ids'].cuda(), tokenized_features['attention_mask'].cuda()
            
            embeddings = model.get_mean_embeddings(input_ids, attention_mask)  
            all_embeddings.append(embeddings)  
    
    # Stack all embeddings  
    all_embeddings_np = torch.cat(all_embeddings, dim=0).cpu().numpy()
    cluster_centers_np = model.cluster_centers.detach().cpu().numpy()
    
    # Assign embeddings to closest center
    assignments, distance = assign_to_closest_centers(all_embeddings_np, cluster_centers_np)
    
    return assignments , all_embeddings_np , distance  

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_instance', type=str, default='local') 
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0], help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--seed', type=int, default=0, help="")
    parser.add_argument('--print_freq', type=float, default=100, help="")
    parser.add_argument('--resdir', type=str, default='./results/')
    parser.add_argument('--s3_resdir', type=str, default='./results')  
    parser.add_argument('--use_cls', type=str2bool, default=False, help="Use the CLS token for embedding instead of mean")
    
    parser.add_argument('--bert', type=str, default='distilroberta', help="")
    parser.add_argument('--use_pretrain', type=str, default='BERT', choices=["BERT", "SBERT", "PAIRSUPCON"])
    
    # Dataset
    parser.add_argument('--datapath', type=str, default='../datasets/')
    parser.add_argument('--dataname', type=str, default='searchsnippets', help="")
    parser.add_argument('--num_classes', type=int, default=8, help="")
    parser.add_argument('--max_length', type=int, default=32)      
    parser.add_argument('--text', type=str, default='text')
    parser.add_argument('--augmentation_1', type=str, default='text1')
    parser.add_argument('--augmentation_2', type=str, default='text2')
    # Learning parameters
    parser.add_argument('--lr', type=float, default=1e-5, help="")
    parser.add_argument('--lr_scale', type=int, default=100, help="")
    parser.add_argument('--max_iter', type=int, default=1000)
    # contrastive learning
    parser.add_argument('--objective', type=str, default='contrastive')
    parser.add_argument('--augtype', type=str, default='virtual', choices=['virtual', 'explicit'])
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument('--temperature', type=float, default=0.5, help="temperature required by contrastive loss")
    parser.add_argument('--eta', type=float, default=1, help="")
    
    # Clustering
    parser.add_argument('--alpha', type=float, default=1.0)
    
    args = parser.parse_args(argv)
    args.use_gpu = args.gpuid[0] >= 0
    args.resPath = None
    args.tensorboard = None

    return args

if __name__ == '__main__':
    import subprocess
       
    args = get_args(sys.argv[1:])
    
    if args.train_instance == "sagemaker":
        assignments, embeddings, distance = run(args)
        joblib.dump(assignments, 'assignments.pkl')        
        joblib.dump(embeddings, 'embeddings.pkl')   
        joblib.dump(distance, 'distance.pkl')   
        subprocess.run(["aws", "s3", "cp", "--recursive", args.resdir, args.s3_resdir])
    else:
        assignments, embeddings, distance = run(args)
        joblib.dump(assignments, 'assignments.pkl')        
        joblib.dump(embeddings, 'embeddings.pkl')        
        joblib.dump(distance, 'distance.pkl')   


    
