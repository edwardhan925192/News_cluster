import os
import torch 
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer


#========================================================================================
# =========================== Choosing the model from here ============================ # 
#========================================================================================

BERT_CLASS = {
    "distilbert": 'distilbert-base-uncased', 
    "deberta": "microsoft/deberta-base",
    "roberta": 'roberta-base'
}

SBERT_CLASS = {
    "distilbert": 'distilbert-base-nli-stsb-mean-tokens',
}


def get_optimizer(model, args):
    
    optimizer = torch.optim.Adam([
        {'params':model.bert.parameters()}, 
        {'params':model.contrast_head.parameters(), 'lr': args.lr*args.lr_scale},
        {'params':model.cluster_centers, 'lr': args.lr*args.lr_scale}
    ], lr=args.lr)
    
    print(optimizer)
    return optimizer 
    

def get_bert(args):            
    # Check if SBERT pretraining is specified
    if args.use_pretrain == "SBERT":        
        bert_model = get_sbert(args)
        tokenizer = bert_model[0].tokenizer
        model = bert_model[0].auto_model
        print("..... loading Sentence-BERT !!!")
        
        
    else:
        # Load plain BERT or DeBERTa based on args.bert
        if args.bert.lower() == "deberta":
            config = AutoConfig.from_pretrained("microsoft/deberta-base")  
            model = AutoModel.from_pretrained("microsoft/deberta-base", config=config)  
            tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")  
            print("..... loading DeBERTa !!!")
            
        elif args.bert.lower() == "roberta":  
            config = AutoConfig.from_pretrained(BERT_CLASS[args.bert])
            model = AutoModel.from_pretrained(BERT_CLASS[args.bert], config=config)
            tokenizer = AutoTokenizer.from_pretrained(BERT_CLASS[args.bert])
            print("..... loading RoBERTa !!!")
            
        elif args.bert in BERT_CLASS:
            config = AutoConfig.from_pretrained(BERT_CLASS[args.bert])
            model = AutoModel.from_pretrained(BERT_CLASS[args.bert], config=config)
            tokenizer = AutoTokenizer.from_pretrained(BERT_CLASS[args.bert])
            print("..... loading plain BERT !!!")
            
        else:
            raise ValueError(f"No model configuration found for {args.bert}.")
    return model, tokenizer



def get_sbert(args):
    sbert = SentenceTransformer(SBERT_CLASS[args.bert])
    return sbert








