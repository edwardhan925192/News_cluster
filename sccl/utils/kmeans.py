import torch
import numpy as np
from utils.metric import Confusion
from sklearn.cluster import KMeans


def get_embeddings(bert, input_ids, attention_mask,use_cls = False ):
    bert_output = bert.forward(input_ids=input_ids, attention_mask=attention_mask)

    # ============================================ #
    # ================= updated ================== #
    # ============================================ #    
    if use_cls:
        return bert_output[0][:, 0, :]
    else:
        attention_mask = attention_mask.unsqueeze(-1)
        mean_output = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return mean_output

    
def get_batch_token(tokenizer, text, max_length):
    token_feat = tokenizer.batch_encode_plus(
        text, 
        max_length=max_length, 
        return_tensors='pt', 
        padding='max_length', 
        truncation=True
    )
    return token_feat


def get_kmeans_centers(bert, tokenizer, train_loader, num_classes, max_length, use_cls=False):
    for i, batch in enumerate(train_loader):

        text = batch['text']
        tokenized_features = get_batch_token(tokenizer, text, max_length)

        # ============================================ #
        # ================= updated ================== #
        # ============================================ #    
        #if model_type not in ['bert', 'roberta']:          
        
        tokenized_features.pop('token_type_ids', None)
        
        corpus_embeddings = get_embeddings(bert, use_cls=use_cls, **tokenized_features)
        
        if i == 0:
            all_embeddings = corpus_embeddings.detach().numpy()
        else:
            all_embeddings = np.concatenate((all_embeddings, corpus_embeddings.detach().numpy()), axis=0)

    # Perform KMeans clustering
    clustering_model = KMeans(n_clusters=num_classes)
    clustering_model.fit(all_embeddings)

    print("all_embeddings shape:", all_embeddings.shape)
    print("Iterations:", clustering_model.n_iter_)
    print("Centers shape:", clustering_model.cluster_centers_.shape)
    
    return clustering_model.cluster_centers_




