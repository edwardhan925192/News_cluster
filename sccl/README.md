# Usage  

### Dependencies:
    !pip install sentence-transformers==2.0.0.
    !pip install transformers
    !pip install tensorboardX==2.4.1
    !pip install sklearn==0.24.1
    !pip install huggingface_hub
    !pip install --upgrade sentence_transformers
      

### SCCL with explicit augmentations 

In additional to the original data, SCCL requires a pair of augmented data for each instance. 

The data format is (text, text1, text2) where text1 and text2 are the column names of augmented pairs. 
 See our NAACL paper for details about the learning objective. 

Step-1. Augment data. Follow the instruction in News_cluster/Aug

Step-2. Run the code 

Whats returned (current dir)  
* Cluster assignments
* embedding vectors  

```python
!python main.py \
--resdir '/content/sccl'\
--use_pretrain SBERT \
--bert distilbert \
--use_cls \  
--datapath '/content/'\
--dataname 'aug_med'\
--num_classes 8 \
--text text \
--objective SCCL \
--augtype explicit \
--temperature 0.5 \
--eta 10 \
--lr 1e-05 \
--lr_scale 100 \
--max_length 40 \
--batch_size 50 \
--max_iter 1100 \
--print_freq 100 \
--gpuid 0 &

```

## Citation:

```bibtex
@inproceedings{zhang-etal-2021-supporting,
    title = "Supporting Clustering with Contrastive Learning",
    author = "Zhang, Dejiao  and
      Nan, Feng  and
      Wei, Xiaokai  and
      Li, Shang-Wen  and
      Zhu, Henghui  and
      McKeown, Kathleen  and
      Nallapati, Ramesh  and
      Arnold, Andrew O.  and
      Xiang, Bing",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.427",
    doi = "10.18653/v1/2021.naacl-main.427",
    pages = "5419--5430",
    abstract = "Unsupervised clustering aims at discovering the semantic categories of data according to some distance measured in the representation space. However, different categories often overlap with each other in the representation space at the beginning of the learning process, which poses a significant challenge for distance-based clustering in achieving good separation between different categories. To this end, we propose Supporting Clustering with Contrastive Learning (SCCL) {--} a novel framework to leverage contrastive learning to promote better separation. We assess the performance of SCCL on short text clustering and show that SCCL significantly advances the state-of-the-art results on most benchmark datasets with 3{\%}-11{\%} improvement on Accuracy and 4{\%}-15{\%} improvement on Normalized Mutual Information. Furthermore, our quantitative analysis demonstrates the effectiveness of SCCL in leveraging the strengths of both bottom-up instance discrimination and top-down clustering to achieve better intra-cluster and inter-cluster distances when evaluated with the ground truth cluster labels.",}

```
    
