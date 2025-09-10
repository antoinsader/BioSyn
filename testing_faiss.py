import torch

import sys
import os

from src.biosyn import (
    QueryDataset, 
    CandidateDataset, 
    DictionaryDataset,
    TextPreprocess, 
    RerankNet, 
    BioSyn
)

import torch
import time
import numpy as np
import random

root = "."
max_length = 25
topk = 20
use_cuda = torch.cuda.is_available()
learning_rate = .0001
weight_decay = 0.01

train_batch_size = 128

print(f"use_cuda: {use_cuda}")



model_name_path= 'dmis-lab/biobert-base-cased-v1.1'
train_dictionary_path =  f"{root}/data/data-ncbi-fair/train_dictionary.txt"
train_dir =  f"{root}/data/data-ncbi-fair/traindev"

def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_dictionary(dictionary_path):
    """
    load dictionary
    
    Parameters
    ----------
    dictionary_path : str
        a path of dictionary
    """
    dictionary = DictionaryDataset(
            dictionary_path = dictionary_path
    )
    
    return dictionary.data
    
def load_queries(data_dir, filter_composite, filter_duplicate, filter_cuiless):
    """
    load query data
    
    Parameters
    ----------
    is_train : bool
        train or dev
    filter_composite : bool
        filter composite mentions
    filter_duplicate : bool
        filter duplicate queries
    filter_cuiless : bool
        filter samples with cuiless
    """
    dataset = QueryDataset(
        data_dir=data_dir,
        filter_composite=filter_composite,
        filter_duplicate=filter_duplicate,
        filter_cuiless=filter_cuiless
    )
    
    return dataset.data



train_dictionary = load_dictionary(dictionary_path=train_dictionary_path)
train_queries = load_queries(
    data_dir = train_dir, 
    filter_composite=True,
    filter_duplicate=True,
    filter_cuiless=True
)

biosyn = BioSyn(
        max_length=max_length,
        use_cuda=use_cuda,
    )
biosyn.load_dense_encoder(
        model_name_or_path=model_name_path
    )
model = RerankNet(
        learning_rate=learning_rate, 
        weight_decay=weight_decay,
        encoder = biosyn.get_dense_encoder(),
        use_cuda=use_cuda
    )


train_set = CandidateDataset(
    queries = train_queries, 
    dicts = train_dictionary, 
    tokenizer = biosyn.tokenizer, 
    topk = topk,
    max_length=max_length
    )


train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=True,
    )

    
names_in_train_dictionary = train_dictionary[:,0]
names_in_train_queries = train_queries[:,0]


train_query_dense_embeds = biosyn.embed_dense(names=names_in_train_queries, show_progress=True)
train_dict_dense_embeds = biosyn.embed_dense(names=names_in_train_dictionary, show_progress=True)
train_dense_score_matrix = biosyn.get_score_matrix(
    query_embeds=train_query_dense_embeds, 
    dict_embeds=train_dict_dense_embeds
)
train_dense_candidate_idxs = biosyn.retrieve_candidate(
    score_matrix=train_dense_score_matrix, 
    topk=topk
)


biosyn.embed_and_build_faiss(batch_size=4096, dictionary_names=names_in_train_dictionary)
cand_idxs = biosyn.embed_queries_with_search(batch_size=4096, queries_names=names_in_train_queries)


print(f"train_dense_candidate_idxs shape: {train_dense_candidate_idxs.shape}, type: {train_dense_candidate_idxs.dtype}")
print(f"cand_idxs shape: {cand_idxs.shape}, type: {cand_idxs.dtype}")
