import os
import pickle
import logging
import torch
import numpy as np
import time
from tqdm import tqdm
from torch import nn
from torch.utils.data.dataloader import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    default_data_collator
)
from .rerankNet import RerankNet
# from huggingface_hub import hf_hub_url, cached_download
# from .sparse_encoder import SparseEncoder
import json

import faiss
#THIS LINE IS SO SO SO IMPORTANT !!!!
import faiss.contrib.torch_utils


LOGGER = logging.getLogger()

class NamesDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

class BioSyn(object):
    """
    Wrapper class for dense encoder and sparse encoder

    COMMENTED LINES ARE FOR DROPING SPARSE ENCODING
    """

    def __init__(self, max_length, use_cuda, topk):
        self.max_length = max_length
        self.use_cuda = use_cuda

        self.tokenizer = None
        self.encoder = None
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.topk = topk

        # self.sparse_encoder = None
        # self.sparse_weight = None

        # if initial_sparse_weight != None:
            # self.sparse_weight = self.init_sparse_weight(initial_sparse_weight)
        
    # def init_sparse_weight(self, initial_sparse_weight):
    #     """
    #     Parameters
    #     ----------
    #     initial_sparse_weight : float
    #         initial sparse weight
    #     """
    #     if self.use_cuda:
    #         self.sparse_weight = nn.Parameter(torch.empty(1).cuda())
    #     else:
    #         self.sparse_weight = nn.Parameter(torch.empty(1))
    #     self.sparse_weight.data.fill_(initial_sparse_weight) # init sparse_weight

    #     return self.sparse_weight

    # def init_sparse_encoder(self, corpus):
    #     self.sparse_encoder = SparseEncoder().fit(corpus)

    #     return self.sparse_encoder

    def get_dense_encoder(self):
        assert (self.encoder is not None)

        return self.encoder

    def get_dense_tokenizer(self):
        assert (self.tokenizer is not None)

        return self.tokenizer

    # def get_sparse_encoder(self):
    #     assert (self.sparse_encoder is not None)
    #     return self.sparse_encoder

    # def get_sparse_weight(self):
    #     assert (self.sparse_weight is not None)
    #     return self.sparse_weight

    def save_model(self, path):
        # save dense encoder
        self.encoder.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        # save sparse encoder
        # sparse_encoder_path = os.path.join(path,'sparse_encoder.pk')
        # self.sparse_encoder.save_encoder(path=sparse_encoder_path)

        # sparse_weight_file = os.path.join(path,'sparse_weight.pt')
        # torch.save(self.sparse_weight, sparse_weight_file)
        # logging.info("Sparse weight saved in {}".format(sparse_weight_file))

    def load_model(self, model_name_or_path):
        self.load_dense_encoder(model_name_or_path)
        # self.load_sparse_encoder(model_name_or_path)
        # self.load_sparse_weight(model_name_or_path)

        return self

    def load_dense_encoder(self, model_name_or_path):
        self.encoder = AutoModel.from_pretrained(model_name_or_path, use_safetensors=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.hidden_size = getattr(getattr(self.encoder, "config", None), "hidden_size", None)
        
        if self.use_cuda:
            self.encoder = self.encoder.to("cuda")
            self.encoder = torch.compile(self.encoder)
        return self.encoder, self.tokenizer
    
    # def load_sparse_encoder(self, model_name_or_path):
    #     sparse_encoder_path = os.path.join(model_name_or_path,'sparse_encoder.pk')
    #     # check file exists
    #     if not os.path.isfile(sparse_encoder_path):
    #         # download from huggingface hub and cache it
    #         sparse_encoder_url = hf_hub_url(model_name_or_path, filename="sparse_encoder.pk")
    #         sparse_encoder_path = cached_download(sparse_encoder_url)

    #     self.sparse_encoder = SparseEncoder().load_encoder(path=sparse_encoder_path)

    #     return self.sparse_encoder
    
    # def load_sparse_weight(self, model_name_or_path):
    #     sparse_weight_path = os.path.join(model_name_or_path,'sparse_weight.pt')
    #     # check file exists
    #     if not os.path.isfile(sparse_weight_path):
    #         # download from huggingface hub and cache it
    #         sparse_weight_url = hf_hub_url(model_name_or_path, filename="sparse_weight.pt")
    #         sparse_weight_path = cached_download(sparse_weight_url)

    #     self.sparse_weight = torch.load(sparse_weight_path)

    #     return self.sparse_weight

    def get_score_matrix(self, query_embeds, dict_embeds):
        """
        Return score matrix
        Parameters
        ----------
        query_embeds : np.array
            2d numpy array of query embeddings
        dict_embeds : np.array
            2d numpy array of query embeddings
        Returns
        -------
        score_matrix : np.array
            2d numpy array of scores
        """
        score_matrix = np.matmul(query_embeds, dict_embeds.T)
        
        return score_matrix

    def retrieve_candidate(self, score_matrix, topk):
        """
        Return sorted topk idxes (descending order)
        Parameters
        ----------
        score_matrix : np.array
            2d numpy array of scores
        topk : int
            The number of candidates
        Returns
        -------
        topk_idxs : np.array
            2d numpy array of scores [# of query , # of dict]
        """
        
        def indexing_2d(arr, cols):
            rows = np.repeat(np.arange(0,cols.shape[0])[:, np.newaxis],cols.shape[1],axis=1)
            return arr[rows, cols]

        # get topk indexes without sorting
        topk_idxs = np.argpartition(score_matrix,-topk)[:, -topk:]

        # get topk indexes with sorting
        topk_score_matrix = indexing_2d(score_matrix, topk_idxs)
        topk_argidxs = np.argsort(-topk_score_matrix) 
        topk_idxs = indexing_2d(topk_idxs, topk_argidxs)

        return topk_idxs

    # def embed_sparse(self, names, show_progress=False):
    #     """
    #     Embedding data into sparse representations
    #     Parameters
    #     ----------
    #     names : np.array
    #         An array of names
    #     Returns
    #     -------
    #     sparse_embeds : np.array
    #         A list of sparse embeddings
    #     """
    #     batch_size=1024
    #     sparse_embeds = []
        
    #     if show_progress:
    #         iterations = tqdm(range(0, len(names), batch_size))
    #     else:
    #         iterations = range(0, len(names), batch_size)
        
    #     for start in iterations:
    #         end = min(start + batch_size, len(names))
    #         batch = names[start:end]
    #         batch_sparse_embeds = self.sparse_encoder(batch)
    #         batch_sparse_embeds = batch_sparse_embeds.numpy()
    #         sparse_embeds.append(batch_sparse_embeds)
    #     sparse_embeds = np.concatenate(sparse_embeds, axis=0)

    #     return sparse_embeds

    def embed_dense(self, names, show_progress=False):
        """
        Embedding data into dense representations

        Parameters
        ----------
        names : np.array or list
            An array of names

        Returns
        -------
        dense_embeds : list
            A list of dense embeddings
        """
        self.encoder.eval() # prevent dropout
        
        batch_size=1024
        dense_embeds = []

        if isinstance(names, np.ndarray):
            names = names.tolist()        
        name_encodings = self.tokenizer(names, padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt")
        if self.use_cuda:
            name_encodings = name_encodings.to('cuda')
        name_dataset = NamesDataset(name_encodings)
        name_dataloader = DataLoader(name_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=batch_size)

        with torch.no_grad():
            for batch in tqdm(name_dataloader, disable=not show_progress, desc='embedding dictionary'):
                outputs = self.encoder(**batch)
                batch_dense_embeds = outputs[0][:,0].cpu().detach().numpy() # [CLS] representations
                dense_embeds.append(batch_dense_embeds)
        dense_embeds = np.concatenate(dense_embeds, axis=0)
        
        return dense_embeds
    
    
    # additional functions for faiss
    
    def embed_queries_with_search(self, queries_names, batch_size=64):
        """
            use the tokens of queries saved in memmap file before and embed them 
            the embeddings are saved in memmap file of queries_embed_mmap_base
        """
        amp_dtype= torch.float16

        if isinstance(queries_names, np.ndarray):
            queries_names = queries_names.tolist()


        #this is a bottleneck because tokenization should be before the training loop
        queries_tokens = self.tokenizer(
                queries_names,
                padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt")

        queries_tokens["input_ids"] = queries_tokens["input_ids"].pin_memory()
        queries_tokens["attention_mask"] = queries_tokens["attention_mask"].pin_memory()

        N = len(queries_names)
        #start embeding and search
        candidates_idxs = []
        self.encoder.eval()
        with torch.inference_mode():
            for start in tqdm(range(0,N, batch_size), desc="embeding and search index for candidates", unit="bach"):
                end = min(start+batch_size, N)



                chunk_input_ids = queries_tokens["input_ids"][start:end]
                chunk_att_mask = queries_tokens["attention_mask"][start:end]

                chunk_input_ids = chunk_input_ids.to(device=self.device, dtype=torch.long)
                chunk_att_mask = chunk_att_mask.to(device=self.device, dtype=torch.long)
                
                if self.use_cuda:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        out_chunk = self.encoder(
                            input_ids= chunk_input_ids,
                            attention_mask=chunk_att_mask,
                            return_dict=True
                        )[0][:,0] # cls (chunk_size, hidden_size)
                else:
                    out_chunk = self.encoder(
                        input_ids= chunk_input_ids,
                        attention_mask=chunk_att_mask,
                        return_dict=True
                    )[0][:,0] # cls (chunk_size, hidden_size)

                assert out_chunk is not None
                out_chunk = out_chunk.contiguous()
                # out_chunk = out_chunk.float().cpu().numpy()

                _, chunk_cand_idxs = self.faiss_index.search(out_chunk, self.topk)
                if self.use_cuda:
                    chunk_cand_idxs = chunk_cand_idxs.cpu().numpy()
                candidates_idxs.append(chunk_cand_idxs)
                del chunk_cand_idxs, out_chunk
        return np.vstack(candidates_idxs)



    def embed_and_build_faiss(self, dictionary_names, batch_size=64):
        amp_dtype= torch.float16


        if isinstance(dictionary_names, np.ndarray):
            dictionary_names = dictionary_names.tolist()
        dictionary_tokens = self.tokenizer(dictionary_names, padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt")


        dictionary_tokens["input_ids"]= dictionary_tokens["input_ids"].pin_memory()
        dictionary_tokens["attention_mask"]= dictionary_tokens["attention_mask"].pin_memory()


        N = len(dictionary_names)
        hidden_size = self.hidden_size



        if self.use_cuda:
            gpu_resources = faiss.StandardGpuResources()
            #Index configurations
            index_conf = faiss.GpuIndexFlatConfig()
            index_conf.device = torch.cuda.current_device()
            index_conf.useFloat16 = bool(self.use_cuda)

            #make the index (this index is on gpu)
            index = faiss.GpuIndexFlatIP(gpu_resources, hidden_size, index_conf)
        else:
            #make normal cpu index 
            index = faiss.IndexFlatIP(hidden_size)

        assert index is not None
        

        self.encoder.eval()

        # I am not using grad graphs here in this embedings for building the faiss index
        with torch.inference_mode():
            for start in tqdm(range(0,N, batch_size), desc="embeding and building faiss", unit="bach"):
                end = min(start+batch_size, N)

                # chunking then embeding
                chunk_input_ids = dictionary_tokens["input_ids"][start:end].to(self.device, dtype=torch.long)
                chunk_att_mask = dictionary_tokens["attention_mask"][start:end].to(self.device, dtype=torch.long)


                if self.use_cuda:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        out_chunk = self.encoder(
                            input_ids= chunk_input_ids,
                            attention_mask=chunk_att_mask,
                            return_dict=True
                        )[0][:,0] # cls (chunk_size, hidden_size)
                else:
                    out_chunk = self.encoder(
                        input_ids= chunk_input_ids,
                        attention_mask=chunk_att_mask,
                        return_dict=True
                    )[0][:,0] # cls (chunk_size, hidden_size)

                assert out_chunk is not None
                out_chunk = out_chunk.contiguous()
                # out_chunk = out_chunk.float().cpu().numpy()
                index.add(out_chunk)
                del out_chunk, chunk_input_ids,chunk_att_mask

        self.faiss_index = index

