import re
import os
import glob
import numpy as np
from torch.utils.data import Dataset
import logging
from tqdm import tqdm
import json
import torch


LOGGER = logging.getLogger(__name__)


class QueryDataset(Dataset):

    def __init__(self, data_dir, 
                filter_composite=False,
                filter_duplicate=False,
                filter_cuiless=False
        ):
        """       
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_composite : bool
            filter composite mentions
        filter_duplicate : bool
            filter duplicate queries
        draft : bool
            use subset of queries for debugging (default False)     
        """
        LOGGER.info("QueryDataset! data_dir={} filter_composite={} filter_duplicate={} filter_cuiless={}".format(
            data_dir, filter_composite, filter_duplicate, filter_cuiless
        ))
        
        self.data = self.load_data(
            data_dir=data_dir,
            filter_composite=filter_composite,
            filter_duplicate=filter_duplicate,
            filter_cuiless=filter_cuiless
        )
        
    def load_data(self, data_dir, filter_composite, filter_duplicate, filter_cuiless):
        """       
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_composite : bool
            filter composite mentions
        filter_duplicate : bool
            filter duplicate queries  
        filter_cuiless : bool
            remove samples with cuiless 
        Returns
        -------
        data : np.array 
            mention, cui pairs
        """
        data = []

        concept_files = glob.glob(os.path.join(data_dir, "*.concept"))
        for concept_file in tqdm(concept_files):
            with open(concept_file, "r", encoding='utf-8') as f:
                concepts = f.readlines()

            for concept in concepts:
                concept = concept.split("||")
                mention = concept[3].strip()
                cui = concept[4].strip()
                is_composite = (cui.replace("+","|").count("|") > 0)

                # filter composite cui
                if filter_composite and is_composite:
                    continue
                # filter cuiless
                if filter_cuiless and cui == '-1':
                    continue

                data.append((mention,cui))
        
        if filter_duplicate:
            data = list(dict.fromkeys(data))
        
        # return np.array data
        data = np.array(data)
        
        return data


class DictionaryDataset():
    """
    A class used to load dictionary data
    """
    def __init__(self, dictionary_path):
        """
        Parameters
        ----------
        dictionary_path : str
            The path of the dictionary
        draft : bool
            use only small subset
        """
        LOGGER.info("DictionaryDataset! dictionary_path={}".format(
            dictionary_path 
        ))
        self.data = self.load_data(dictionary_path)

    def load_data(self, dictionary_path):
        name_cui_map = {}
        data = []
        with open(dictionary_path, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip()
                if line == "": continue
                cui, name = line.split("||")
                data.append((name,cui))
        data = np.array(data)
        return data


class CandidateDataset(Dataset):
    """
    Candidate Dataset for:
        query_tokens, candidate_tokens, label
    """
    def __init__(
        self, 
        query_ids, 
        dictionary_ids, 
        max_length, 
        topk,
        query_tokenized_mmap_base,
        dictionary_tokenized_mmap_base,
        tokens_inputs_file_suffix,
        tokens_attentions_file_suffix
        ):
        """
        Retrieve top-k candidates based on sparse/dense embedding
        Parameters
        ----------
        query_ids : list
            A list of cuis (id)
        dictionary_ids : list
            A list of cuis (id)
        topk : int
            The number of candidates

        query_tokenized_mmap_base,
        dictionary_tokenized_mmap_base,
        tokens_inputs_file_suffix,
        tokens_attentions_file_suffix:
            Those are configuration strings should be the same as when you execute tokenizer.py
        """

        self.query_ids = query_ids
        self.dict_ids = dictionary_ids

        self.topk = topk
        self.n_dense = int(topk )
        self.max_length = max_length

        self.d_candidate_idxs = None


        queries_input_ids_mmap_path = query_tokenized_mmap_base + tokens_inputs_file_suffix
        queries_attention_mask_mmap_path = query_tokenized_mmap_base + tokens_attentions_file_suffix

        dictionary_input_ids_mmap_path = dictionary_tokenized_mmap_base + tokens_inputs_file_suffix
        dictionary_attention_mask_mmap_path = dictionary_tokenized_mmap_base + tokens_attentions_file_suffix

        self.tokens = {
            "dictionary_inputs":  np.memmap(
                dictionary_input_ids_mmap_path,
                mode="r+",
                dtype=np.int32,
                shape=self.load_mmap_shape(dictionary_tokenized_mmap_base) 
            ),
            "dictionary_attention":  np.memmap(
                dictionary_attention_mask_mmap_path,
                mode="r+",
                dtype=np.int32,
                shape=self.load_mmap_shape(dictionary_tokenized_mmap_base) 
            ),
            "query_inputs":  np.memmap(
                queries_input_ids_mmap_path,
                mode="r+",
                dtype=np.int32,
                shape=self.load_mmap_shape(query_tokenized_mmap_base) 
            ),
            "query_attention":  np.memmap(
                queries_attention_mask_mmap_path,
                mode="r+",
                dtype=np.int32,
                shape=self.load_mmap_shape(query_tokenized_mmap_base) 
            )
        }


        LOGGER.info("CandidateDataset: len(queries)={} len(dicts)={} topk={} ".format(
            self.tokens["query_inputs"].shape[0],self.tokens["dictionary_inputs"].shape[0], topk))


    def set_dense_candidate_idxs(self, d_candidate_idxs):
        self.d_candidate_idxs = d_candidate_idxs

    def __getitem__(self, query_idx):
        assert (self.d_candidate_idxs is not None)



        # query_name = self.query_names[query_idx]
        # query_token = self.tokenizer(query_name, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        query_token = {
            "input_ids":  torch.from_numpy(self.tokens["query_inputs"][query_idx]),
            "attention_mask": torch.from_numpy(self.tokens["query_attention"][query_idx]),
        }



        d_candidate_idx = self.d_candidate_idxs[query_idx]

        topk_candidate_idx = d_candidate_idx[:self.topk]

        # sanity check
        assert len(topk_candidate_idx) == self.topk
        assert len(topk_candidate_idx) == len(set(topk_candidate_idx))

        # candidate_names = [self.dict_names[candidate_idx] for candidate_idx in topk_candidate_idx]
        # candidate_tokens = self.tokenizer(candidate_names, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        candidate_tokens = {
            "input_ids": torch.from_numpy(self.tokens["dictionary_inputs"][topk_candidate_idx]),
            "attention_mask": torch.from_numpy(self.tokens["dictionary_attention"][topk_candidate_idx]),
        }


        labels = self.get_labels(query_idx, topk_candidate_idx).astype(np.float32)
        return (query_token, candidate_tokens), labels

    def __len__(self):
        return self.tokens["query_inputs"].shape[0]

    def check_label(self, query_id, candidate_id_set):
        label = 0

        query_ids = query_id.split("|")
        """
        All query ids should be included in dictionary id
        """
        for q_id in query_ids:
            if q_id in candidate_id_set:
                label = 1
                continue
            else:
                label = 0
                break
        return label

    def get_labels(self, query_idx, candidate_idxs):
        labels = np.array([])
        query_id = self.query_ids[query_idx]
        candidate_ids = np.array(self.dict_ids)[candidate_idxs]
        for candidate_id in candidate_ids:
            label = self.check_label(query_id, candidate_id)
            labels = np.append(labels, label)
        return labels


    def load_mmap_shape(self, base):
        with open(base+".json") as f:
            meta = json.load(f)
        return tuple(meta["shape"])


