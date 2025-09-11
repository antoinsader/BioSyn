
import torch
import os
import time
import argparse
import logging
import json
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

from src.biosyn import (
    QueryDataset, 
    DictionaryDataset,
)



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


def tokenize_names_with_memmap(
    tokenizer,
    names,
    mmap_path_base,
    max_length,
    tokenized_inputs_suffix,
    tokenized_att_suffix,
    batch_size = 4096):

    if isinstance(names, np.ndarray):
        names  = names.tolist()
    names_size = len(names)

    #create mmap conf for input_ids and attention mask
    input_ids_mmap_path = mmap_path_base + tokenized_inputs_suffix
    attention_mask_mmap_path = mmap_path_base + tokenized_att_suffix


    input_ids_array = np.memmap(input_ids_mmap_path,
                                mode="w+",
                                dtype=np.int32,
                                shape=(names_size, max_length)
                                )
    att_mask_array = np.memmap(attention_mask_mmap_path,
                                mode="w+",
                                dtype=np.int32,
                                shape=(names_size, max_length)
                                )
    #saving meta 
    _meta = {"shape": (names_size, max_length)}
    with open(mmap_path_base + ".json", "w") as f:
        json.dump(_meta, f)

    #tokenize in epochs
    for start in tqdm(range(0, names_size, batch_size), desc="tokenizing", unit="batch"):
        end = min(start + batch_size , names_size)
        names_batch = names[start: end]
        tokens = tokenizer(
            names_batch,
            padding="max_length", 
            max_length=max_length, 
            truncation=True, 
            return_tensors="pt"
        )
        input_ids_array[start:end, : ] = tokens["input_ids"]
        att_mask_array[start:end, : ] = tokens["attention_mask"]
    input_ids_array.flush()
    att_mask_array.flush()
    return True




LOGGER = logging.getLogger()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Biosyn train')

    # Required
    parser.add_argument('--model_name_or_path', required=True,
                        help='Directory for pretrained model')
    parser.add_argument('--train_dictionary_path', type=str, required=True,
                    help='train dictionary path')
    parser.add_argument('--train_dir', type=str, required=True,
                    help='training set directory')
    parser.add_argument('--tokenizer_output_dir', type=str, required=True,
                        help='Directory for tokenizer output')
    parser.add_argument('--max_length', default=25, type=int)

    args = parser.parse_args()
    return args

def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)


def main(args):
    print(f"args: {args}")
    model_name_or_path,train_dictionary_path, train_dir, tokenizer_output_dir = args.model_name_or_path,args.train_dictionary_path, args.train_dir, args.tokenizer_output_dir
    max_length = args.max_length

    LOGGER.info(f"init tokenizer from {model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


    train_dictionary  = load_dictionary(train_dictionary_path)
    train_queries = load_queries(
        data_dir = train_dir,
        filter_composite=True,
        filter_duplicate=True,
        filter_cuiless=True
    )

    query_names = [row[0] for row in train_queries]
    dictionary_names = [row[0] for row in train_dictionary]


    query_tokenized_dir = tokenizer_output_dir + '/queries'
    dictionary_tokenized_dir = tokenizer_output_dir + '/dictionary'

    os.makedirs(query_tokenized_dir, exist_ok=True)
    os.makedirs(dictionary_tokenized_dir, exist_ok=True)

    query_tokenized_mmap_base = query_tokenized_dir + "/queries_"
    dictionary_tokenized_mmap_base = dictionary_tokenized_dir + "/dictionary_"


    t0 = time.time()


    tok_queries =  tokenize_names_with_memmap(
        tokenizer = tokenizer, 
        names=query_names, 
        mmap_path_base=query_tokenized_mmap_base, 
        max_length=max_length, 
        tokenized_inputs_suffix="_inputs.mmap", 
        tokenized_att_suffix="_att.mmap"
    )
    if tok_queries:
        LOGGER.info(f"queries has {len(query_names)} names, was tokenized in dir: {query_tokenized_dir}, took time: {time.time()-t0}s")

    t0 = time.time()
    tok_dicts = tokenize_names_with_memmap(
        tokenizer = tokenizer, 
        names=dictionary_names, 
        mmap_path_base=dictionary_tokenized_mmap_base, 
        max_length=max_length, 
        tokenized_inputs_suffix="_inputs.mmap", 
        tokenized_att_suffix="_att.mmap"
    )
    if tok_dicts:
        LOGGER.info(f"dictionary has {len(dictionary_names)} names, was tokenized in dir: {dictionary_tokenized_dir}, took time: {time.time()-t0}s")


if __name__ == '__main__':
    init_logging()
    LOGGER.info("tokenizing!")
    args = parse_args()
    main(args)


# python tokenizer.py --model_name_or_path='dmis-lab/biobert-base-cased-v1.1'  --train_dictionary_path='./data/data-ncbi-fair/train_dictionary.txt' --train_dir='./data/data-ncbi-fair/traindev' --tokenizer_output_dir='./data/output/tokenized/'






