import numpy as np
import torch
import argparse
import logging
import time
import pdb
import os
import json
import random
from utils import (
    evaluate
)
from tqdm import tqdm
from src.biosyn import (
    QueryDataset, 
    CandidateDataset, 
    DictionaryDataset,
    TextPreprocess, 
    RerankNet, 
    BioSyn
)
import config

torch.set_float32_matmul_precision('high')

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

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for output')

    parser.add_argument('--tokenizer_output_dir', type=str, required=True,
                        help='Directory for tokenizer')
    
    # Tokenizer settings
    parser.add_argument('--max_length', default=25, type=int)

    # Train config
    parser.add_argument('--seed',  type=int, 
                        default=0)
    parser.add_argument('--use_cuda',  action="store_true")
    #Draft argument was moved to tokenizer.py
    # parser.add_argument('--draft',  action="store_true")
    parser.add_argument('--topk',  type=int, 
                        default=20)
    parser.add_argument('--learning_rate',
                        help='learning rate',
                        default=0.0001, type=float)
    parser.add_argument('--weight_decay',
                        help='weight decay',
                        default=0.01, type=float)
    parser.add_argument('--train_batch_size',
                        help='train batch size',
                        default=16, type=int)
    parser.add_argument('--epoch',
                        help='epoch to train',
                        default=10, type=int)
    # parser.add_argument('--initial_sparse_weight',
    #                     default=0, type=float)

    # parser.add_argument('--dense_ratio', type=float,
    #                     default=0.5)
    
    parser.add_argument('--save_checkpoint_all', action="store_true")

    args = parser.parse_args()
    return args

def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000

    LOGGER.info("Using seed={}, pid={}".format(seed, os.getpid()))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    

def train(args, data_loader, model):
    LOGGER.info("train!")
    
    train_loss = 0
    train_steps = 0
    model.train()
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        model.optimizer.zero_grad()
        batch_x, batch_y = data
        batch_pred = model(batch_x)  
        loss = model.get_loss(batch_pred, batch_y)
        loss.backward()
        model.optimizer.step()
        train_loss += loss.item()
        train_steps += 1

    train_loss /= (train_steps + 1e-9)
    return train_loss
    
def main(args):
    init_logging()
    init_seed(args.seed)
    print(args)

    # prepare for output
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    queries_dir, dictionary_dir = config.queries_dir , config.dictionary_dir
    queries_files_prefix, dictionary_files_prefix =  config.queries_files_prefix , config.dictionary_files_prefix
    ids_file_suffix,tokens_inputs_file_suffix, tokens_attentions_file_suffix = config.ids_file_suffix, config.tokens_inputs_file_suffix, config.tokens_attentions_file_suffix

    tokenizer_output_dir = args.tokenizer_output_dir
    query_tokenized_dir = tokenizer_output_dir + queries_dir
    dictionary_tokenized_dir = tokenizer_output_dir + dictionary_dir
    query_tokenized_mmap_base = query_tokenized_dir + queries_files_prefix
    dictionary_tokenized_mmap_base = dictionary_tokenized_dir + dictionary_files_prefix
    


    #senity check
    assert os.path.isfile(query_tokenized_mmap_base + ids_file_suffix), f"Please execute tokenizer.py before"

    query_ids = np.load(query_tokenized_mmap_base + ids_file_suffix)
    dictionary_ids = np.load(dictionary_tokenized_mmap_base + ids_file_suffix )







    # prepare for data loader of train and dev
    train_set = CandidateDataset(
        query_ids = query_ids, 
        dictionary_ids = dictionary_ids, 
        topk = args.topk, 
        max_length=args.max_length,
        query_tokenized_mmap_base=query_tokenized_mmap_base,
        dictionary_tokenized_mmap_base=dictionary_tokenized_mmap_base,
        tokens_inputs_file_suffix=tokens_inputs_file_suffix,
        tokens_attentions_file_suffix=tokens_attentions_file_suffix
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.train_batch_size,
        shuffle=True,
    )

    # load BERT tokenizer, dense_encoder, sparse_encoder
    biosyn = BioSyn(
        max_length=args.max_length,
        use_cuda=args.use_cuda,
        topk=args.topk,
        tokens=train_set.tokens
    )
    biosyn.load_dense_encoder(
        model_name_or_path=args.model_name_or_path
    )

    # load rerank model
    model = RerankNet(
        learning_rate=args.learning_rate, 
        weight_decay=args.weight_decay,
        encoder = biosyn.get_dense_encoder(),
        use_cuda=args.use_cuda
    )



    start = time.time()
    for epoch in range(1,args.epoch+1):
        LOGGER.info("Epoch {}/{}".format(epoch,args.epoch))
        LOGGER.info("train_set dense embedding for iterative candidate retrieval")

        biosyn.embed_and_build_faiss(batch_size=4096)
        cand_idxs = biosyn.embed_queries_with_search(batch_size=4096)
        train_set.set_dense_candidate_idxs(d_candidate_idxs=cand_idxs)


        # train
        train_loss = train(args, data_loader=train_loader, model=model)
        LOGGER.info('loss/train_per_epoch={}/{}'.format(train_loss,epoch))


        # save model every epoch
        if args.save_checkpoint_all:
            checkpoint_dir = os.path.join(args.output_dir, "checkpoint_{}".format(epoch))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            biosyn.save_model(checkpoint_dir)

        # save model last epoch
        if epoch == args.epoch:
            biosyn.save_model(args.output_dir)

    end = time.time()
    training_time = end-start
    training_hour = int(training_time/60/60)
    training_minute = int(training_time/60 % 60)
    training_second = int(training_time % 60)
    LOGGER.info("Training Time!{} hours {} minutes {} seconds".format(training_hour, training_minute, training_second))
    
if __name__ == '__main__':
    args = parse_args()
    main(args)





# python train.py --use_cuda --tokenizer_output_dir='./data/output/tokenized'  --model_name_or_path='dmis-lab/biobert-base-cased-v1.1'    --train_dictionary_path="./data/data-ncbi-fair/train_dictionary.txt"  --train_dir="./data/data-ncbi-fair/traindev" --output_dir="./data/output"  --train_batch_size=32


