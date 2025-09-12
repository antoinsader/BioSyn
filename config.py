import torch
use_cuda = torch.cuda.is_available() 

#[TOKENIZER]
queries_dir, dictionary_dir = '/queries', '/dictionary'
queries_files_prefix, dictionary_files_prefix = "/queries_", "/dictionary_"
ids_file_suffix,tokens_inputs_file_suffix, tokens_attentions_file_suffix = '_ids.npy',  '_inputids.mmap', '_attentionmask.mmap'


#[LOGGER]
global_log_path = "./global_log.json"
logs_dir = "./data/logs"