<h3 align="center"> BioSyn Replicate </h3>


## Install datasets:

You can use install_ds.sh to download the dataset into folder data/data-ncbi-fair
Also, the bash script will download the requirement libraries

```bash
$ bash install_ds.bash
```


If you want to do it manually, you have to:

- Unzip train_dictionary.zip into data directory
- Unzip traindev.zip into data directory
- Create python virtual environment
- Install required libraries:

```bash
$ pip install torch
$ pip install faiss-gpu-cu12
$ pip install tqdm transformers requests psutil
```

If you don't have GPU, please install faiss-cpu instead of faiss-gpu-cu12:
```bash
$ pip install faiss-cpu
```

## Tokenize:

Before start the training, you should do the tokenization step by executing:

```bash

MODEL_NAME_OR_PATH=dmis-lab/biobert-base-cased-v1.1
TRAIN_DICTIONARY_PATH=./data/data-ncbi-fair/train_dictionary.txt
TRAIN_DIR=./data/data-ncbi-fair/traindev
TOKENIZED_OUTPUT_DIR=./data/output/tokenized

CUDA_VISIBLE_DEVICES=1 python tokenizer.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --train_dictionary_path ${TRAIN_DICTIONARY_PATH} \
    --train_dir ${TRAIN_DIR} \
    --tokenizer_output_dir ${TOKENIZED_OUTPUT_DIR} \
    --max_length 25 \
```

When the execution finished, you should have a folder inside the specified TOKENIZED_OUTPUT_DIR containing two folders:
- dictionary
- queries

Each folder should have:
- .json file containing configuration
- __inputs.mmap contatining np array of input ids
- __att.mmap contatining np array of attention mask



## Train:

```bash

MODEL_NAME_OR_PATH=dmis-lab/biobert-base-cased-v1.1
TRAIN_DICTIONARY_PATH=./data/data-ncbi-fair/train_dictionary.txt
TRAIN_DIR=./data/data-ncbi-fair/traindev
OUTPUT_DIR=./data/output

CUDA_VISIBLE_DEVICES=1 python train.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --train_dictionary_path ${TRAIN_DICTIONARY_PATH} \
    --train_dir ${TRAIN_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --use_cuda \
    --topk 20 \
    --epoch 10 \
    --train_batch_size 48\
    --learning_rate 1e-5 \
    --max_length 25 \
```



```




