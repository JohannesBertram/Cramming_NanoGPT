import os
import re
from tqdm import tqdm
import numpy as np
from nltk.tokenize import WhitespaceTokenizer
from datasets import load_dataset

# number of workers in .map() call
num_proc = 4

# number of workers in load_dataset() call
num_proc_load_dataset = num_proc

# tokenizer for preprocessed data
tokenizer = WhitespaceTokenizer()

def preprocess_text(text):
    # convert to lowercase and remove special characters
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    return cleaned_text

if __name__ == '__main__':
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)

    # create train and val splits
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    # preprocess the dataset
    preprocessed_dataset = split_dataset.map(lambda example: {'text': preprocess_text(example['text'])})

    # tokenize the preprocessed dataset
    def process(example):
        ids = [tokenizer.vocab_size + 1 if token == '<eos>' else tokenizer.token_to_id(token) for token in tokenizer.tokenize(example['text'])]
        ids.append(tokenizer.vocab_size)  # add end-of-sequence token
        out = {'ids': ids, 'len': len(ids)}
        return out

    tokenized = preprocessed_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}_lc.bin')
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()