import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

# Number of workers in .map() call
num_proc = 8

# Number of workers in load_dataset() call
num_proc_load_dataset = num_proc

# Get the GPT-2 encoding
enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # Load the OpenWebText dataset
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)

    # Create a test split
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # Rename the test split to val

    # Define the encoding function for case-insensitive dataset
    def process_case_insensitive(example):
        text_lower = example['text'].lower()  # Convert to lowercase
        ids = enc.encode_ordinary(text_lower) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # Add the end of text token, e.g. 50256 for gpt2 bpe
        out = {'ids': ids, 'len': len(ids)}
        return out

    # Tokenize the case-insensitive dataset
    tokenized_ci = split_dataset.map(
        process_case_insensitive,
        remove_columns=['text'],
        desc="tokenizing the splits (case-insensitive)",
        num_proc=num_proc,
    )

    # Function to save tokenized dataset to binary file
    def save_tokenized_dataset(tokenized_dataset, prefix):
        for split, dset in tokenized_dataset.items():
            arr_len = np.sum(dset['len'], dtype=np.uint64)
            filename = os.path.join(os.path.dirname(__file__), f'{prefix}_{split}.bin')
            dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
            total_batches = 1024

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                # Batch together samples for faster write
                batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                arr_batch = np.concatenate(batch['ids'])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    # Save the tokenized datasets
    save_tokenized_dataset(tokenized_ci, 'ci')  # Case-insensitive

"""import os
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
        arr.flush()"""