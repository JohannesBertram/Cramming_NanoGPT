import os
import collections
import logging
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# Number of workers in .map() call
num_proc = 8

# Number of workers in load_dataset() call
num_proc_load_dataset = num_proc

# Get the GPT-2 encoding
enc = tiktoken.get_encoding("gpt2")

# Set up logging
log_file = 'vocab_analysis_case_sensitive.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s')

# Define the encoding function for case-sensitive dataset
def process_case_sensitive(example):
    text = example['text']
    ids = enc.encode_ordinary(text)    # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token)          # Add the end of text token, e.g., 50256 for gpt2 bpe
    return {'ids': ids}

if __name__ == '__main__':
    # Load the OpenWebText dataset
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)

    # Create a test split
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')  # Rename the test split to val

    # Tokenize the case-sensitive dataset
    tokenized_cs = split_dataset.map(
        process_case_sensitive,
        remove_columns=['text'],
        desc="Tokenizing the splits (case-sensitive)",
        num_proc=num_proc,
    )

    # Count the frequency of each token in the tokenized datasets
    token_counts_cs = collections.Counter()

    for split, dset in tokenized_cs.items():
        for example in tqdm(dset, desc=f"Counting tokens in {split} split (case-sensitive)"):
            token_counts_cs.update(example['ids'])

    # Determine the cumulative frequency distribution for case-sensitive tokens
    total_tokens_cs = sum(token_counts_cs.values())
    sorted_tokens_cs = sorted(token_counts_cs.items(), key=lambda x: x[1], reverse=True)

    cumulative_frequency_cs = 0
    desired_vocab_size_cs = 50304  # Fixed vocabulary size for case-sensitive

    logging.info(f"Total number of tokens (case-sensitive): {total_tokens_cs}")

    # Calculate coverage for the case-sensitive model
    for i, (token, count) in enumerate(sorted_tokens_cs):
        if i < desired_vocab_size_cs:
            cumulative_frequency_cs += count
            if i % 1000 == 0:  # Log every 1000 tokens
                logging.info(f"Token {i}: count = {count}, cumulative_frequency_cs = {cumulative_frequency_cs}")

    coverage_cs = cumulative_frequency_cs / total_tokens_cs
    logging.info(f"Cumulative frequency (case-sensitive): {cumulative_frequency_cs}")
    logging.info(f"Coverage for case-sensitive model with 50304 tokens: {coverage_cs*100:.2f}%")
