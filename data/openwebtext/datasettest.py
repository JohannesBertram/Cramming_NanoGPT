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
log_file = 'vocab_analysis.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s')

# Define the encoding function for case-sensitive dataset
def process_case_sensitive(example):
    text = example['text']
    ids = enc.encode_ordinary(text)    # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token)          # Add the end of text token, e.g., 50256 for gpt2 bpe
    return {'ids': ids}

# Define the encoding function for case-insensitive dataset
def process_case_insensitive(example):
    text_lower = example['text'].casefold()  # Use casefold for more aggressive lowercasing
    ids = enc.encode_ordinary(text_lower)    # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token)                # Add the end of text token, e.g., 50256 for gpt2 bpe
    return {'ids': ids}

def count_tokens(token_list):
    counter = collections.Counter()
    for tokens in token_list:
        counter.update(tokens)
    return counter

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

    # Tokenize the case-insensitive dataset
    tokenized_ci = split_dataset.map(
        process_case_insensitive,
        remove_columns=['text'],
        desc="Tokenizing the splits (case-insensitive)",
        num_proc=num_proc,
    )

    # Count the frequency of each token in the tokenized datasets
    token_counts_cs = collections.Counter()
    token_counts_ci = collections.Counter()

    for split, dset in tokenized_cs.items():
        for example in tqdm(dset, desc=f"Counting tokens in {split} split (case-sensitive)"):
            token_counts_cs.update(example['ids'])

    for split, dset in tokenized_ci.items():
        for example in tqdm(dset, desc=f"Counting tokens in {split} split (case-insensitive)"):
            token_counts_ci.update(example['ids'])

    # Determine the cumulative frequency distribution for case-sensitive tokens
    total_tokens_cs = sum(token_counts_cs.values())
    sorted_tokens_cs = sorted(token_counts_cs.items(), key=lambda x: x[1], reverse=True)

    cumulative_frequency_cs = 0
    desired_vocab_size_cs = 50304  # Fixed vocabulary size for case-sensitive

    # Calculate coverage for the case-sensitive model
    coverage_cs = 0
    for i, (token, count) in enumerate(sorted_tokens_cs):
        cumulative_frequency_cs += count
        if i + 1 == desired_vocab_size_cs:
            coverage_cs = cumulative_frequency_cs / total_tokens_cs
            break

    logging.info(f"Coverage for case-sensitive model with 50304 tokens: {coverage_cs*100:.2f}%")

    # Determine the cumulative frequency distribution for case-insensitive tokens
    total_tokens_ci = sum(token_counts_ci.values())
    sorted_tokens_ci = sorted(token_counts_ci.items(), key=lambda x: x[1], reverse=True)

    cumulative_frequency_ci = 0
    coverage_thresholds = [0.95, 0.97, 0.99]  # Coverage thresholds to test
    vocab_sizes_ci = {}

    for threshold in coverage_thresholds:
        cumulative_frequency_ci = 0
        vocab_size_ci = 0
        for token, count in sorted_tokens_ci:
            cumulative_frequency_ci += count
            vocab_size_ci += 1
            if cumulative_frequency_ci / total_tokens_ci >= threshold:
                vocab_sizes_ci[threshold] = vocab_size_ci
                break

    for threshold, size in vocab_sizes_ci.items():
        logging.info(f"Vocabulary size for {threshold*100}% coverage (case-insensitive): {size}")
