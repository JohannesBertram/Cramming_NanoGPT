import numpy as np
import os

# Path to the case-insensitive train.bin file
train_bin_path = 'ci_train.bin'

def count_unique_tokens_in_bin(file_path):
    dtype = np.uint16  # Assuming the tokens are stored as uint16
    tokens = np.fromfile(file_path, dtype=dtype)
    tokens = tokens[:np.round(0.25*len(tokens))]
    unique_tokens = np.unique(tokens)
    vocab_size = len(unique_tokens)
    return vocab_size

# Get the number of distinct tokens
vocab_size = count_unique_tokens_in_bin(train_bin_path)

# Print the result
print(f"Vocabulary size (number of distinct tokens) in the case-insensitive dataset: {vocab_size}")
