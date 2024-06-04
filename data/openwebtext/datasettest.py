import numpy as np
import os

# Path to the case-insensitive train.bin file
train_bin_path = 'ci_train.bin'

def count_unique_tokens_in_bin(file_path):
    dtype = np.uint16  # Assuming the tokens are stored as uint16
    tokens = np.fromfile(file_path, dtype=dtype)
    tokens1 = tokens[:int(np.round(0.25*len(tokens)))]
    tokens2 = tokens[int(np.round(0.25*len(tokens))):int(np.round(0.5*len(tokens)))]
    tokens3 = tokens[int(np.round(0.5*len(tokens))):int(np.round(0.75*len(tokens)))]
    tokens4 = tokens[int(np.round(0.75*len(tokens))):]
    unique_tokens1 = set(np.unique(tokens1))
    print("1")
    unique_tokens2 = set(np.unique(tokens2))
    unique_tokens3 = set(np.unique(tokens3))
    unique_tokens4 = set(np.unique(tokens4))
    vocab_size = len(unique_tokens1.union(unique_tokens2).union(unique_tokens3).union(unique_tokens4))
    return vocab_size

# Get the number of distinct tokens
vocab_size = count_unique_tokens_in_bin(train_bin_path)

# Print the result
print(f"Vocabulary size (number of distinct tokens) in the case-insensitive dataset: {vocab_size}")
