import numpy as np
import os
import json

# Paths to the case-insensitive train.bin and val.bin files
train_bin_path = 'ci_train.bin'
val_bin_path = 'ci_val.bin'

# Chunk size to read at a time
chunk_size = 10**6  # Adjust this based on available memory

# Function to read tokens in chunks
def read_tokens_in_chunks(file_path, chunk_size):
    dtype = np.uint16
    with open(file_path, 'rb') as f:
        while True:
            chunk = np.fromfile(f, dtype=dtype, count=chunk_size)
            if not chunk.size:
                break
            yield chunk

# Function to get unique tokens from chunks
def get_unique_tokens(file_path, chunk_size):
    unique_tokens = set()
    for chunk in read_tokens_in_chunks(file_path, chunk_size):
        unique_tokens.update(chunk)
    return unique_tokens

# Get unique tokens from both train and val datasets
train_unique_tokens = get_unique_tokens(train_bin_path, chunk_size)
val_unique_tokens = get_unique_tokens(val_bin_path, chunk_size)

# Combine unique tokens from both datasets
all_unique_tokens = np.unique(np.array(list(train_unique_tokens | val_unique_tokens)))

# Ensure the number of unique tokens does not exceed the smaller vocabulary size
smaller_vocab_size = 33408
if len(all_unique_tokens) > smaller_vocab_size:
    raise ValueError(f"Number of unique tokens {len(all_unique_tokens)} exceeds the smaller vocabulary size {smaller_vocab_size}")

# Create a mapping dictionary
token_mapping = {token: idx for idx, token in enumerate(all_unique_tokens)}

# Save the reverse mapping to remap back later
reverse_mapping = {v: k for k, v in token_mapping.items()}

# Save the reverse mapping to a JSON file
reverse_mapping_path = 'reverse_mapping.json'
with open(reverse_mapping_path, 'w') as f:
    json.dump(reverse_mapping, f)

# Function to map and save tokens in chunks
def map_and_save_tokens(file_path, mapped_file_path, token_mapping, chunk_size):
    dtype = np.uint16
    with open(file_path, 'rb') as f_in, open(mapped_file_path, 'wb') as f_out:
        while True:
            chunk = np.fromfile(f_in, dtype=dtype, count=chunk_size)
            if not chunk.size:
                break
            mapped_chunk = np.vectorize(token_mapping.get)(chunk)
            mapped_chunk.tofile(f_out)

# Map and save tokens for both train and val datasets
mapped_train_bin_path = 'mapped_train.bin'
mapped_val_bin_path = 'mapped_val.bin'

map_and_save_tokens(train_bin_path, mapped_train_bin_path, token_mapping, chunk_size)
map_and_save_tokens(val_bin_path, mapped_val_bin_path, token_mapping, chunk_size)
