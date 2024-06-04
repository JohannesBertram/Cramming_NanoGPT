import numpy as np
import os

# Paths to the case-insensitive train.bin and val.bin files
train_bin_path = 'ci_train.bin'
val_bin_path = 'ci_val.bin'

# Read tokens from binary files
def read_tokens(file_path):
    dtype = np.uint16  # Assuming the tokens are stored as uint16
    tokens = np.fromfile(file_path, dtype=dtype)
    return tokens

train_tokens = read_tokens(train_bin_path)
val_tokens = read_tokens(val_bin_path)

# Combine tokens from both datasets
all_tokens = np.concatenate((train_tokens, val_tokens))

# Get the unique tokens
unique_tokens = np.unique(np.append(np.append(np.unique(all_tokens[:int(np.round(0.3*len(all_tokens)))]), 
                                              np.unique(all_tokens[int(np.round(0.3*len(all_tokens))):int(np.round(0.6*len(all_tokens)))])), 
                                    np.unique(all_tokens[int(np.round(0.3*len(all_tokens))):int(np.round(0.6*len(all_tokens)))])))

# Ensure the number of unique tokens does not exceed the smaller vocabulary size
smaller_vocab_size = 33408
if len(unique_tokens) > smaller_vocab_size:
    raise ValueError(f"Number of unique tokens {len(unique_tokens)} exceeds the smaller vocabulary size {smaller_vocab_size}")

# Create a mapping dictionary
token_mapping = {token: idx for idx, token in enumerate(unique_tokens)}

# Save the reverse mapping to remap back later
reverse_mapping = {v: k for k, v in token_mapping.items()}

# Save the reverse mapping to a JSON file
reverse_mapping_path = 'reverse_mapping.json'
with open(reverse_mapping_path, 'w') as f:
    json.dump(reverse_mapping, f)

# Apply the token mapping to both datasets
def map_tokens(tokens, token_mapping):
    mapped_tokens = np.vectorize(token_mapping.get)(tokens)
    return mapped_tokens

mapped_train_tokens = map_tokens(train_tokens, token_mapping)
mapped_val_tokens = map_tokens(val_tokens, token_mapping)

# Save the mapped tokens to new binary files
mapped_train_bin_path = 'mapped_train.bin'
mapped_val_bin_path = 'mapped_val.bin'

mapped_train_tokens.tofile(mapped_train_bin_path)
mapped_val_tokens.tofile(mapped_val_bin_path)
