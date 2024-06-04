import numpy as np
import os

# Path to the case-insensitive train.bin file
train_bin_path = 'ci_train.bin'

# Ensure the file exists
if not os.path.isfile(train_bin_path):
    raise FileNotFoundError(f"File not found: {train_bin_path}")

# Read the binary file and calculate the number of tokens
def count_tokens_in_bin(file_path):
    dtype = np.uint16  # Assuming the tokens are stored as uint16
    tokens = np.fromfile(file_path, dtype=dtype)
    num_tokens = len(tokens)
    return num_tokens

# Get the number of tokens
num_tokens = count_tokens_in_bin(train_bin_path)

# Print the result
print(f"Number of tokens in the case-insensitive dataset: {num_tokens}")
