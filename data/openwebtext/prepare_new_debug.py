import os
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer
from datasets import load_dataset  # huggingface datasets

# Number of workers in .map() call
num_proc = 8

# Number of workers in load_dataset() call
num_proc_load_dataset = num_proc

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

if __name__ == '__main__':
    # Load the OpenWebText dataset
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)

    # Create a test split
    split_dataset = dataset["train"].train_test_split(test_size=0.000005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')  # Rename the test split to val

    # Define the encoding function for the new tokenizer
    def process_with_new_tokenizer(example):
        text = example['text'].lower()  # Convert to lowercase if required
        ids = tokenizer.encode(text, add_special_tokens=True)  # Add special tokens
        out = {'ids': ids, 'len': len(ids)}
        return out

    # Tokenize only the validation set with the new tokenizer for debugging
    tokenized_val_dataset = split_dataset['val'].map(
        process_with_new_tokenizer,
        remove_columns=['text'],
        desc="tokenizing the validation set with new tokenizer",
        num_proc=num_proc,
    )

    # Function to save tokenized dataset to binary file
    def save_tokenized_dataset(tokenized_dataset, prefix):
        for split, dset in tokenized_dataset.items():
            arr_len = np.sum(dset['len'], dtype=np.uint64)
            filename = os.path.join(os.path.dirname(__file__), f'{prefix}_{split}.bin')
            dtype = np.uint32  # Change dtype if necessary for the new tokenizer
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
            total_batches = 1024

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                # Batch together samples for faster write
                batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                
                # Debug: Print the batch ids before concatenation
                print(f'Batch {batch_idx} ids: {batch["ids"]}')
                
                arr_batch = np.concatenate([x for x in batch['ids']])
                
                # Debug: Print the concatenated batch ids
                print(f'Concatenated Batch {batch_idx} ids: {arr_batch}')
                
                # Check if arr_batch contains any zeros
                if np.any(arr_batch == 0):
                    print(f'Zero found in concatenated batch {batch_idx}')

                # Write into mmap
                arr[idx: idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
                
                # Debug: Check the mmap content after writing the batch
                print(f'Memmap content at index {idx - len(arr_batch)} to {idx}: {arr[idx - len(arr_batch): idx]}')

            arr.flush()

    # Save the tokenized validation dataset for debugging
    save_tokenized_dataset({'val': tokenized_val_dataset}, 'new_tokenizer')  # Using new tokenizer
