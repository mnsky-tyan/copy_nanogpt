import os
import tiktoken
import numpy as np
import re

# --- Paths ---
dir_path = os.path.dirname(__file__)
input_file = os.path.join(dir_path, 'input.txt')
input_eos_file = os.path.join(dir_path, 'input_eos.txt')
train_file = os.path.join(dir_path, 'train.bin')
val_file = os.path.join(dir_path, 'val.bin')

# --- Read original data ---
with open(input_file, 'r', encoding='utf-8') as f:
    data = f.read()

# --- Setup tokenizer ---
enc = tiktoken.get_encoding("gpt2")
eos_char = enc.decode([enc.eot_token])  # decode EOS token to string

# --- Insert EOS after sentence-ending punctuation --- 
# Only after '.', '!', '?'
pattern = re.compile(r'([.!?])')
data_with_eos = pattern.sub(r'\1' + eos_char, data)

# --- Save modified text ---
with open(input_eos_file, 'w', encoding='utf-8') as f:
    f.write(data_with_eos)

# --- Encode tokens, allowing special tokens (so EOS is properly recognized) ---
all_ids = enc.encode(data_with_eos, allowed_special="all")  # important change

# --- Split into train/val ---
split_idx = int(0.9 * len(all_ids))
train_ids = np.array(all_ids[:split_idx], dtype=np.uint16)
val_ids = np.array(all_ids[split_idx:], dtype=np.uint16)

# --- Save .bin files ---
train_ids.tofile(train_file)
val_ids.tofile(val_file)

print(f"input_eos.txt saved ({len(all_ids)} tokens total)")
print(f"train.bin: {len(train_ids)} tokens")
print(f"val.bin: {len(val_ids)} tokens")
