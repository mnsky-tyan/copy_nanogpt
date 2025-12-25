import os
import requests
import tiktoken
import numpy as np

# dirname(a) finds the path of the directory of a
# __file__ refer to the current script file
dir_path = os.path.dirname(__file__)
input_file_path = os.path.join(dir_path, 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    # with open 'x' creates file only, if exist raise error, optional safety
    # use encoding='utf-8' almost always
    with open(input_file_path, 'x', encoding='utf-8') as f:
        # request
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
train_data = data[:int(len(data)*0.9)]
val_data = data[int(len(data)*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

# .tofile() overwrites whatever exist
train_ids.tofile(os.path.join(dir_path, 'train.bin'))
val_ids.tofile(os.path.join(dir_path, 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
