"""
prepare_eos.py

Downloads Tiny Shakespeare (input.txt) if missing, tokenizes it with the GPT-2
tiktoken encoder, and inserts the GPT-2 end-of-text token (enc.eot_token) after
sentence-ending punctuation (. ! ?). Writes:
- train.bin / val.bin: uint16 token-id streams split 90/10 for training
- input_eos.txt: a human-readable copy with <|endoftext|> markers (inspection only)
"""

# NOTE:
# - encode(allowed_special= ) will read the allowed_special token as a single token from raw text
# - eos/bos/pad tokens in tokenizer are special
# - if encode_ordinary(), eos token will definitely not be intepreted as 50256 during encoding, but inferencing 50256 is allowed
# - encode(allowed_special= ) exists for eos token but not always used, main use cases including <|assistant|>, <|system|>, <|user|>, etc.

import os
import re
import requests
import numpy as np
import tiktoken


def download_if_missing(path):
    if os.path.exists(path):
        return
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(requests.get(url, timeout=60).text)


def main():
    dir_path = os.path.dirname(__file__)

    input_txt = os.path.join(dir_path, "input.txt")
    input_eos_txt = os.path.join(dir_path, "input_eos.txt")
    train_bin = os.path.join(dir_path, "train.bin")
    val_bin = os.path.join(dir_path, "val.bin")

    download_if_missing(input_txt)

    with open(input_txt, "r", encoding="utf-8") as f:
        data = f.read()

    enc = tiktoken.get_encoding("gpt2")
    eos_id = enc.eot_token

    parts = re.split(r"([.!?])", data)

    ids: list[int] = []
    buf = ""

    for part in parts:
        if part is None or part == "":
            continue
        buf += part
        if part in ".!?":
            ids.extend(enc.encode_ordinary(buf))
            ids.append(eos_id) 
            buf = ""

    if buf:
        ids.extend(enc.encode_ordinary(buf))

    EOS_MARKER = "<|endoftext|>" 

    data_with_marker = re.sub(r"([.!?])", r"\1" + EOS_MARKER, data)
    with open(input_eos_txt, "w", encoding="utf-8") as f:
        f.write(data_with_marker)

    split_idx = int(0.9 * len(ids))
    train_ids = np.array(ids[:split_idx], dtype=np.uint16)
    val_ids = np.array(ids[split_idx:], dtype=np.uint16)

    train_ids.tofile(train_bin)
    val_ids.tofile(val_bin)

    print(f"input.txt: {input_txt}")
    print(f"input_eos.txt (visual marker only): {input_eos_txt}")
    print(f"EOS token id: {eos_id}")
    print(f"total tokens (with EOS): {len(ids):,}")
    print(f"train.bin tokens: {len(train_ids):,}")
    print(f"val.bin tokens:   {len(val_ids):,}")
    print(f"wrote: {train_bin}")
    print(f"wrote: {val_bin}")

if __name__ == "__main__":
    main()