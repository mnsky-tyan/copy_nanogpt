"""
prepare_eos.py

- Downloads tinyshakespeare input.txt (if missing)
- Writes input_eos.txt (human-readable) with a visible EOS marker string AFTER sentence punctuation
- Produces train.bin / val.bin from token ids where EOS is appended at the ID level
  (no allowed_special needed)

This avoids inserting "<|endoftext|>" into the text before encoding.
"""

import os
import re
import requests
import numpy as np
import tiktoken


def download_if_missing(path: str) -> None:
    if os.path.exists(path):
        return
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(requests.get(url, timeout=60).text)


def main() -> None:
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

    # ---------
    # 1) Build token ids WITHOUT allowed_special
    #    We split the raw text and append eos_id after sentence-ending punctuation.
    # ---------
    # This keeps punctuation tokens in the stream and inserts EOS right after them.
    # re.split keeps delimiters if we capture them.
    parts = re.split(r"([.!?])", data)

    ids: list[int] = []
    buf = ""

    for part in parts:
        if part is None or part == "":
            continue
        buf += part
        if part in ".!?":
            # encode text up to and including the punctuation
            ids.extend(enc.encode_ordinary(buf))
            ids.append(eos_id)  # <-- EOS inserted at token-id level
            buf = ""

    # leftover (if text doesn't end with punctuation)
    if buf:
        ids.extend(enc.encode_ordinary(buf))

    # ---------
    # 2) Write a human-readable input_eos.txt
    #    This is ONLY for inspection. It is NOT used for encoding.
    # ---------
    # Use a marker that is easy to see and unlikely to appear in the dataset.
    EOS_MARKER = "<|eos|>"
    # Reconstruct text from tokens, then insert marker after punctuation in plain text.
    # (This is just for readability; training uses `ids` above.)
    data_with_marker = re.sub(r"([.!?])", r"\1" + EOS_MARKER, data)
    with open(input_eos_txt, "w", encoding="utf-8") as f:
        f.write(data_with_marker)

    # ---------
    # 3) Split ids into train/val and write .bin
    # ---------
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