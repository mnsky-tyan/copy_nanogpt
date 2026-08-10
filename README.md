# copy_nano

A learning-focused project inspired by and initially copied from [NanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy.

This repo is **for my own learning purposes**. I added comments and made incremental changes while experimenting with ideas and techniques beyond the original GPT‑2-style baseline.

## Credits / Inspiration

- Original inspiration and starting point: **NanoGPT** by Andrej Karpathy  

I’m grateful to Karpathy for making NanoGPT available and easy to learn from.

## What this repo is

- A personal fork and study implementation based on NanoGPT
- Extra comments and small refactors to improve readability
- Experimental improvements (newer techniques compared to GPT‑2 baseline), as I learn
- Currently study on model.py is finished, and train.py is much difficult that i do not very familiar with

## How to run

- Run `python data/prepareeos.py` to fetch/prepare the data. This script writes `input.txt` and generates `data/train.bin` and `data/val.bin` from it. (`input_eos.txt` is for reference only and is not used by the code.)
- Run `python train.py` to train the model using `data/train.bin` and `data/val.bin`.
- After training, the checkpoint is saved to `saved_model/ckpt.pt`.
- Use `python saved_model/inspect_ckpt.py` to inspect the saved checkpoint.
- Run `python chat.py` for interactive chatbot mode, or `python sample.py` to generate a batch of samples from a fixed model input.
- `train.py`, `chat.py`, and `sample.py` all accept CLI flags (e.g. `python train.py --batch_size=4`).
- `train.py`, `chat.py`, and `sample.py` also accept a Python config file (e.g. `python train.py my_config.py`). Variables defined in the file override the default arguments. Note: the config file is executed as Python code—only use trusted files, and keep it to argument/variable definitions.