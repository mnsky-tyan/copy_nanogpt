"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model_copy import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out'
start = '\n'
num_samples = 10
eos_id = tiktoken.get_encoding('gpt2').eot_token # None for always producing max_tokens
max_new_tokens = 100
temperature = 0.8
top_k = 10
seed = 1337
device = 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'
compile = False
exec(open('configurator_copy.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

elif init_from.startswith('gpt2'):
    pass



model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding('gpt2')
    encode = lambda s: enc.encode(s)
    decode = lambda l: enc.decode(l)

if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()

# batch initialization
start_ids = encode(start)
x1 = torch.tensor(start_ids, dtype=torch.long, device=device)[None, :]  # (1, T)

B = 4  # batch size you want
x = x1.repeat(B, 1)  # (B, T)

with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y, lengths = model.generate(
                x,
                max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=eos_id,
                return_eos=False,        # default, but explicit
                return_lengths=True,     # default, but explicit
                pad_token_id=eos_id      # optional; default is eos_id anyway if eos_token_id is set
            )
            print(f"=== sample {k+1} ===")
            for b in range(y.size(0)):
                seq = y[b, :lengths[b].item()].tolist()
                print(decode(seq))
                print('---------------')