"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False
NOTE: compile=False for easier debugging and starts instantly 

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 16 gpus across 2 nodes, (8 gpus per node):
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py

NOTE: rarely needed, try it only when backend='nccl'
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1, before torchrun)
"""

# torchrun is the launcher of pytorch for distributed training 
# nproc_per_node is the number of processes on this node, most likely 1 gpu = 1 process
# standalone = only use one node(machine)
# node_rank: from 0 to nnodes-1 (which node is this)
# nnodes = number of total nodes participating
# WORLD_SIZE: number of all processes(gpus)
# LOCAL_RANK: from 0 to nproc_per_node-1 (on this node)
# master_addr = ip address of the master node 
# master_port = tcp port  
# NCCL is Nvidia Collective Communications Library for gpu
# torchrun sets RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT as environment variables automatically

import os
import time 
import math
import pickle

import torch
import numpy as np
from contextlib import nullcontext

# for distributed training
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from model_copy import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O 
out_dir = 'out'  # the directory of the model saved
eval_interval = 50
log_interval = 1
eval_iters = 10 # how many batches to average over in estimate_loss
eval_only = False
always_save_checkpoint = True
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*' (* means start with gpt2)

# wandb logging
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'gpt2' 

# data
dataset = 'data'
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 1    # if gradient_accumulation_steps > 1, this is the micro-batch size(number of sequences per gpu)
block_size = 128

# model
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.0    # e.g. 0.0 for pretraining(data is diverse), 0.1 for finetuning(prevent overfitting)
bias = False

# adamw optimizer
learning_rate = 6e-4
max_iters = 200
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 50
lr_decay_iters = 200
min_lr = 6e-5

# DDP settings
# communication backend, when more than one node is involved 
backend = 'gloo'  # 'gloo' for general cpu or gpu,  'nccl' for nvda gpu

# system
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1'
dtype = 'float32'
compile = False # use PyTorch 2.0 to compile the model
# -----------------------------------------------------------------------------
# isinstance check for v
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator_copy.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# torchrun runs the following code on every gpu independently

ddp = 'RANK' in os.environ     # is this a ddp run? 
if ddp:    
    # init variables
    ddp_rank = int(os.environ['RANK'])  # os.environ stores value as string
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed

    # work is divided equally by gradient_accumulation_stpes, not batch_size
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size

    # init device
    torch.cuda.set_device(device)    # make sure the process use the correct "device: idx"
    init_process_group(backend=backend) 
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
# one iter is a single weight update
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
# automatic mixed precision for forward pass, ptdtype is the lowest precision preferred when safe to use
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
data_dir = dataset
# useful only when using fp32, good habit to include either ways
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':

        # np.memmap is open() for arrays with better efficiency
        # uint16: 0 to 65535    # int16: -32768 to 32767
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')

    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    # ix is the token position to start with, batch_size is the return shape 
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # x is the actual sequence y is the target sequence
    # data[i] -> astype -> from_numpy -> repeat for ix times to get a tensor -> stack 
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    if device_type == 'cuda':

        # pin_memory is a specific cpu to nvda gpu technique, usually with non_blocking=True
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these before init_from, can override if init_from=='resume' (i.e. from a checkpoint)
iter_num = 0 
best_val_loss = 1e9     # lowest validation loss achieved
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias,  vocab_size=None, dropout=dropout)    

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    # attempt to derive vocab_size from the dataset
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # .pt is pytorch binary file extension
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)

    # ckpt.pt is a dictionary including key: model_args, model
    checkpoint_model_args = checkpoint['model_args']

    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:

        # k is the key, make the value (dict[k]) same as the ckpt value
        model_args[k] = checkpoint_model_args[k]

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']

    # torch.compile(model) might add prefix _orig_mod. 
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):

            # old_key = k                           
            # old_value = state_dict.pop(old_key)   
            # prefix_length = len(unwanted_prefix)  
            # new_key = old_key[prefix_length:]     
            # state_dict[new_key] = old_value 
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

# only the dropout can be overriden, this is the fine tuning of pretrained gpt2 
elif init_from.startswith('gpt2'):
    pass

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size


model.to(device)
# initialize a GradScalar. If enabled=False scaler is a no-op
# no-op function means it does nothing when called
scaler = torch.amp.GradScaler("cuda", enabled=(device_type == "cuda" and dtype == "float16"))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
# complete overwriting the optimizer if resume
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory 

if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model  # useful for debugging only 
    # model assigned back to model immediately, so unoptimized model is not compiled
    model = torch.compile(model)

# wrap model into DDP container
# always after compiling the model
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrary accurate loss over either split using many batches
# do forward pass and compute loss, with no backward pass
@torch.no_grad()
def estimate_loss():
    out = dict()

    # sets self.training to False, we customize the model behaviour when self.training == False
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)

            # ctx(amp), always before forward pass model()
            with ctx:
                logits, loss = model(X, Y)
            # loss is a tensor, .item() converts that to a float
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out 
            
# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    if it < warmup_iters:
        # the learning rate will increase when it increases, eventually very close to the set learning rate
        return learning_rate * (it + 1) / (warmup_iters + 1)
    
    # ~= max_iters per Chinchilla, thus usually not hitting this block
    if it > lr_decay_iters:
        return min_lr
    
    # apply cosine decay 
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi *  decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)
    
# logging
if wandb_log and master_process:
    import wandb  # weights and bias
    # config is whatever settings wandb record as run configuration
    wandb.init(project=wandb_project, name=wandb_run_name,config=config)

# training loop 
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0 
raw_model = model.module if ddp else model
running_mfu = -1.0

while iter_num <= max_iters:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    # optimizer.param_groups is the attribute of optimizer which conatins the dictionary of all attributes
    for param_group in optimizer.param_groups:
        # the original lr set to the new lr
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                'iter': iter_num,
                'train/loss': losses['train'],
                'val/loss': losses['val'],
                'lr': lr,
                'mfu': running_mfu*100,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    # call estimate_loss() then break 
    if iter_num == 0 and eval_only:
        break

    # calculate the accumulated gradient 
    for micro_step in range(gradient_accumulation_steps):
        sync_context = nullcontext()
        if ddp and micro_step != gradient_accumulation_steps-1:
            # no gradient sync for all gpus, which only happens when using loss.backward()
            sync_context = model.no_sync()
        with sync_context:
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps
            X, Y = get_batch('train')
        
            # loss.backward(), calculate the gradient
            # here we scaled the loss for applying amp
            scaler.scale(loss).backward()

    if grad_clip != 0.0:

        # scaler has scaled the loss for amp, unscale the gradient
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # optimizer.step()
    scaler.step(optimizer)

    # update the scale that is used to scale up loss
    scaler.update()

    # set to None, instead of default zeroing out
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:

        # loss.item() is float
        lossf = loss.item() * gradient_accumulation_steps

        # local_iter_num in case the model is resumed
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            # exponential moving average(EMA) of mfu
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

if ddp:
    destroy_process_group()   