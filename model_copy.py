import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn

# torch.nn.functional are the lowest level of python code, lower level codes are cuda and c++
from torch.nn import functional as F


# normalization, keep the scale , for stable and faster training 
# without: exploding gradients, activations, gradient descent becomes hard
# activations: all the intermediate state of the input to output process
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False. Use nn.LayerNorm for bias"""

    def __init__(self, config):
        
        # super init for self.parameters()/ model.to() eval() train()/ state_dict() load_state_dict()
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.n_embd))
        self.bias = nn.Parameter(torch.zeros(config.n_embd)) if config.bias else None
    
    # calling model like model(x) or self(x) will by default use the forward method
    def forward(self, x):
        
        # x = B*T*C
        # x_hat = (x[i] - mean)/(sqrt(variance + epsilon)) for each single number in c    # standardization 
        # output = weight * x_hat + bias        # weight and x_hat and bias are shpae (c, )
        # weight * x_hat is elementwise, C = (weight_0*x_hat_0, weight_1*x_hat_1, ...) for all B and T
        # 1e-5 is the epsilon for stablizing the calculation
        # self.weight.shape is the dimension to apply normalization, accept single number or shape     
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 

        # nn.Linear(input_dimension, ouptut_dimension, bias)
        # nn.Linear weight always 2d, bias always 1d
        # x @ w(transposed) + b
        # x = n_embd
        # weight.shape = (output_dim, input_dim)
        # n_embd(c_in) @ (output_dim, input_dim).t() = shape(output_dim, )    # .transpose(...) for general usage, .t() for 2d matrix only
        # broadcast c to b, t      
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias = config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias = config.bias)

        # dropout only affects the activations, zeroing out the activations but not the parameters 
        self.resid_dropout = nn.Dropout(config.dropout)

        # add attributes from config to reuse it in class methods
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.shape

        # dim = the dimension of the split on matrix
        # after passing self.c_attn the shape is (b, t, 3 * c) 
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)  
        
        # (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) 
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)

        # NOTE: 
        # the following attn_mask is equal to is_causal = True 
        # attn_mask = torch.ones(T, T)
        # attn_mask = torch.triu(attn_mask, diagonal=1)
        # attn_mask[attn_mask == 1] = -torch.inf
        # attn_mask[attn_mask == 0] = 0.0
        # output:
        # tensor([[  0., -inf, -inf, -inf],
        #         [  0.,   0., -inf, -inf],
        #         [  0.,   0.,   0., -inf],
        #         [  0.,   0.,   0.,   0.]])

        # NOTE: 
        # manual apply of SDPA FIXME
        # att = (q @ k.transpose(2, 3)) * (1.0 / math.sqrt(k.size(-1)))     output:(B, nh, T, T)
        # (T, hs) @ (hs, T) = (T, T) for each in every batch and every heads
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, -torch.inf)
        # apply mask manually (masked_fill not yet defined)
        # att = F.softmax(att, dim=-1)
        # T(row) * T(column) 
        # apply softmax on column: vectors in a row will sum to one 
        # att = self.attn_dropout(att)
        # apply dropout manually (self.attn_dropout not defined) 
        # y = att @ v 
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # (T, hs): hs is the actual activations for each T, with dim (n_embd//head_size)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        
        # use contiguous before view if it's after transpose() immediately, which is the only use case of contiguous() 
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        
        # do a linear before residual to learn the multi-head feature
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()

        # output shape by 4 for better performance of activation function 
        # fc = fully connected(dense)
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)

        # activation function(Gaussian Error Linear Unit)
        # adding non-linearity, fixed expression(not trainable)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config)
        self.mlp = MLP(config)

    # layernorm1 -> attn -> residual -> layernorm2 -> mlp -> residual 
    def forward(self, x):

        # adding x is called residual connection, where x is the residual
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:

    # name: type = value strucutre is required, it is also a way to define data type in python 
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for gpu efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        # can use self.config.x for other methods in the same class
        self.config = config

        # nn.Embedding is a module that input idx and output embedding without matmul, just simple mapping
        # wte word token embedding, wpe word position embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # last layernorm 
        self.ln_f = LayerNorm(config)

        # last linear layer
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # wte is at the start of the model (idx) to (activations)
        # lm_head is at the last of the model (activations) to (idx) 
        # gradient is calculated from both methods for each update of weight 
        self.wte.weight = self.lm_head.weight 

        # self defined _init_weights, use apply to do recursive initialization
        self.apply(self._init_weights)

        # for c_proj in causalselfattention and mlp, we override _init_weights and lower the std
        # lower std because they are immediately added to residual which might explode the std
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):

                # init with normal(Gaussian) distribution for simple and stable weights
                nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print(f"number of parameters: {self.get_num_params() / 1e6:.2f}M")



    def get_num_params(self, non_embeddings=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings(wte) would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
    
        n_params = sum(p.numel() for p in self.parameters())
        if non_embeddings:
            n_params -= self.wpe.weight.numel()
        return n_params


    #start with _ to remind user this function should not be called outside of the class
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):

            # module.weight/ module.bias = self.weight/self.bias for that module
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):

        # idx is a tensor which always have the attribute of .device
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # pos = position
        # create a 1d tensor from 0 to T-1 in order
        # this is the position idx
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.dropout(tok_emb + pos_emb)        
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)

        if targets is not None:

            # output shape(B, T, vocab_size)
            logits = self.lm_head(x)

            # format of F.cross_entropy(input_tensor, output_tensor, ignore_index)            
            # input_tensor shape: (B*T, vocab_size)
            # output_shape: (vocab_size)
            # ignore_index = the index to be ignored for target 
            # NOTE: both input and output should be real tensors

            # cross_entropy use the probability of the token to compute loss, not the embedding of the target    
            # cross_entropy is a loss function which use negative log-likelihood      

            # logits.size(-1) is the vocab_size, logits.view(-1) is automatic reshaping into(B*T, vocab_size)
            # targets.view(-1) is also shaping into (B*T)
            # comparing the current logits with the current idx for each token, for tokens in B*T
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:

            # use [-1] to preserve the 3d shape of logits for training, fetching the lass row(T)
            logits = self.lm_head(x[:,[-1],:])
            loss = None
        
        return logits, loss


    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size

        # overwriting the original parameter
        self.wpe.weight = nn.Parameter(self.wpe.weight[:block_size, :])
        # if manual attn, need to include attn.bias because attn_mask is coded with block_size
        # sdpa with is_casual=True does not depend on block_size

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        pass

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):

        # decay_params for weights but not bias 
        # weight decay is L2 regularization, discouraging large weight values
        decay_params = [p for pn, p in self.named_parameters() if p.requires_grad and p.dim() >= 2]
        non_decay_params = [p for pn, p in self.named_parameters() if p.requires_grad and p.dim() < 2]

        # can pass in groups for optimizer, or pass in list that contains dict for each group settings
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': non_decay_params, 'weight_decay': 0.0}
        ]
        
        num_decay = sum(p.numel() for p in decay_params)
        non_num_decay = sum(p.numel() for p in non_decay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay:,} parameters")
        print(f"num non-decayed parameter tensors: {len(non_decay_params)}, with {non_num_decay:,} parameters")

        # checking if 'fused' is in the arguments of that function/class
        # .parameters as the name of the arguments, or .key() is also ok
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'

        # use dict() for **kwargs
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f'using fused AdamW: {use_fused}')

        return optimizer


    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        N = self.get_num_params()
        cfg = self.config

        # PaLM paper
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt) 
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu


    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_token_id=None):

        # must hit range(max_new_tokens) every time 
        for _ in range(max_new_tokens):

            # cut to the block_size, seeing the most recent tokens only
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # self() is calling forward pass
            logits, _ = self(idx_cond)

            # get rid of the time dimension since its not training
            # higher temperature tends to cause logits to be closer, higher randomness of inference
            logits = logits[:, -1, :] / temperature

            if top_k is not None:

                # min() to find the smallest between top_k and vocab_size, error handling
                # output of torch.topk: (v, i) v is value of of the highest to lowest logits, i is the index
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))

                # v[:, [-1]] checks for the smallest value within top_k
                # logits[True] (boolena tensor, only in torch)
                # and logits that are smaller than v[:, [-1]] are set to -inf
                logits[logits < v[:, [-1]]] = -torch.inf

            probs = F.softmax(logits, dim=-1)

            # torch.multinomial is the sampling step 
            idx_next = torch.multinomial(probs, num_samples=1)

            # dim=1 is the T dimension
            idx = torch.cat((idx, idx_next), dim=1)

            if eos_token_id is not None and idx_next == eos_token_id:
                break

        if eos_token_id is not None:
            eos_positions = (idx == eos_token_id).nonzero(as_tuple=False)
            if len(eos_positions) > 0:
                first_eos_idx = eos_positions[0, 1]
                idx = idx[:, :first_eos_idx]

        return idx

# if __name__ == "__main__":
#     import torch

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     config = GPTConfig(
#         block_size=128,
#         vocab_size=100,
#         n_layer=2,
#         n_head=2,
#         n_embd=64,
#         dropout=0.0,
#         bias=True
#     )

#     model = GPT(config).to(device)
#     model.eval()

#     idx = torch.zeros((1, 1), dtype=torch.long, device=device)

#     with torch.no_grad():
#         out = model.generate(idx, max_new_tokens=10)

#     print("Sanity check passed.")
#     print("Output shape:", out.shape)
