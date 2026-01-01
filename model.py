import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F  # lower level codes are c++ cuda


# NOTE: LayerNorm is unused in this model
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = B*T*C
        # x_hat = (x[i] - mean)/(sqrt(variance + epsilon)) for each single number in c    # standardization 
        # output = weight * x_hat + bias        # weight and x_hat and bias are shpae (c, )
        # weight * x_hat is elementwise, C = (weight_0*x_hat_0, weight_1*x_hat_1, ...) for all B and T
        # 1e-5 is the epsilon for stablizing the calculation
        # self.weight.shape is the dimension to apply normalization, accept single number or shape     
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


# manually apply Root Mean Square Normalization, the better version is fused RMSNorm with module import on GPU
class RMSNorm(nn.Module):
    """RMSNorm with optional bias (bias is typically False in modern LLMs)."""

    def __init__(self, config, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(config.n_embd))
        self.bias = nn.Parameter(torch.zeros(config.n_embd)) if config.bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        # rms = sqrt(mean(x^2) + eps) over the last dimension C
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(mean_square + self.eps)
        x_norm = x * inv_rms

        y = x_norm * self.weight  # broadcast over (B, T)
        if self.bias is not None:
            y = y + self.bias
        return y


# Causal Self Attention with Rotary Position Embedding (RoPE)
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
        q_dim = config.n_embd
        kv_dim = config.n_kv_head * (config.n_embd // config.n_head)
        self.c_attn = nn.Linear(config.n_embd, q_dim + 2 * kv_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # dropout only affects the activations, zeroing out the activations but not the parameters)
        self.attn_dropout = float(getattr(config, "attn_dropout", config.dropout))
        self.resid_dropout_p = float(getattr(config, "resid_dropout", config.dropout))
        self.resid_dropout = nn.Dropout(self.resid_dropout_p)

        # add attributes from config to reuse it in class methods
        self.block_size = config.block_size
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_kv_head = config.n_kv_head
        self.n_rep = self.n_head // self.n_kv_head  # how many query heads share one kv head/ config.n_head
        self.qk_rmsnorm = config.qk_rmsnorm
        self.qk_rmsnorm_eps = float(getattr(config, "qk_rmsnorm_eps", 1e-6))
        self.head_dim = config.n_embd // config.n_head
        assert self.head_dim % 2 == 0, "RoPE requires even head_dim"
        assert self.n_head % self.n_kv_head == 0

        self.rope_base = config.rope_base
        self.rope_scale_factor = float(getattr(config, "rope_scale_factor", 1.0))
        if self.rope_scale_factor <= 0:
            raise ValueError("rope_scale_factor must be > 0")

        # register buffer is used to store tensors that are not trainable
        # persistent=False means the buffer will not be saved in the checkpoint
        #
        # NOTE:
        # We keep a bounded RoPE "window" cache to avoid unbounded growth when using
        # KV-cache + sliding window attention. This cache stores cos/sin for a fixed
        # contiguous span of absolute positions.
        self.rope_window_size = config.block_size  # or separate config
        self.register_buffer("rope_cos_window", None, persistent=False)  # (1,1,W,hd)
        self.register_buffer("rope_sin_window", None, persistent=False)  # (1,1,W,hd)
        self.rope_window_start_pos = 0  # python int, not a buffer

        self.register_buffer("rope_cos_cached", None, persistent=False)
        self.register_buffer("rope_sin_cached", None, persistent=False)
        self.rope_cache_len = 0

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_kv_head, T, head_dim)
        return: (B, n_head, T, head_dim) by repeating each kv head n_rep times
        """
        if self.n_rep == 1:
            return x
        return x.repeat_interleave(self.n_rep, dim=1).contiguous()
    
    @staticmethod
    def _rms_norm_lastdim(x: torch.Tensor, eps: float) -> torch.Tensor:
        # x: (..., head_dim)
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

    def _gqa_sdpa_no_repeat(
        self,
        q: torch.Tensor,  # (B, n_head, T, hd)
        k: torch.Tensor,  # (B, n_kv_head, S, hd)
        v: torch.Tensor,  # (B, n_kv_head, S, hd)
        attn_mask: torch.Tensor | None,
        dropout_p: float,
        is_causal: bool,
    ) -> torch.Tensor:
        B, nh, T, hd = q.shape
        _, nk, S, _ = k.shape
        assert nh == self.n_head and nk == self.n_kv_head
        assert nh % nk == 0
        n_rep = nh // nk

        # (B, nk, n_rep, T, hd)
        qg = q.view(B, nk, n_rep, T, hd)

        outs = []
        for g in range(nk):
            # q: (B, n_rep, T, hd)
            q_g = qg[:, g, :, :, :]
            # k,v: (B, 1, S, hd) shared for all reps in this group
            k_g = k[:, g:g+1, :, :].expand(B, n_rep, S, hd)
            v_g = v[:, g:g+1, :, :].expand(B, n_rep, S, hd)

            # Broadcast k_g/v_g across heads automatically (1 -> n_rep)
            out_g = F.scaled_dot_product_attention(
                q_g, k_g, v_g,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )  # (B, n_rep, T, hd)
            outs.append(out_g)

        # concat group outputs back to (B, n_head, T, hd)
        y = torch.cat(outs, dim=1)
        return y

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        # x is a tensor of shape (B, nh, T, hs)
        # rotation is done on the last dimension (hs)

        # x1 = x[..., ::2] means x1 = x[0], x[2], x[4], ...
        # x2 = x[..., 1::2] means x2 = x[1], x[3], x[5], ...
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        # input = [10, 11, 20, 21, 30, 31]
        # output = [-11, 10, -21, 20, -31, 30]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    @staticmethod
    def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary embedding to x.
        x: (B, nh, T, hs)
        cos/sin: (1, 1, T, hs)
        """

        # standard rotation for (a, b)
        # the return of _apply_rope is same as (a, b) = (acosθ − bsinθ, asinθ + bcosθ)
        return (x * cos) + (CausalSelfAttention._rotate_half(x) * sin)

    def _rope_cos_sin_from_positions(
        self,
        positions: torch.Tensor,   # (T,) int/long positions
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build RoPE cos/sin for given absolute positions.
        Returns cos/sin of shape (1, 1, T, head_dim) in `dtype` on `device`.
        """
        if self.head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim, got {self.head_dim}")

        pos = positions.to(device=device, dtype=torch.float32)  # (T,)
        if self.rope_scale_factor != 1.0:
            pos = pos / self.rope_scale_factor

        inv_freq = 1.0 / (
            self.rope_base ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim)
        )  # (head_dim/2,)

        freqs = torch.outer(pos, inv_freq)                      # (T, head_dim/2)
        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1) # (T, head_dim)

        cos = emb.cos()[None, None, :, :].to(dtype=dtype)       # (1,1,T,head_dim)
        sin = emb.sin()[None, None, :, :].to(dtype=dtype)
        return cos, sin

    def _causal_mask_with_past(self, q_len: int, k_len: int, past_len: int, device: torch.device) -> torch.Tensor:
        """
        Returns a boolean mask of shape (q_len, k_len) where True means "masked out".
        Allows query position i to attend to keys <= past_len + i.
        """
        # key positions: 0..k_len-1
        key_pos = torch.arange(k_len, device=device)  # (k_len,)
        # query positions in the *full* sequence: past_len..past_len+q_len-1
        query_pos = past_len + torch.arange(q_len, device=device)  # (q_len,)

        # broadcast: (q_len, k_len)
        # mask out keys with position > query_pos
        mask = key_pos[None, :] > query_pos[:, None]
        return mask

    def _get_rope_cos_sin(self, T: int, device: torch.device, dtype: torch.dtype):
        need_rebuild = (
            self.rope_cos_cached is None
            or self.rope_sin_cached is None
            or self.rope_cache_len < T
            or self.rope_cos_cached.device != device
            or self.rope_sin_cached.device != device
            or self.rope_cos_cached.dtype != dtype
            or self.rope_sin_cached.dtype != dtype
        )
        if need_rebuild:
            positions = torch.arange(T, device=device, dtype=torch.long)
            cos, sin = self._rope_cos_sin_from_positions(positions, device=device, dtype=dtype)
            self.rope_cos_cached = cos
            self.rope_sin_cached = sin
            self.rope_cache_len = T

        return self.rope_cos_cached, self.rope_sin_cached

    def _build_rope_window(
        self,
        start_pos: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        W = self.rope_window_size
        positions = torch.arange(start_pos, start_pos + W, device=device, dtype=torch.long)
        cos, sin = self._rope_cos_sin_from_positions(positions, device=device, dtype=dtype)
        self.rope_cos_window = cos
        self.rope_sin_window = sin
        self.rope_window_start_pos = start_pos

    def _get_rope_cos_sin_bounded(
        self,
        positions: torch.Tensor,  # (T,)
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        assert positions.dim() == 1 and positions.numel() > 0

        min_pos = int(positions.min().item())
        max_pos = int(positions.max().item())

        start = self.rope_window_start_pos
        end = start + self.rope_window_size - 1

        need_rebuild = (
            self.rope_cos_window is None
            or self.rope_sin_window is None
            or self.rope_cos_window.device != device
            or self.rope_sin_window.device != device
            or self.rope_cos_window.dtype != dtype
            or self.rope_sin_window.dtype != dtype
            or min_pos < start
            or max_pos > end
        )

        if need_rebuild:
            # Center window around max_pos. For decode T=1 this is the token position.
            new_start = max(0, max_pos - (self.rope_window_size - 1))
            self._build_rope_window(new_start, device, dtype)
            start = self.rope_window_start_pos

        # map absolute positions -> window indices
        idx = (positions - start).to(torch.long)  # (T,)
        cos = self.rope_cos_window.index_select(2, idx)  # (1,1,T,hd)
        sin = self.rope_sin_window.index_select(2, idx)
        return cos, sin

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        pos_offset: int = 0,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], int]:
        B, T, C = x.shape

        head_dim = self.head_dim
        q_dim = self.n_embd
        kv_dim = self.n_kv_head * head_dim

        qkv = self.c_attn(x) 
        q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=2)
        
        # (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        k = k.view(B, T, self.n_kv_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, head_dim).transpose(1, 2)

        # total length = cached + current
        past_len = 0 if past_kv is None else past_kv[0].size(2)

        # build (or rebuild) cache for this sequence length/device/dtype
        # cache shapes: (1, 1, T, head_dim)
        if not use_cache:
            # training/full forward: positions are [0..T-1]
            cos, sin = self._get_rope_cos_sin(T, x.device, q.dtype)
            assert cos.device == x.device and sin.device == x.device
            assert cos.dtype == q.dtype and sin.dtype == q.dtype
            assert cos.size(-1) == self.head_dim and sin.size(-1) == self.head_dim
            q = self._apply_rope(q, cos, sin)
            k = self._apply_rope(k, cos, sin)
        else:
            # decoding/KV cache: absolute positions with bounded window
            positions = torch.arange(
                pos_offset + past_len,
                pos_offset + past_len + T,
                device=x.device,
                dtype=torch.long,
            )
            cos, sin = self._get_rope_cos_sin_bounded(positions, x.device, q.dtype)
            q = self._apply_rope(q, cos, sin)
            k = self._apply_rope(k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)  # concat on T dimension
            v = torch.cat([past_kv[1], v], dim=2)

        if k.size(2) > self.block_size:
            drop = k.size(2) - self.block_size
            k = k[:, :, drop:, :].contiguous()
            v = v[:, :, drop:, :].contiguous()
            pos_offset += drop

        present = (k, v)
        past_len_eff = k.size(2) - T
        dropout_p = self.attn_dropout if self.training else 0.0

        if self.qk_rmsnorm:
            q = self._rms_norm_lastdim(q, self.qk_rmsnorm_eps)
            k = self._rms_norm_lastdim(k, self.qk_rmsnorm_eps)

        k_len = k.size(2)   # S
        q_len = q.size(2)   # T

        attn_mask = None
        is_causal = True

        # When decoding with cache, K is longer than Q; be explicit and safe
        if use_cache and past_len_eff > 0:
            attn_mask = self._causal_mask_with_past(
                q_len=q_len,
                k_len=k_len,
                past_len=past_len_eff,
                device=x.device,
            )
            is_causal = False

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
        # manual apply of SDPA
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
        y = self._gqa_sdpa_no_repeat(
            q=q,
            k=k,
            v=v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

        # use contiguous before view if it's after transpose() immediately, which is the only use case of contiguous()
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # do a linear before residual to learn the multi-head feature
        y = self.resid_dropout(self.c_proj(y))

        return y, present, pos_offset


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Common modern choice is hidden_dim ~= int(8/3 * n_embd) to keep compute similar to 4x GELU MLP
        hidden_dim = int((8 * config.n_embd) / 3)
        # make it divisible by 64 for GPU friendliness 
        hidden_dim = (hidden_dim + 63) // 64 * 64
        self.hidden_dim = hidden_dim

        # fc = fully connected(dense)
        self.c_fc = nn.Linear(config.n_embd, 2 * hidden_dim, bias=config.bias)

        self.silu = nn.SiLU()
 
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(getattr(config, "resid_dropout", config.dropout))
        
    def forward(self, x):
        x = self.c_fc(x)

        # SwiGlu, Swish Gated Linear Unit
        a, b = x.split(self.hidden_dim, dim=-1)
        x = a * self.silu(b)

        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = RMSNorm(config) # originally layernorm
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config)
        self.mlp = MLP(config)

    # layernorm1 -> attn -> residual -> layernorm2 -> mlp -> residual 
    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        pos_offset: int = 0,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], int]:

        attn_out, present, pos_offset = self.attn(self.ln_1(x), past_kv=past_kv, pos_offset=pos_offset, use_cache=use_cache)

        # adding x is called residual connection, where x is the residual
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present, pos_offset


@dataclass
class GPTConfig:
    # name: type = value strucutre is required, it is also a way to define data type in python 
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for gpu efficiency
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 3
    n_embd: int = 768
    dropout: float = 0.0
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    emb_dropout: float = 0.0
    bias: bool = False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    rope_base: float = 10000.0
    rope_scale_factor: float = 1.0
    qk_rmsnorm: bool = False
    qk_rmsnorm_eps: float = 1e-6
    

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.vocab_size is not None
        assert config.block_size is not None

        # can use self.config.x for other methods in the same class
        self.config = config

        # nn.Embedding is a module that input idx and output embedding without matmul, just simple mapping
        # wte word token embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        emb_p = getattr(config, "emb_dropout", config.dropout)
        self.dropout = nn.Dropout(emb_p)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # last layernorm 
        self.ln_f = RMSNorm(config) # originally layernorm

        # last linear layer
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # wte is at the start of the model (idx) to (activations)
        # lm_head is at the last of the model (activations) to (idx) 
        # gradient is calculated from both methods for each update of weight 
        self.lm_head.weight = self.wte.weight

        # self defined _init_weights, use apply to do recursive initialization
        self.apply(self._init_weights)

        # for c_proj in causalselfattention and mlp, we override _init_weights and lower the std
        # lower std because they are immediately added to residual which might explode the std
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):

                # init with normal(Gaussian) distribution for simple and stable weights
                nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print(f"number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
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

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        pos_offsets: list[int] | None = None,
        use_cache: bool = False,
    ):

        # idx is a tensor which always have the attribute of .device
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # idx must be integer token ids
        assert idx.dtype == torch.long, f"idx must be torch.long, got {idx.dtype}"
        assert idx.dim() == 2, f"idx must be (B,T), got shape {tuple(idx.shape)}"

        tok_emb = self.wte(idx)
        x = self.dropout(tok_emb)
     
        if use_cache:
            if past_kv is None:
                past_kv = [None] * len(self.h)
            if pos_offsets is None:
                pos_offsets = [0] * len(self.h)

            presents: list[tuple[torch.Tensor, torch.Tensor]] = []
            new_offsets: list[int] = []

            for i, (block, layer_past) in enumerate(zip(self.h, past_kv)):
                x, present, new_off = block(
                    x,
                    past_kv=layer_past,
                    pos_offset=pos_offsets[i],
                    use_cache=True,
                )
                presents.append(present)
                new_offsets.append(new_off)

        else:
            # training/full forward: no KV cache bookkeeping
            for block in self.h:
                x, _, _ = block(x, past_kv=None, pos_offset=0, use_cache=False)
            presents = None
            new_offsets = None

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

        if use_cache:
            return logits, loss, presents, new_offsets
        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        for block in self.h:
            block.attn.block_size = block_size
            block.attn.rope_window_size = block_size
        # if manual attn, need to include attn.bias because attn_mask is coded with block_size
        # sdpa with is_casual=True does not depend on block_size

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

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        min_new_tokens: int = 0,
        eos_token_id: int | None = None,
        return_eos: bool = False,
        pad_token_id: int | None = None,
        return_lengths: bool = True,
    ):

        assert idx.dim() == 2, f"idx must be (B,T), got shape {tuple(idx.shape)}"
        assert idx.dtype == torch.long, f"idx must be torch.long, got {idx.dtype}"
        
        B, T = idx.size()
        device = idx.device
        past_kv = None
        pos_offsets = None
        # finished is a boolean tensor of shape (B, )
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        # eos is a single number tensor, eos_token_id is an int
        # create eos for torch.where()
        eos = torch.tensor(eos_token_id, device=device, dtype=idx.dtype) if eos_token_id is not None else None

        for step in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:] if past_kv is None else idx[:, -1:]
            logits, _, past_kv, pos_offsets = self(idx_cond, past_kv=past_kv, pos_offsets=pos_offsets, use_cache=True)

            # get the last token
            logits = logits[:, -1, :]  # (B, vocab)

            if temperature < 0:
                raise ValueError("temperature must be >= 0")

            # Ban EOS before min_new_tokens
            if eos_token_id is not None and step < min_new_tokens:
                # fetch the index of eos_token_id
                logits[:, eos_token_id] = -torch.inf

            # Greedy sampling
            if temperature == 0.0:
                # argmax returns the index of the max value
                # dim=-1(column) means return the max value for each row(batch)
                # keepdim=True returns (B, 1) instead of (B,)
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            # Normal sampling
            else:
                logits = logits / temperature

                if top_k is not None and top_k > 0:
                    # in case top_k is larger than the vocabulary size
                    k = min(int(top_k), logits.size(-1))
                    # topk returns the k largest values and their indices
                    topk_vals, _ = torch.topk(logits, k)
                    # mask out the values that are smaller than the k largest values
                    # topk_vals[-1] is the smallest k, preserving the dimension
                    logits = logits.masked_fill(logits < topk_vals[:, [-1]], -torch.inf)

                # apply top_p after some logits are masked by top_k
                if top_p is not None:
                    p = float(top_p)
                    if not (0.0 < p <= 1.0):
                        raise ValueError("top_p must be in (0, 1].")

                    # sort returns the sorted values and their indices for all the batches
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    
                    # e.g. sorted_probs = [0.40, 0.25, 0.15], cumulative_probs = [0.40, 0.65, 0.80]
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    # e.g. sorted_mask = [False, False, ..., True, True, ...]
                    # allows cumsum exceed p once
                    sorted_mask = (cumulative_probs - sorted_probs) > p

                    # fallback, the index 0 token is always False
                    # sorted_mask[..., 0] is an ellipsis, same as sorted_mask[:, 0] for 2d and [:, :, 0] for 3d
                    sorted_mask[..., 0] = False

                    # zeros_like() returns a tensor of the same shape and dtype as sorted_mask
                    # .scatter(dim, index, src), index=place to write, src=value to write
                    mask = torch.zeros_like(sorted_mask).scatter(-1, sorted_indices, sorted_mask)
                    logits = logits.masked_fill(mask, -torch.inf)

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None:
                # if finished = True, idx_next = eos
                idx_next = torch.where(finished.view(B, 1), eos, idx_next)

                # idx_next is (B, 1), squeeze it becomes (B,)
                # if finished = False, check the condition until True, and |= keeps it True
                finished |= (idx_next.view(B, ) == eos_token_id)

            idx = torch.cat((idx, idx_next), dim=1)

            if eos_token_id is not None and finished.all().item():
                break

        lengths = None
        if eos_token_id is not None and not return_eos:
            trimmed = []
            lengths_list = []
            for b in range(B):
                # look at the whole batch
                seq = idx[b]
                # .nonzero() returns the indices of non-zero values 
                eos_pos = (seq == eos_token_id).nonzero(as_tuple=False)
                if eos_pos.numel() > 0:
                    cut = int(eos_pos[0].item())
                    seq = seq[:cut]
                    
                # [tensor([...]), tensor([...]), tensor([...]), ...]
                trimmed.append(seq)
                lengths_list.append(seq.numel())

            # torch.long = torch.int64
            lengths = torch.tensor(lengths_list, device=device, dtype=torch.long)
            max_len = int(lengths.max().item())

            if pad_token_id is None:
                pad_token_id = eos_token_id

            # .new_full() returns a tensor of the same dtype and device as idx
            out = idx.new_full((B, max_len), fill_value=int(pad_token_id))

            # write batch by batch to out
            for b, seq in enumerate(trimmed):
                out[b, :seq.numel()] = seq
            idx = out

        # returns a tensor length B with all values T, not meaningful
        else:
            if return_lengths:
                lengths = torch.full((B,), idx.size(1), device=device, dtype=torch.long)

        if return_lengths:
            return idx, lengths
        return idx