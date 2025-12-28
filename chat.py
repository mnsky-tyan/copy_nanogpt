# chat.py
# Terminal chat REPL for your GPT model checkpoint.
# Commands use backslash: \help, \exit, \clear, ...

import os
import pickle
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import tiktoken

from model import GPTConfig, GPT


# ----------------------------- Defaults -----------------------------

@dataclass
class ChatSettings:
    out_dir: str = "saved_model"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "float32"
    compile: bool = False

    # Generation controls
    max_new_tokens: int = 200
    min_new_tokens: int = 10
    temperature: float = 0.8
    top_k: int | None = 40
    top_p: float | None = 0.95

    # Chat formatting
    system_prompt: str = "You are a helpful assistant."
    show_prompt: bool = False          # \prompt toggles
    stop_on_tags: bool = True          # \tags toggles
    echo_prompt_tokens: bool = False   # debug


SETTINGS = ChatSettings()


# ----------------------------- Utilities -----------------------------

def _clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _print_help() -> None:
    print(
        r"""
Commands
  \help                 Show this help
  \exit                 Exit the chat
  \clear                Clear conversation history (keeps system prompt)
  \reset                Reset everything (history + system prompt back to default)

  \system <text>        Set the system prompt to <text>
  \system               Show current system prompt

  \seed <int>           Set RNG seed (affects sampling). Also reseeds CUDA if available.
  \params               Show current generation parameters
  \set <k> <v>          Set a generation parameter (see below)

  \prompt               Toggle printing the full formatted prompt (debug)
  \tags                 Toggle stopping when the model starts emitting role tags (debug)

Generation parameters for \set
  temperature <float>   e.g. \set temperature 0.7
  top_k <int|none>      e.g. \set top_k 50  or  \set top_k none
  top_p <float|none>    e.g. \set top_p 0.95 or \set top_p none
  max_new_tokens <int>  e.g. \set max_new_tokens 256
  min_new_tokens <int>  e.g. \set min_new_tokens 20

Notes
- This is not an instruction-tuned chat model; it's a next-token LM with a chat-like interface.
- EOS stopping follows sample.py behavior: we stop on eos_id and trim it out (so it won't print).
"""
    )


def _parse_none_int(s: str) -> int | None:
    s = s.strip().lower()
    if s in {"none", "null"}:
        return None
    return int(s)


def _parse_none_float(s: str) -> float | None:
    s = s.strip().lower()
    if s in {"none", "null"}:
        return None
    return float(s)


def _print_params(cfg: ChatSettings) -> None:
    print("Generation parameters:")
    print(f"  temperature    = {cfg.temperature}")
    print(f"  top_k          = {cfg.top_k}")
    print(f"  top_p          = {cfg.top_p}")
    print(f"  max_new_tokens = {cfg.max_new_tokens}")
    print(f"  min_new_tokens = {cfg.min_new_tokens}")
    print(f"Debug toggles:")
    print(f"  show_prompt    = {cfg.show_prompt}")
    print(f"  stop_on_tags   = {cfg.stop_on_tags}")


def _set_param(cfg: ChatSettings, key: str, value: str) -> None:
    key = key.strip()
    value = value.strip()

    if key == "temperature":
        v = float(value)
        if v < 0:
            raise ValueError("temperature must be >= 0")
        cfg.temperature = v

    elif key == "top_k":
        cfg.top_k = _parse_none_int(value)

    elif key == "top_p":
        cfg.top_p = _parse_none_float(value)
        if cfg.top_p is not None and not (0.0 < cfg.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1].")

    elif key == "max_new_tokens":
        v = int(value)
        if v <= 0:
            raise ValueError("max_new_tokens must be > 0")
        cfg.max_new_tokens = v

    elif key == "min_new_tokens":
        v = int(value)
        if v < 0:
            raise ValueError("min_new_tokens must be >= 0")
        cfg.min_new_tokens = v

    else:
        raise ValueError(f"Unknown parameter '{key}'")


# ----------------------------- Model + Tokenizer -----------------------------

def load_model(out_dir: str, device: str, compile_model: bool):
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)

    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    if compile_model:
        model = torch.compile(model)

    return model, checkpoint


def load_tokenizer(checkpoint):
    # Matches your sample.py logic
    load_meta = False
    if "config" in checkpoint and "dataset" in checkpoint["config"]:
        meta_path = os.path.join("data", checkpoint["config"]["dataset"], "meta.pkl")
        load_meta = os.path.exists(meta_path)

    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        stoi, itos = meta["stoi"], meta["itos"]
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda ids: "".join([itos[i] for i in ids])
        eos_id = None
        print("Using char-level encoding from meta.pkl (no EOS id from tiktoken).")
        return encode, decode, eos_id
    else:
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s)   # strict: will error if <|endoftext|> appears in the prompt text
        decode = lambda ids: enc.decode(ids)
        eos_id = enc.eot_token             # 50256
        return encode, decode, eos_id


def format_history(system_prompt: str, history: list[tuple[str, str]]) -> str:
    parts: list[str] = []
    if system_prompt:
        parts.append(f"<|system|>\n{system_prompt}\n")
    for role, text in history:
        if role == "user":
            parts.append(f"<|user|>\n{text}\n")
        else:
            parts.append(f"<|assistant|>\n{text}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


# ----------------------------- Generation -----------------------------

@torch.no_grad()
def generate_reply(model, encode, decode, prompt_text: str, eos_id: int | None, cfg: ChatSettings, ctx):
    # Encode prompt
    prompt_ids = encode(prompt_text)
    prompt_len = len(prompt_ids)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=cfg.device)[None, :]

    # Generate full sequence (prompt + continuation) and trim EOS like sample.py when eos_id is set
    y = None
    lengths = None

    with ctx:
        if eos_id is None:
            # No EOS id available; just generate max_new_tokens.
            y = model.generate(
                x,
                cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                min_new_tokens=cfg.min_new_tokens,
                eos_token_id=None,
                return_eos=True,
                return_lengths=False,
                pad_token_id=0,
            )
            if isinstance(y, tuple):
                y = y[0]
            out_ids = y[0].tolist()
        else:
            y, lengths = model.generate(
                x,
                cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                min_new_tokens=cfg.min_new_tokens,
                eos_token_id=eos_id,
                return_eos=False,          # IMPORTANT: trimmed, so EOS won't decode/print
                return_lengths=True,
                pad_token_id=eos_id,
            )
            out_len = int(lengths[0].item())
            out_ids = y[0, :out_len].tolist()

    # Slice out the assistant completion in token space
    new_ids = out_ids[prompt_len:]
    reply = decode(new_ids)

    # Optional: stop when the model starts a new role tag
    if cfg.stop_on_tags:
        for stop in ["<|user|>", "<|system|>", "<|assistant|>"]:
            j = reply.find(stop)
            if j != -1:
                reply = reply[:j]
                break

    return reply.strip()


# ----------------------------- REPL -----------------------------

def main():
    # Seeding
    seed = 1337
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # AMP context
    device_type = "cuda" if "cuda" in SETTINGS.device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[SETTINGS.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Load model + tokenizer
    model, checkpoint = load_model(SETTINGS.out_dir, SETTINGS.device, SETTINGS.compile)
    encode, decode, eos_id = load_tokenizer(checkpoint)

    # Conversation state
    default_system = SETTINGS.system_prompt
    history: list[tuple[str, str]] = []

    print(r"Terminal chat. Type \help for commands.")
    print()

    while True:
        try:
            user_in = input("you> ").rstrip("\n")
        except (EOFError, KeyboardInterrupt):
            print("\n\\exit")
            break

        if not user_in.strip():
            continue

        # Commands
        if user_in.startswith("\\"):
            parts = user_in[1:].strip().split(" ", 1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            try:
                if cmd == "help":
                    _print_help()

                elif cmd == "exit":
                    break

                elif cmd == "clear":
                    history = []
                    print("(cleared history)\n")

                elif cmd == "reset":
                    history = []
                    SETTINGS.system_prompt = default_system
                    SETTINGS.show_prompt = False
                    SETTINGS.stop_on_tags = True
                    SETTINGS.temperature = 0.8
                    SETTINGS.top_k = 40
                    SETTINGS.top_p = 0.95
                    SETTINGS.max_new_tokens = 200
                    SETTINGS.min_new_tokens = 10
                    print("(reset all settings)\n")

                elif cmd == "system":
                    if arg.strip():
                        SETTINGS.system_prompt = arg.strip()
                        print("(updated system prompt)\n")
                    else:
                        print("System prompt:")
                        print(SETTINGS.system_prompt)
                        print()

                elif cmd == "seed":
                    s = int(arg.strip())
                    torch.manual_seed(s)
                    torch.cuda.manual_seed(s)
                    print(f"(seed set to {s})\n")

                elif cmd == "params":
                    _print_params(SETTINGS)
                    print()

                elif cmd == "set":
                    bits = arg.split()
                    if len(bits) != 2:
                        raise ValueError(r"Usage: \set <key> <value>")
                    _set_param(SETTINGS, bits[0], bits[1])
                    print("(updated)\n")

                elif cmd == "prompt":
                    SETTINGS.show_prompt = not SETTINGS.show_prompt
                    print(f"(show_prompt = {SETTINGS.show_prompt})\n")

                elif cmd == "tags":
                    SETTINGS.stop_on_tags = not SETTINGS.stop_on_tags
                    print(f"(stop_on_tags = {SETTINGS.stop_on_tags})\n")

                else:
                    print(f"Unknown command: \\{cmd}")
                    print(r"Type \help for a list of commands.")
                    print()

            except Exception as e:
                print(f"Command error: {e}\n")

            continue

        # Normal user turn
        history.append(("user", user_in))

        prompt_text = format_history(SETTINGS.system_prompt, history)
        if SETTINGS.show_prompt:
            print("----- PROMPT START -----")
            print(prompt_text)
            print("----- PROMPT END -----\n")

        try:
            reply = generate_reply(model, encode, decode, prompt_text, eos_id, SETTINGS, ctx)
        except ValueError as e:
            # This most commonly happens if the prompt contains disallowed special tokens like <|endoftext|>
            print(f"Tokenizer error: {e}")
            print(r"If you pasted '<|endoftext|>' into the chat, remove it or clear history with \clear.")
            print()
            history.pop()  # remove last user turn
            continue

        print(f"bot> {reply}\n")
        history.append(("assistant", reply))


if __name__ == "__main__":
    main()