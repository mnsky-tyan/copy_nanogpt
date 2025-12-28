'''
inspect the checkpoint file
'''

import torch

CKPT_PATH = "ckpt.pt"
N_KEYS = 30        # how many keys to show
N_TENSORS = 10     # how many tensors to sample from state_dict
N_OPT_STATE = 3    # how many optimizer state entries to sample


def main():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")

    print("type(ckpt):", type(ckpt))
    if not isinstance(ckpt, dict):
        print("ckpt is not a dict, printing raw:")
        print(ckpt)
        return

    print("\nTop-level keys:")
    for k in ckpt.keys():
        print(f"  {k}: {type(ckpt[k])}")

    # Raw prints for common dict fields
    if "iter_num" in ckpt:
        print("\niter_num:", ckpt["iter_num"])
    if "best_val_loss" in ckpt:
        print("best_val_loss:", ckpt["best_val_loss"])

    if "model_args" in ckpt:
        print("\nmodel_args (raw):")
        print(ckpt["model_args"])

    if "config" in ckpt:
        print("\nconfig (raw):")
        print(ckpt["config"])

    # Model weights
    if "model" in ckpt:
        sd = ckpt["model"]
        print("\nmodel / state_dict:")
        print("  type:", type(sd))
        if isinstance(sd, dict):
            keys = list(sd.keys())
            print("  num keys:", len(keys))
            print(f"  first {min(N_KEYS, len(keys))} keys:")
            for name in keys[:N_KEYS]:
                v = sd[name]
                if torch.is_tensor(v):
                    print(f"    {name:35}  tensor  shape={tuple(v.shape)} dtype={v.dtype}")
                else:
                    print(f"    {name:35}  {type(v)}")

            # show a few random-ish sample tensors from later too
            print(f"\n  sample {min(N_TENSORS, len(keys))} tensors (spread out):")
            if keys:
                step = max(1, len(keys) // min(N_TENSORS, len(keys)))
                sample_names = keys[::step][:N_TENSORS]
                for name in sample_names:
                    v = sd[name]
                    if torch.is_tensor(v):
                        print(f"    {name:35}  shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(sd)

    # Optimizer
    if "optimizer" in ckpt:
        opt = ckpt["optimizer"]
        print("\noptimizer:")
        print("  type:", type(opt))
        if isinstance(opt, dict):
            print("  keys:", list(opt.keys()))

            if "param_groups" in opt:
                print("\n  optimizer['param_groups'] (raw):")
                print(opt["param_groups"])

            if "state" in opt:
                st = opt["state"]
                print("\n  optimizer['state']:")
                print("    type:", type(st))
                if isinstance(st, dict):
                    state_keys = list(st.keys())
                    print("    num state entries:", len(state_keys))
                    print(f"    first {min(N_KEYS, len(state_keys))} state param-ids:")
                    print("    ", state_keys[:N_KEYS])

                    print(f"\n    sample {min(N_OPT_STATE, len(state_keys))} state entries (raw keys + tensor shapes):")
                    for pid in state_keys[:N_OPT_STATE]:
                        entry = st[pid]
                        print(f"      param_id {pid}: type={type(entry)}")
                        if isinstance(entry, dict):
                            for kk, vv in entry.items():
                                if torch.is_tensor(vv):
                                    print(f"        {kk:12} tensor shape={tuple(vv.shape)} dtype={vv.dtype}")
                                else:
                                    print(f"        {kk:12} {vv!r}")
        else:
            print(opt)


if __name__ == "__main__":
    main()