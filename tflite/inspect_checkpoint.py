"""
Inspect a ReaLLM-Forge checkpoint for the TFLite/LiteRT conversion (Phase 0).

Dumps the GPTConfig and the parameter name -> shape/dtype table so we can map the
ReaLLM-Forge `GPT` state_dict onto the litert-torch Generative-API blocks (Phase 1),
and validate that a directory really is the smollm2_135M source checkpoint.

Usage:
  python tflite/inspect_checkpoint.py <ckpt_dir_or_ckpt.pt> [--reallm /home/xinting/ReaLLM-Forge]

Runs with plain torch (no litert-torch needed) — use the `reallmforge` or `tflite` conda env.
See doc/tflite_conversion_plan.md.
"""
import argparse
import os
import sys
from dataclasses import asdict, is_dataclass


def _add_reallm_to_path(reallm):
    # ckpt.pt may pickle references to ReaLLM-Forge classes (GPTConfig, etc.); make them importable.
    for cand in [reallm, "/home/xinting/ReaLLM-Forge", "/home/xinting/LLMForge"]:
        if cand and os.path.isdir(cand) and cand not in sys.path:
            sys.path.insert(0, cand)


def _resolve_ckpt(path):
    if os.path.isdir(path):
        p = os.path.join(path, "ckpt.pt")
        if not os.path.isfile(p):
            sys.exit(f"no ckpt.pt in directory {path!r}")
        return p
    if os.path.isfile(path):
        return path
    sys.exit(f"not found: {path!r}")


def _find_state_dict(obj):
    if isinstance(obj, dict):
        for k in ("model", "state_dict", "model_state_dict", "net"):
            v = obj.get(k)
            if isinstance(v, dict) and v and all(hasattr(t, "shape") for t in v.values()):
                return v, k
        # maybe obj itself is a flat tensor dict
        if obj and all(hasattr(t, "shape") for t in obj.values()):
            return obj, "<root>"
    sys.exit("could not locate a state_dict (tensor mapping) in the checkpoint")


def _find_config(obj):
    if isinstance(obj, dict):
        for k in ("model_args", "config", "gptconf", "gpt_config", "args"):
            if k in obj:
                return obj[k], k
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="checkpoint dir (containing ckpt.pt) or a .pt file")
    ap.add_argument("--reallm", default="/home/xinting/ReaLLM-Forge",
                    help="ReaLLM-Forge source dir (for unpickling class refs)")
    args = ap.parse_args()

    _add_reallm_to_path(args.reallm)
    import torch  # after path setup

    ckpt_path = _resolve_ckpt(args.ckpt)
    print(f"# loading {ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg, cfg_key = _find_config(obj)
    print(f"\n## GPTConfig  (from key: {cfg_key})")
    if cfg is None:
        print("  <none found>")
    else:
        cfg_dict = asdict(cfg) if is_dataclass(cfg) else (dict(cfg) if isinstance(cfg, dict) else vars(cfg))
        # print only the fields that are set to a non-default-ish / conversion-relevant value
        relevant = [
            "n_layer", "n_head", "n_kv_group", "n_embd", "block_size", "vocab_size",
            "activation_variant", "mlp_variant", "mlp_size", "mlp_expansion_factor",
            "norm_variant_attn", "norm_variant_output", "use_rotary_embeddings", "rope_variant",
            "use_abs_pos_embeddings", "wte_weight_tying", "use_pre_ln", "use_peri_ln", "use_post_ln",
            "attention_variant", "use_concat_heads", "n_qk_head_dim", "n_v_head_dim", "n_cproj",
            "n_head_layerlist", "n_qk_head_dim_layerlist", "n_v_head_dim_layerlist",
            "mlp_size_layerlist", "attention_variant_layerlist", "n_kv_group_layerlist",
        ]
        for k in relevant:
            if k in cfg_dict and cfg_dict[k] not in (None, [], "", False, 0):
                print(f"  {k} = {cfg_dict[k]}")
        print(f"  (+{len(cfg_dict)} total config fields; use --reallm's gpt_conf.py for the rest)")

    sd, sd_key = _find_state_dict(obj)
    print(f"\n## state_dict  (from key: {sd_key}; {len(sd)} tensors)")
    total = 0
    rows = []
    for name, t in sd.items():
        clean = name.replace("_orig_mod.", "")  # torch.compile prefix
        n = 1
        for d in t.shape:
            n *= d
        total += n
        rows.append((clean, tuple(t.shape), str(t.dtype).replace("torch.", "")))
    for name, shape, dt in sorted(rows):
        print(f"  {name:52s} {str(shape):22s} {dt}")
    print(f"\n## total parameters: {total:,} ({total/1e6:.1f}M)")


if __name__ == "__main__":
    main()
