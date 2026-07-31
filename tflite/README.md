# TFLite / LiteRT conversion track

Convert the custom ReaLLM-Forge models to LiteRT (TFLite) for on-device Android, targeting the
**Pixel Tensor TPU** (full-int8), with a GPU-delegate + XNNPACK-int8-CPU baseline that ships on any
Pixel. Full design, feasibility research, challenges, and the phased plan:
[`../doc/tflite_conversion_plan.md`](../doc/tflite_conversion_plan.md).

Conversion starts from the **PyTorch** `GPT` (`/home/xinting/ReaLLM-Forge`) + checkpoint — NOT the
`.rlm`/`.bin` C artifacts. This track is independent of the C engines under `src/`.

## Env (conversion workstation — this Linux box)

Isolated conda env `tflite` (kept separate from `reallmforge`):
```bash
conda create -y -n tflite python=3.11
conda run -n tflite pip install litert-torch "torch>=2.4" ai-edge-litert numpy
# verified: litert_torch 0.9.1, torch 2.12.1
```

## Files

- `inspect_checkpoint.py` — dump a checkpoint's `GPTConfig` + param name→shape table (weight-mapping
  aid; validates a dir is the right source ckpt). Runs with plain torch.
  `python tflite/inspect_checkpoint.py <ckpt_dir>`
- `parity.py` — acceptance gate. `inspect` a `.tflite`'s signatures/tensors; `generate` greedy tokens
  to diff against `reallmforge/ref_dump.py` (token-exact for fp32/fp16; track-then-diverge for int8).
- `convert_smollm2.py` — *(Phase 1, TODO)* re-author smollm2 in the litert-torch Generative API, load
  weights, export prefill/decode signatures, quantize.

## Status

- [x] Phase 0 — env + parity/inspect scaffolding
- [ ] Phase 1 — smollm2_135M: fp32 → weight-only int8 → full-int8; Android GPU-delegate benchmark
- [ ] Phase 2 — nsga_best3: fork attention (v_dim≠qk_dim, non-square proj), hetero/peri-LN/identity
- [ ] Phase 3 — Pixel-10 Tensor TPU AOT compile (needs Tensor SDK Beta access + a Pixel 10)

## Blocked on / needed

- Exact **smollm2 checkpoint dir** (`ckpt.pt` + `GPTConfig`) used for `models/smollm2_135M.q8.bin`.
- **Tensor SDK Beta** application (Phase 3 long pole) + a **Pixel 10**.
