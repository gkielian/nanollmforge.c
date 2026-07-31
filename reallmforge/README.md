# ReaLLM-Forge → llama2.c

Tooling to export a model trained in [ReaLLM-Forge](https://github.com/) and run it in this
pure-C engine (`src/run.c` / `src/runq.c`), including on Android. Full design, validation, and gate
results: [`../doc/reallmforge_to_llama2c.md`](../doc/reallmforge_to_llama2c.md).

## Files
- `export_reallmforge.py` — ReaLLM-Forge `ckpt.pt` → llama2.c `.bin` (`--version 0` fp32 for src/run.c,
  `--version 2` Q8_0 for src/runq.c). Validates the supported "Tier-1" architecture subset (uniform dims).
- `export_reallm_hetero.py` — ReaLLM-Forge `ckpt.pt` → `.rlm` (per-layer format) for **NSGA models with
  infinite-head attention**: heterogeneous per-layer dims, `infinite`/`identity` attention, GQA, peri-LN.
  `--version 1` fp32 (`make runreallm` → `./run_reallm model.rlm ...`, token-exact vs PyTorch);
  `--version 2` Q8_0 int8, ~4× smaller (`make runqreallm` → `./runq_reallm model.q8.rlm ...`).
  Design + parity results: [`../doc/reallmforge_hetero_infinite.md`](../doc/reallmforge_hetero_infinite.md).
- `export_gpt2_tokenizer.py` — tiktoken "gpt2" ranks → `tokenizer_gpt2.bin` for the C BPE tokenizer.
- `ref_dump.py` — PyTorch fp32 CPU greedy reference, for numeric-parity validation.
- `run_live.sh` — live inference on the bundled model; interactive REPL or one-shot. Streams tokens.
  `reallmforge/run_live.sh` (type prompts) or `reallmforge/run_live.sh "Once upon a"`.
  Env knobs: `MODEL TOK ENGINE TEMP TOPP STEPS` (e.g. `TEMP=0` for greedy).
- `prove_parity.sh` — end-to-end proof the C engine == the original PyTorch model, token-for-token.
  `reallmforge/prove_parity.sh [CKPT_DIR] [PROMPT] [N_STEPS]`.

The C-side tokenizer lives in `../src/bpe.h` (compiled into the engines); its unit test is `../src/bpe_test.c`.

## Quickstart (desktop)
```bash
# from repo root
python reallmforge/export_reallmforge.py <ckpt_dir> smollm2_135M_v2.bin --version 2
python reallmforge/export_gpt2_tokenizer.py tokenizer_gpt2.bin
make rungelu
./runq_gelu smollm2_135M_v2.bin -g tokenizer_gpt2.bin -i "The quick brown fox" -t 0 -n 64
```

## Android
```bash
make runq_android ANDROID_NDK=$HOME/android-ndk-r27c        # arm64 (phones / Apple-Silicon AVD)
make runq_android_x86_64 ANDROID_NDK=$HOME/android-ndk-r27c # x86_64 (x86-host AVD)
```
See the design doc's "Android deploy" section for the `adb push` + run recipe.
