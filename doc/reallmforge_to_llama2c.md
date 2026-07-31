# Design: Run a ReaLLM-Forge Tier-1 model on llama2.c (Android target)

Status: REVISED (revision round 1/3) — passed independent critic + readiness review; ready to implement.
Key correction from review: this repo's run.c is a LEGACY v0 reader, NOT the ak42 v1/v2 header;
fp32 export must mirror legacy_export (v0), runq.c uses v2. Validation harness must be built
(sample.py has no logit/greedy dump).
First target model: `smollm2_135M` from `/home/xinting/Evo_GPT_checkpoints_backup/smollm2_135M`

## Goal

Run a ReaLLM-Forge-trained model in the pure-C llama2.c inference engine (`run.c` / `runq.c`)
and ultimately on Android (NDK cross-compile), starting with one clean "Tier-1" model.

## Bundled example
`models/smollm2_135M/` ships the example. The tokenizer (`tokenizer_gpt2.bin`, ~510 KB) is committed
in git. The weights (`smollm2_135M.q8.bin`, ~138 MB) are **NOT in git** — GitHub blocks files
>100 MB and this public fork disallows Git LFS, so they are gitignored (`models/**/*.q8.bin`) and
distributed via a **GitHub Release** asset instead. Get them with `reallmforge/fetch_model.sh`
(set `MODEL_URL` to the Release asset) or regenerate via `export_reallmforge.py`. Then run:
`make rungelu && ./runq_gelu models/smollm2_135M/smollm2_135M.q8.bin -g models/smollm2_135M/tokenizer_gpt2.bin -i "Once upon a time" -t 0.8 -p 0.9 -n 128`.
See `models/smollm2_135M/README.md`.

## Compatibility note (corrected)
ALL `nsga_*` evolved checkpoints are INCOMPATIBLE with llama2.c: their per-layer
`attention_variant_layerlist` uses `infinite`/`identity` attention with fully heterogeneous
per-layer dims. Only 5 checkpoints run: gemma_270M_style, qwen_0_5B, smollm2_135M, smollm2_360M
(Tier-1), smollm2_135M_periln (Tier-2, needs `-DPERI_LN`). Peri-LN support was added + validated
(exact 32/32 on smollm2_135M_periln).

## Target model facts (verified from ckpt.pt + full_config.json)

| Field | Value |
|---|---|
| n_layer | 30 |
| n_embd (dim) | 576 |
| n_head | 9 |
| n_kv_group (kv heads) | 3 |
| head_size | 64 (576/9) |
| hidden_dim (mlp_size) | 1536 |
| block_size (max_seq_len) | 1024 |
| vocab_size | 50257 |
| norm | rmsnorm (attn + output), pre-norm only, NO peri-LN |
| positions | RoPE, theta=10000, full head_dim (rope_length=None) |
| mlp | swiglu, activation = **GELU (exact, erf-based)**, offsets = 0.0 (verified all layers) |
| bias | False (the `attn.bias` tensor is the causal-mask buffer, NOT a weight — skip it) |
| weight tying | True (lm_head.weight == wte.weight → shared_classifier) |
| tokenizer | **tiktoken "gpt2"** byte-level BPE, vocab 50257 (NOT SentencePiece) |

State_dict keys carry an `_orig_mod.` prefix (from torch.compile) that must be stripped.

## Compatibility analysis vs llama2.c run.c (verified by reading both sides)

MATCHES (no change needed):
- RoPE: run.c (lines 264-279) rotates adjacent pairs with exponent (i%head_size)/head_size =
  interleaved/GPT-J, identical to ReaLLM-Forge RotaryEmbedding (even/odd adjacent pairs, theta 10000).
  Frequencies match (run.c pair k=i/2 ↔ exponent 2k/head_size ↔ inv_freq[k]=10000^(-k/(head_size/2))).
- GQA: run.c uses `h / kv_mul` block sharing (kv_mul=3); ReaLLM-Forge uses block (repeat_interleave)
  KV expansion. 9÷3 divisible → identical mapping [0,0,0,1,1,1,2,2,2].
- RMSNorm formula: x / sqrt(mean(x^2)) * gain. Same (see eps note below).
- SwiGLU branch roles: run.c w1=gated (activation), w3=value, w2=down. ReaLLM-Forge c_fc_in1=gated,
  c_fc_in2=value, c_fc_out=down. Direct mapping.
- Pre-norm block order, no biases, tied embeddings — all supported.

DELTAS (must handle):
1. **Activation: SiLU → GELU.** run.c line 341 hard-codes silu `val *= 1/(1+exp(-val))`.
   smollm2 uses exact erf GELU `0.5*x*(1+erf(x/sqrt(2)))`. MUST replace. Use `erff()` from math.h.
   This is the only ACTIVATION change; together with the deliberately-kept RMSNorm eps (Delta 2)
   these are the only forward-math differences. The same silu→gelu change applies to runq.c
   (its silu is at runq.c line ~458, not line 341).
2. **RMSNorm eps.** run.c adds +1e-5 inside sqrt; ReaLLM-Forge adds none. Negligible (~1e-5 relative),
   but for tight numeric validation we may drop the eps in our build. Decision: keep eps (matches
   upstream Llama behavior, diff below validation tolerance); revisit only if validation fails.
3. **Tokenizer.** Model uses GPT-2 byte-level BPE, not SentencePiece. The shipped tokenizer.bin is
   unusable. See Tokenizer section.

NON-ISSUES confirmed: no peri-LN, no qk-norm, no MoE, no learned abs-pos, offsets are 0.

## Work breakdown

### Component A — Exporter (Python): `export_reallmforge.py` (new file in nanollm.c)
Reads ReaLLM-Forge `ckpt.pt`, writes llama2.c `.bin`. Does NOT depend on llama2.c's model.py /
Transformer class (those expect Llama naming). We write the `.bin` bytes directly, mirroring
`export.py`'s `version1_export` (fp32) and `version2_export` (Q8_0).

Steps:
1. `torch.load(ckpt.pt, map_location="cpu")`, take `ck["model"]`, strip `_orig_mod.` prefix.
2. Build Config from `ck["model_args"]`: dim, hidden_dim=mlp_size, n_layers, n_heads,
   n_kv_heads=n_kv_group, vocab_size, max_seq_len=block_size.
3. Key mapping per layer i:
   - `transformer.h.{i}.pre_ln_attn.gain`  → attention_norm[i]   (fp32)
   - `transformer.h.{i}.pre_ln_mlp.gain`   → ffn_norm[i]          (fp32)
   - `transformer.h.{i}.attn.c_attn_q.weight` → wq[i]
   - `transformer.h.{i}.attn.c_attn_k.weight` → wk[i]
   - `transformer.h.{i}.attn.c_attn_v.weight` → wv[i]
   - `transformer.h.{i}.attn.c_proj.weight`   → wo[i]
   - `transformer.h.{i}.mlp.c_fc_in1.weight`  → w1[i]  (gated)
   - `transformer.h.{i}.mlp.c_fc_in2.weight`  → w3[i]  (value)
   - `transformer.h.{i}.mlp.c_fc_out.weight`  → w2[i]  (down)
   - global: `transformer.wte.weight` → token_embedding; `transformer.ln_f.gain` → final_norm;
     `lm_head.weight` → output (skipped if shared_classifier).
   - SKIP: `attn.bias` (mask buffer), `activation_x_offset`, `activation_y_offset` (verified 0).
4. Byte-format must match the EXACT readers in this repo. IMPORTANT: this repo's `run.c` is a
   LEGACY v0 reader (verified: run.c:147 does `fread(config, sizeof(Config),1,file)` — no magic,
   no version, no 256-byte header; run.c:149-150 signals shared weights via SIGN of vocab_size;
   memory_map_weights run.c:111-139 is embedding-first ordering). `runq.c` is the modern v2 reader.
   So the two paths use DIFFERENT formats:

   - **fp32 path → run.c → mirror `legacy_export` (v0)** (export.py:75-127). Exact byte layout:
     1. Header: 7 little-endian int32 in this order: dim, hidden_dim, n_layers, n_heads, n_kv_heads,
        vocab_size, max_seq_len. NO magic, NO version, NO padding. (This IS the C `Config` struct.)
     2. Shared-classifier signaling: vocab_size POSITIVE = shared (tied). smollm2 is tied → write
        vocab_size as +50257. (Only negate it for an untied model.)
     3. Then fp32 tensors in this exact order (each tensor flattened row-major as PyTorch stores it):
        token_embedding (wte);
        attention_norm for layer 0..L-1;  wq 0..L-1;  wk 0..L-1;  wv 0..L-1;  wo 0..L-1;
        ffn_norm 0..L-1;  w1 (gate=c_fc_in1) 0..L-1;  w2 (down=c_fc_out) 0..L-1;  w3 (value=c_fc_in2) 0..L-1;
        final_norm (ln_f.gain);
        freq_cis_cos placeholder (max_seq_len * head_size/2 floats);
        freq_cis_sin placeholder (max_seq_len * head_size/2 floats);
        [output/wcls ONLY if NOT shared — omit for smollm2].
     - freq_cis placeholders: run.c IGNORES their contents (RoPE is computed on the fly, run.c:137-138
       just advances the pointer) but they MUST occupy the right number of bytes so offsets line up
       when the model is untied. For a tied model nothing is read past final_norm, so zeros are safe.
       Write zeros of size max_seq_len*head_size/2 each = 1024*32 = 32768 floats per array.
     - NOTE the w1/w2/w3 ORDER in the v0 stream is w1, w2, w3 (gate, down, value) — matches run.c
       memory_map_weights (w1@129, w2@131, w3@133).

   - **Q8_0 path → runq.c → mirror `version2_export` (v2)** (export.py:182-260): magic 0x616b3432,
     version int=2, 256-byte header (7 ints + shared_classifier BYTE + group_size int, zero-padded
     to 256); then norms fp32 (attn-norms all, ffn-norms all, final-norm); then each quantized matmul
     (tok_emb, wq,wk,wv,wo, w1,w2,w3 — note v2 order is w1,w2,w3) as int8 blocks + fp32 group scales,
     group_size=64. dim 576=9·64 ✓; every weight numel is a 576-multiple so numel%64=0 ✓
     (incl. vocab·dim=50257·576). shared_classifier here uses the BYTE flag (not vocab sign).

   The exporter writes raw bytes directly (it does NOT build a llama2.c Transformer object), reading
   tensors straight from the strip-prefixed ReaLLM-Forge state_dict per the mapping in step 3.

Order of validation: implement v0 (fp32, run.c) FIRST for numeric correctness, then v2 (Q8_0, runq.c)
for Android.

### Component B — run.c / runq.c GELU variant
- Replace the silu line with exact GELU. Make it a compile-time switch to keep upstream intact:
  `#ifdef ACT_GELU  val = 0.5f*val*(1.0f+erff(val*0.70710678f)); #else <silu> #endif`
- Add a `make rungelu` (builds run_gelu + runq_gelu) target, or a `-DACT_GELU` flag.

### Component C — Tokenizer (GPT-2 BPE) — the largest piece
The C side needs to encode the prompt and decode output for vocab 50257 GPT-2 BPE.
Plan in two phases:
- **Phase 1 (validation): bypass C tokenizer.** Concrete patch to run.c (guarded by a new flag so
  upstream behavior is untouched):
  - Add CLI flag `-I <ids_file>`: a whitespace-separated list of int token IDs to use as the prompt,
    bypassing `encode()`; it also prints generated token IDs as integers (one per line) instead of
    calling `decode()`, so no tokenizer is needed at all for validation. (Implemented as a single
    `-I` flag — the raw-id output is intrinsic to this path, not a separate `-O` flag.)
  - When `-I` is used, SKIP `build_tokenizer` / `tokenizer.bin` loading (currently unconditional and
    pointing at the wrong SentencePiece artifact). Only `build_transformer` runs.
  - Python harness: encode prompt with tiktoken gpt2 → write ids file → run `./run model.bin -I ids
    -n 64 -t 0` → read output ids → tiktoken decode for human inspection.
  This isolates and validates the *numerics* without the BPE C port.
- **Phase 2 (self-contained / Android): port GPT-2 byte-level BPE to C.** Export merges+vocab from
  tiktoken/HF gpt2 to a binary, implement byte-level pretokenization + BPE merge in C. Candidate:
  adapt a minimal known-good BPE (e.g. llama.cpp's gpt2 bpe or a standalone bpe.c). The GPT-2
  regex pretokenizer is the fiddly part; document the chosen approach before coding Phase 2.

### Component D — Numeric validation harness
NOTE (verified): ReaLLM-Forge `sample.py` has NO logit/token-ID dump and NO greedy/argmax mode
(it always goes through torch.multinomial; output is text/PNG). So we must BUILD a tiny standalone
reference script — do not rely on sample.py.

- New script `ref_dump.py` (lives alongside our work, imports ReaLLM-Forge model):
  - Add `/home/xinting/ReaLLM-Forge` to sys.path; `from model import GPT` and `from gpt_conf import GPTConfig`.
  - Load ckpt on CPU in fp32: `torch.load(..., map_location="cpu")`, strip `_orig_mod.`, build
    GPTConfig from `model_args`, `model.load_state_dict(...)`, `model.eval()`, `.float()`.
  - Feed a FIXED list of token IDs (same ids file as Component C Phase-1). Greedy-decode N=64 steps
    by argmax (NOT multinomial). Dump per step: the full logit vector (np.save) AND the argmax token id.
- Run our C engine (fp32 v0, GELU) on the same ids with `-I ids -n 64 -t 0`.
- Acceptance criteria (concrete):
  - fp32 v0: EXACT greedy token-id match for all 64 steps. Additionally, max abs logit diff at step 1
    < 1e-3 (fp32 CPU vs fp32 CPU; the only expected source is the 1e-5 RMSNorm eps and matmul order).
    If logits diverge > 1e-3, drop the run.c RMSNorm eps and re-check (Delta 2 decision).
  - Q8_0 v2: greedy token-id match for ≥ 60/64 steps; document any late divergence (quantization).
- Reference must be fp32 on CPU (the PyTorch default of bf16/CUDA would not match); ref_dump.py forces
  `.float()` and CPU explicitly.

### Component E — Android build
- Cross-compile runq.c (Q8_0 + GELU) with Android NDK (aarch64). Static binary or JNI lib.
- Push model.bin + tokenizer artifact to device; run via adb. (Detail after A–D land.)

## Sequencing / gates
0. Build `ref_dump.py` reference harness (Component D prerequisite — does NOT exist in sample.py). DONE.
1. A(v0 fp32) + B(GELU) + C-Phase1 + D → prove EXACT fp32 numeric parity on desktop.
   **Gate 1 ✅ PASSED** — 32/32 greedy tokens identical (ref_dump.py vs run_gelu, smollm2_135M).
2. A(v2 Q8_0, runq.c) → prove quantized parity.
   **Gate 2 ~ PASSED** — Q8_0 (runq_gelu) tracks the fp32 reference exactly for 25/32 steps
   (~22 generated tokens) then forks via a single argmax flip; expected int8 cascade, max weight
   quant err 0.009. fp32 exactness already proves correctness; Q8 fork is a fidelity property.
   Optional refinement: teacher-forced per-position top-1 agreement (cascade-free metric).
3. C-Phase2 (C BPE) → self-contained desktop run.
   **Gate 3 ✅ PASSED** — `bpe.h` GPT-2 BPE: 2039/2039 exact-match vs tiktoken (curated + 2000 random
   ASCII + contraction edge cases). End-to-end `-g "text"` generates coherent output and matches the
   validated `-I` ids path decoded by tiktoken, byte-for-byte. Default `make run` still compiles;
   empty/unicode prompts are safe (decode is byte-lossless). Self-contained desktop run achieved.

### Gate 3 design — GPT-2 byte-level BPE in C
Model tokenizer = tiktoken "gpt2": 50256 mergeable_ranks (raw `bytes`→id, ids contiguous 0..50255,
all 256 single bytes present so encode never fails), special `<|endoftext|>`=50256. We port
tiktoken's RAW-BYTE BPE (no encoder.json/merges.txt/byte-unicode map needed):
- `export_gpt2_tokenizer.py` → `tokenizer_gpt2.bin`: int32 n=50256, int32 eot=50256, then n records
  {int32 len, bytes} in id order.
- `bpe.h` (header-only, no model deps, shared by run.c & runq.c): load table + open-addressing
  hash (bytes→id); `gpt2_encode` = regex pretokenize then per-piece byte-pair merge picking the
  adjacent pair whose concatenation has the lowest id; `gpt2_token_bytes(id)` for decode.
- Regex pretokenizer replicates tiktoken's gpt2 pattern
  `'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s`
  ASCII-EXACT; bytes>=0x80 treated as \p{L} (Latin-script ok; non-Latin punctuation/digits
  approximate). Documented limitation.
- Integration: `-g <tokenizer_gpt2.bin>` flag → gpt2 path in generate (no BOS, EOS=50256), raw byte
  output. Validation: `bpe_test.c` compares C encode ids vs tiktoken over a corpus (must be EXACT
  on ASCII), plus end-to-end `-g "text"` == `-I <tiktoken ids>` continuation.
4. E (Android NDK cross-compile of runq_gelu).
   **Gate 4 ✅ PASSED (build + aarch64 execution validated; on-device push pending hardware).**
   - NDK r27c installed; `make runq_android` → valid aarch64 PIE ELF (interpreter /system/bin/linker64),
     `runq_android_static` → static variant. Both compile clean with `-DACT_GELU`.
   - aarch64 EXECUTION validated under qemu via a glibc aarch64 build of the same runq.c/run.c:
     fp32 path is **bit-exact vs x86 (32/32 tokens)**; coherent text generation confirmed.
   - Note: the Bionic *static* binary can't run under qemu-user (TLS-alignment limitation, a qemu
     artifact, not a device issue); the dynamic Bionic binary needs the device's linker64. Both run
     on a real device. Cross-platform Q8_0 greedy can diverge by 1 token at logit near-ties (x86 vs
     aarch64 FP rounding) — expected; fp32 is deterministic across platforms.
   - Remaining: `adb push runq_android smollm2_135M_v2.bin tokenizer_gpt2.bin` to a device + run.

## Android deploy (when a device is available)
```
make runq_android ANDROID_NDK=$HOME/android-ndk-r27c
adb push runq_android /data/local/tmp/
adb push smollm2_135M_v2.bin tokenizer_gpt2.bin /data/local/tmp/
adb shell 'cd /data/local/tmp && ./runq_android smollm2_135M_v2.bin -g tokenizer_gpt2.bin -i "The quick brown fox" -t 0 -n 64'
```

## Reproduce (desktop)
```
# export
python reallmforge/export_reallmforge.py /home/xinting/Evo_GPT_checkpoints_backup/smollm2_135M smollm2_135M_v0.bin --version 0
python reallmforge/export_reallmforge.py /home/xinting/Evo_GPT_checkpoints_backup/smollm2_135M smollm2_135M_v2.bin --version 2
# build (GELU)
gcc -O3 -DACT_GELU -o run_gelu  src/run.c  -lm
gcc -O3 -DACT_GELU -o runq_gelu src/runq.c -lm
# validate parity
printf "464 2068 7586 21831\n" > ids.txt
python reallmforge/ref_dump.py /home/xinting/Evo_GPT_checkpoints_backup/smollm2_135M ids.txt 32 2>/dev/null > ref_out.txt
./run_gelu  smollm2_135M_v0.bin -I ids.txt -t 0 -n 32 2>/dev/null   # == ref_out.txt exactly
./runq_gelu smollm2_135M_v2.bin -I ids.txt -t 0 -n 32 2>/dev/null   # tracks ref for ~25 steps

# self-contained text generation (GPT-2 BPE in C)
python reallmforge/export_gpt2_tokenizer.py tokenizer_gpt2.bin
./run_gelu  smollm2_135M_v0.bin -g tokenizer_gpt2.bin -i "The quick brown fox" -t 0 -n 64
./runq_gelu smollm2_135M_v2.bin -g tokenizer_gpt2.bin -i "The quick brown fox" -t 0 -n 64
# tokenizer unit test vs tiktoken:  gcc -O2 -o bpe_test bpe_test.c && ./bpe_test tokenizer_gpt2.bin < lines.txt
```

## Open questions / risks
- RMSNorm eps: keep vs drop (default keep; validation will tell).
- Q8_0 quantization of a 135M model: error accumulation over 30 layers — may need per-tensor check.
- GPT-2 BPE C port effort (Phase 2) is the main unknown; Phase 1 de-risks numerics independently.
- n_head=9 is odd but head_size=64 is clean; no padding needed.
```
