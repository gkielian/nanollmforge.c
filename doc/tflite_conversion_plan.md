# Plan: converting the custom ReaLLM-Forge models to TFLite/LiteRT (Pixel TPU target)

Status: **design complete, Phase 0/1 in progress**. Author: research-backed (4 web-research passes, 2026-07).
Decisions locked with the user: target the **Pixel-10 Tensor TPU** path; convert **smollm2_135M first**, then the custom **nsga_best3**; precision **full-integer int8**.

---

## 1. Goal (as decided)

Run our custom small LLMs on a **Pixel phone and offload compute to the Tensor TPU**, integrated via
**Android Studio**. Precision: **full-integer int8** (mandatory for any NPU/TPU delegate).

### Honest feasibility headline (from research — read before committing effort)

Offloading a **custom** (non-Gemini) model to the Pixel Tensor TPU is **narrowly possible and gated**:

- Reachable **only on Pixel 10 (Tensor G5)**, **only** via the **Google Tensor SDK** (LiteRT Torch /
  `CompiledModel`), which is **Beta, sign-up/approval-gated, and non-production** per its terms.
  Source: https://developers.google.com/edge/litert/next/tensor-sdk , https://developers.googleblog.com/google-tensor-sdk-beta-with-litert/
- **Pixel 6–9 have no supported path**: NNAPI (the old vendor-agnostic NPU route) is **deprecated in
  Android 15** and absent from Play Services LiteRT (GPU + XNNPACK only). The TPU there is reachable
  only by Google's Gemini Nano via **AICore**, which accepts prompt + small LoRA — never a custom
  architecture/weights. Source: https://developer.android.com/ndk/guides/neuralnetworks/migration-guide , https://developer.android.com/ai/gemini-nano
- The Tensor-TPU path is **AOT-compile-only** (offline: Ubuntu 22.04 x86_64, Bazel 7.4.1, 16–64 GB RAM →
  a Tensor-specific binary shipped via Play Feature Delivery / AI Packs), with **automatic CPU/GPU
  fallback** when ops don't compile. Source: https://developers.google.com/edge/litert/next/npu

**Therefore:** the TPU is pursued as the primary goal per the user's decision, but the plan keeps a
**GPU-delegate + XNNPACK-int8-CPU fallback as the always-shipping baseline**, because (a) it works on
every Pixel and (b) the TPU compile may push LLM ops back to CPU/GPU anyway. If HW-NPU (not Pixel
specifically) were the goal, Qualcomm QNN / MediaTek NeuroPilot are more mature — noted as a hedge.

---

## 2. The two models are very different difficulty

| | **smollm2_135M** (Tier-1 standard) | **nsga_best3** (custom infinite-head) |
|---|---|---|
| Arch | RoPE + GeGLU + RMSNorm + GQA, uniform dims, tied emb | infinite-head attn (v_dim≠qk_dim, non-square c_proj), per-layer hetero dims, identity layer, peri-LN, no-eps RMSNorm |
| litert-torch path | **Generative-API happy path** (SmolLM/Llama-3.2/AMD-Llama-135m are shipped examples) | **Fork the attention block** required; otherwise decomposes to unfused ops |
| Fused SDPA/KV-cache | Yes | Layers likely miss fusion → CPU fallback |
| TPU candidate? | **Yes** (Phase 3 stretch) | **No** — target GPU/CPU only |
| Order | **Phase 1** | **Phase 2** |

---

## 3. Toolchain (decided)

- **`litert-torch`** — the Jan-2026 rename of `ai-edge-torch` (old package deprecated). Ships (a) a
  generic `torch.export`-based `convert()` and (b) the **Generative API** (re-author with
  mobile-optimized transformer blocks → emits fused SDPA + KV-cache + multi-signature prefill/decode).
  Source: https://github.com/google-ai-edge/litert-torch , https://developers.google.com/edge/litert/conversion/pytorch/overview
- **Env pin:** Linux, **Python 3.11**, `torch ≥ 2.4`, `tf-nightly`, latest `litert-torch`, `ai-edge-litert`
  (interpreter). Being installed into an isolated conda env `tflite` (NOT the working `reallmforge` env).
- **Entry point:** the PyTorch float graph — the ReaLLM-Forge `GPT` (`/home/xinting/ReaLLM-Forge/model.py`,
  config `gpt_conf.py`) + checkpoint. NOT the `.rlm`/`.bin` C artifacts.
- **Interpreter/parity:** `ai_edge_litert.interpreter` (or `tf.lite.Interpreter`) to run the `.tflite`.

### Existing local assets to reuse (discovered)

- `/home/xinting/ReaLLM-Forge/exutorch/` — a prior **ExecuTorch** export experiment
  (`export_nanogpt_xnnpack.py`, `model.py`, `vocab.json`). Useful reference for the torch.export path
  and the XNNPACK int8 recipe; ExecuTorch is the fallback runtime (no Tensor-TPU backend though).
- conda env `coralnpu` — prior Coral/NPU exploration.
- `reallmforge/ref_dump.py` + `prove_parity.sh` — the **golden-reference + token-exact parity**
  methodology to extend to TFLite (see §6).

---

## 4. Phased plan

**Phase 0 — Env + parity harness (in progress).**
Isolated `tflite` conda env (py3.11) + `litert-torch`/`tf-nightly`/`ai-edge-litert` (installing).
Parity harness (`tflite/parity.py`) mirroring `prove_parity.sh`: PyTorch greedy golden ref
(`ref_dump.py`) diffed token-for-token against the LiteRT interpreter. Checkpoint-inspection helper
(`tflite/inspect_checkpoint.py`) to dump the `GPTConfig` + tensor shapes for weight mapping.

**Phase 1 — smollm2_135M, staged precision (the de-risk).**
1. Re-author with Generative-API blocks: RoPE cache, GQA via `num_query_groups`, RMSNorm, `GE_GLU`,
   exact-`GELU` enum; export **prefill + decode** signatures with a **static max-seq KV buffer**.
2. Map ReaLLM-Forge `GPT` state_dict → litert-torch block params (the fiddly bit — automate + verify
   shapes via `inspect_checkpoint.py`).
3. **fp32 → token-exact parity** vs PyTorch (should match, like the existing Tier-1 work).
4. **Weight-only int8** (`dynamic_wi8_afp32`) — correctness + confirm SDPA/KV **actually fused**
   (Model Explorer). Note: weight-only is **not** TPU-eligible; it's a correctness checkpoint.
5. **Full-integer int8 (W8A8)**, selective recipe (see §5), representative calibration set. Expect the
   Q8_0-style "tracks fp32 then diverges at logit ties" behavior, not token-exact.
6. **Android**: load via `com.google.ai.edge.litert`; GPU delegate + benchmark on the Pixel.

**Phase 2 — nsga_best3 (fork the attention).**
Fork `CausalSelfAttention` to support `v_dim ≠ qk_dim` + non-square concat→projection (the one real
code blocker). Express the rest in-config: per-layer `block_configs`, `epsilon=0`→tiny-eps RMSNorm,
peri-LN via pre/post-norm configs, identity layer as a custom pass-through block. Target a correct
**GPU/CPU** `.tflite`; do **not** expect TPU.

**Phase 3 — Pixel-10 Tensor TPU (the decided primary, gated).**
Prereqs the user owns: **(a) a Pixel 10**, **(b) Tensor SDK Beta access** (apply now — it's the long
pole), **(c)** accept AOT compile + non-production terms. Then AOT-compile the *standard* int8 model
via LiteRT Torch → `CompiledModel`, ship via Play Feature Delivery, and **audit the delegate partition
report** for op-fallback (LLM dynamic ops commonly demote off the TPU). Host note: Tensor SDK wants
Ubuntu 22.04 x86_64 — this box is 24.04, so use a **22.04 container** for that compile step (Bazel 7.4.1,
16–64 GB RAM; this host has 230 GB / 1.6 TB, ample).

---

## 5. Quantization strategy (full-int8, selective)

TPU/NPU need **full static-range int8**; weight-only is rejected. But naive all-int8 destroys LLM
quality — the fix is **mixed precision with float islands**:

- **int8 (per-axis symmetric weights, per-tensor activations)** on the bulk matmuls: FFN up/gate/down,
  attention q/k/v/out projections.
- **Keep in float/bf16:** token embedding, **LM head** (logits feed argmax — most sensitive),
  **softmax** (TFLite int8 softmax is a forced 1/256 grid; pre-softmax logits have outliers),
  **RMSNorm** (per-token statistic), **RoPE** (phase-sensitive).
- **Calibration:** representative dataset (a few hundred token windows from the training corpus, e.g.
  `ReaLLM-Forge/data/smollm-corpus`) via PT2E `HistogramObserver` to set activation ranges.
- **Custom recipe:** AI Edge Quantizer JSON to pin the float islands.
Sources: https://developers.google.com/edge/litert/conversion/tensorflow/quantization/quantization_spec , https://arxiv.org/pdf/2506.10443

---

## 6. Parity / acceptance (extends the repo's existing discipline)

The repo's bar is **token-exact greedy parity** vs PyTorch (not bitwise logits) — used for the Tier-1
and `.rlm` work. Apply the same, per precision:

- **fp32 / fp16 `.tflite`:** must be **token-exact** vs `ref_dump.py` greedy over ≥64 tokens, ≥3 prompts.
- **full-int8 `.tflite`:** pragmatic bar — coherent text; **tracks fp32 for N tokens then diverges at
  ties** (same as the shipped Q8_0 `.rlm`). Report the track-length + a small eval (e.g. perplexity
  delta) rather than exact match.

Harness now spans **three engines**: PyTorch `GPT` ↔ our C `.rlm`/`.bin` ↔ TFLite interpreter.

---

## 7. Challenges, ranked

1. **🔴 TPU access (platform, not code).** Pixel-10-only, gated Beta, non-production. *Mitigation:* apply
   for Tensor SDK Beta now (long pole); keep GPU+CPU baseline as the shipping target.
2. **🟠 Infinite-head attention fork** (v_dim≠qk_dim, non-square proj) — the only genuine re-authoring;
   also drops nsga_best3 off the fused fast path.
3. **🟠 Full-int8 parity** — won't be token-exact; use selective recipe + the pragmatic bar.
4. **🟡 No-epsilon RMSNorm landmine** — `rsqrt(0)→inf/NaN` + calibration blow-up. *Mitigation:* inject a
   sub-resolution epsilon (1e-12) at every norm incl. peri-LN. Preserves parity.
5. **🟡 Tied-embedding duplication** under XNNPACK (size hit; risk of two separate scales). *Mitigation:*
   one shared quant scale; verify the flatbuffer.
6. **🟡 Static-shape KV-cache** — truly dynamic tensors silently fall to CPU. *Mitigation:* static
   max-length buffer + position + mask; validate prefill AND decode partitioning.
7. **🟡 Heterogeneous dims** → less kernel reuse, bigger binary, some un-fused layers. *Mitigation:*
   accept; audit per-layer fusion in Model Explorer.
8. **🟡 Android tokenizer** — `bpe.h` is C; a LiteRT app needs a Java/Kotlin GPT-2 BPE (or bundle one).
9. **🟢 Op set is otherwise fine** — non-square proj, hetero dims, uneven GQA, identity layers, exact-erf
   GELU (int8 = 256-entry LUT, free), GQA repeat all convert cleanly.

---

## 8. Open inputs needed from the user

- **Exact smollm2 checkpoint dir** (containing `ckpt.pt` + `GPTConfig`) used to export
  `models/smollm2_135M.q8.bin` — not yet located under `ReaLLM-Forge`/`LLMForge`. Needed for Phase 1
  weight mapping. (`tflite/inspect_checkpoint.py <dir>` validates it.)
- **Tensor SDK Beta application** (Phase 3 long pole) + confirmation a **Pixel 10** is available.

---

## 9. Op → capability map (both models)

| Op | Converts? | Rewrite? | int8-safe? | Delegate/TPU? |
|---|---|---|---|---|
| RoPE (interleaved) | ✅ | no | keep float | ✅ float island |
| RMSNorm | ✅ | no | keep float | ✅ |
| RMSNorm no-eps | ✅ | inject tiny eps | only with eps | ✅ |
| GQA repeat (uneven ok) | ✅ | no | ✅ | ✅ |
| Causal softmax | ✅ | no | scores float | ✅ fused SDPA |
| exact-erf GELU | ✅ `approximate=false` | no | ✅ (256-LUT) | ✅ (check NPU) |
| tied embedding | ✅ | no | LM-head ≥fp16 | ✅ |
| dynamic KV-cache | ✅ static buf+mask | prefill/decode sig | ✅ if static | ✅ only if static |
| non-square out-proj | ✅ | no | ✅ | ✅ |
| per-layer hetero dims | ✅ | no | ✅ | ✅ less reuse |
| identity attention | ✅ | no | n/a | ✅ |
| peri-LN extra norm | ✅ | +tiny eps | keep float | ✅ (float islands) |
