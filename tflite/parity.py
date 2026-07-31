"""
Parity harness for the TFLite/LiteRT conversion — the acceptance gate (Phase 0).

Extends the repo's existing `reallmforge/prove_parity.sh` discipline (PyTorch golden
reference via `ref_dump.py`, token-exact greedy diff) to the LiteRT interpreter, so the
same bar applies across THREE engines: PyTorch GPT  <->  our C .rlm/.bin  <->  TFLite.

Two modes:
  inspect   Load a .tflite and print its signatures + input/output tensor details.
            (Runnable NOW — use it to finalize the greedy loop's I/O names in Phase 1.)
  generate  Greedy-decode `-n` tokens from a seed ids file and print the ids, one per
            line, in the same format ref_dump.py emits, so `diff` proves token-exact parity.

Usage:
  python tflite/parity.py inspect  model.tflite
  python tflite/parity.py generate model.tflite --ids ids.txt -n 64 > c_tflite.txt
  # then: diff c_tflite.txt <(python reallmforge/ref_dump.py <ckpt_dir> ids.txt 64)

Env: the `tflite` conda env (needs ai-edge-litert / tf.lite). See doc/tflite_conversion_plan.md.
"""
import argparse
import sys


def _load_interpreter(path):
    try:
        from ai_edge_litert.interpreter import Interpreter  # LiteRT runtime
    except Exception:
        try:
            from tensorflow.lite import Interpreter  # fallback: tf.lite
        except Exception as e:
            sys.exit(f"no LiteRT interpreter available ({e}); pip install ai-edge-litert in the tflite env")
    it = Interpreter(model_path=path)
    it.allocate_tensors()
    return it


def cmd_inspect(args):
    it = _load_interpreter(args.model)
    sigs = it.get_signature_list()
    print(f"# {args.model}")
    print(f"## signatures ({len(sigs)}):")
    for name, spec in sigs.items():
        print(f"  '{name}': inputs={spec.get('inputs')}  outputs={spec.get('outputs')}")
    print("\n## input tensors:")
    for d in it.get_input_details():
        print(f"  {d['name']:40s} shape={tuple(d['shape'])} dtype={d['dtype'].__name__} "
              f"quant={d.get('quantization')}")
    print("\n## output tensors:")
    for d in it.get_output_details():
        print(f"  {d['name']:40s} shape={tuple(d['shape'])} dtype={d['dtype'].__name__} "
              f"quant={d.get('quantization')}")
    print("\n# Wire these signature/input/output names into cmd_generate() for Phase 1.")


def cmd_generate(args):
    # Greedy decode. The exact prefill/decode signature + tensor names are finalized from
    # `inspect` once the Phase-1 export exists; litert-torch Generative API convention is a
    # 'prefill' signature (tokens, input_pos) and a 'decode' signature (single token, kv-cache).
    import numpy as np  # noqa: F401  (used once the loop below is wired)

    ids = [int(x) for x in open(args.ids).read().split()]
    it = _load_interpreter(args.model)
    sigs = it.get_signature_list()
    if not sigs:
        sys.exit("model has no signatures; run `inspect` — a raw (non-Generative) export needs a "
                 "different driver loop.")
    sys.exit(
        "cmd_generate is a Phase-1 stub: run `parity.py inspect <model>` first to read the exact\n"
        "prefill/decode signature + KV-cache tensor names, then wire the greedy loop here to match\n"
        "ref_dump.py's output format. Blocked on the Phase-1 .tflite existing."
    )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("inspect"); pi.add_argument("model"); pi.set_defaults(fn=cmd_inspect)
    pg = sub.add_parser("generate")
    pg.add_argument("model"); pg.add_argument("--ids", required=True); pg.add_argument("-n", type=int, default=64)
    pg.set_defaults(fn=cmd_generate)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
