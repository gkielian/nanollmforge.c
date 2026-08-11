#!/usr/bin/env python3
import os
import sys
import csv
import json
import time
import shutil
import argparse
import subprocess
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, "scripts", "sweep", "configs", "watch_anchor_part.csv")
DEFAULT_OUT_CSV = os.path.join(REPO_ROOT, "scripts", "sweep", "outputs", "sweep_anchor_results.csv")
LOCAL_CKPT = "temp_device_ckpt.pt"
LOCAL_RLM = "temp_device_model.q8.rlm"

def find_adb():
    if os.environ.get("ADB"):
        return os.environ.get("ADB")
    which_adb = shutil.which("adb")
    if which_adb:
        return which_adb
    home_sdk_adb = os.path.expanduser("~/Library/Android/sdk/platform-tools/adb")
    if os.path.isfile(home_sdk_adb):
        return home_sdk_adb
    for env_var in ["ANDROID_HOME", "ANDROID_SDK_ROOT"]:
        sdk = os.environ.get(env_var)
        if sdk and os.path.isfile(os.path.join(sdk, "platform-tools", "adb")):
            return os.path.join(sdk, "platform-tools", "adb")
    return "adb"

def find_ndk_compiler(arch):
    candidates = []
    if os.environ.get("NDK"):
        candidates.append(os.environ.get("NDK"))
    if os.environ.get("ANDROID_NDK_HOME"):
        candidates.append(os.environ.get("ANDROID_NDK_HOME"))
    
    home_ndk_dir = os.path.expanduser("~/Library/Android/sdk/ndk")
    if os.path.isdir(home_ndk_dir):
        import glob
        versions = sorted(glob.glob(os.path.join(home_ndk_dir, "*")), reverse=True)
        candidates.extend(versions)

    for ndk_root in candidates:
        bin_dir = os.path.join(ndk_root, "toolchains", "llvm", "prebuilt", "darwin-x86_64", "bin")
        if not os.path.isdir(bin_dir):
            bin_dir = os.path.join(ndk_root, "toolchains", "llvm", "prebuilt", "linux-x86_64", "bin")
        if os.path.isdir(bin_dir):
            if "armv7" in arch or "armv8l" in arch or "32" in arch:
                comp = os.path.join(bin_dir, "armv7a-linux-androideabi24-clang")
            else:
                comp = os.path.join(bin_dir, "aarch64-linux-android24-clang")
            if os.path.isfile(comp):
                return comp
    return None

def find_tokenizer_gpt2():
    candidates = [
        "tokenizer_gpt2.bin",
        "models/nsga_best3_rotary_periln_105M/tokenizer_gpt2.bin",
        "models/smollm2_135M/tokenizer_gpt2.bin"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def ensure_compiled_and_pushed(adb_base):
    try:
        arch_res = subprocess.run(adb_base + ["shell", "uname -m"], capture_output=True, text=True, check=True)
        arch = arch_res.stdout.strip()
    except Exception:
        print("\n" + "="*60)
        print("❌ Error: No Android device connected over ADB.")
        print("Please connect your device via USB or Wi-Fi ADB:")
        print("  adb connect <device_ip>:5555")
        print("  adb devices")
        print("="*60 + "\n")
        sys.exit(1)

    compiler = find_ndk_compiler(arch)
    if not compiler:
        print(f"Error: NDK compiler not found for {arch}. Please set NDK or ANDROID_NDK_HOME environment variable.")
        sys.exit(1)

    print(f"Compiling runq_reallm for {arch} using {os.path.basename(compiler)}...")
    cmd = [compiler, "-O3", "-Isrc", "-o", "runq_reallm_device", "src/runq_reallm.c", "-lm"]
    subprocess.run(cmd, check=True)
    
    if os.path.exists("src/power_sampler.c"):
        print("Compiling power_sampler for device...")
        cmd_ps = [compiler, "-O3", "-o", "power_sampler_device", "src/power_sampler.c"]
        subprocess.run(cmd_ps, check=True)
        subprocess.run(adb_base + ["push", "power_sampler_device", "/data/local/tmp/power_sampler"], capture_output=True, check=True)
        subprocess.run(adb_base + ["shell", "chmod +x /data/local/tmp/power_sampler"], capture_output=True, check=True)
        if os.path.exists("power_sampler_device"):
            os.remove("power_sampler_device")

    print("Pushing runq_reallm engine and tokenizer to device...")
    subprocess.run(adb_base + ["push", "runq_reallm_device", "/data/local/tmp/runq_reallm"], check=True)
    subprocess.run(adb_base + ["shell", "chmod +x /data/local/tmp/runq_reallm"], check=True)
    if os.path.exists("runq_reallm_device"):
        os.remove("runq_reallm_device")

    tok_path = find_tokenizer_gpt2()
    if tok_path:
        print(f"Pushing tokenizer ({tok_path}) -> /data/local/tmp/tokenizer_gpt2.bin...")
        subprocess.run(adb_base + ["push", tok_path, "/data/local/tmp/tokenizer_gpt2.bin"], capture_output=True, check=True)
    else:
        print("Warning: tokenizer_gpt2.bin not found locally! Ensure it exists on device.")

    return arch

def generate_mock_ckpt(n_head, n_kv, qk, vd, mlp_hidden, n_layer, n_embd):
    vocab_size = 50257
    block_size = 256
    
    ma = {
        "n_layer": n_layer,
        "n_embd": n_embd,
        "vocab_size": vocab_size,
        "block_size": block_size,
        "n_head_layerlist": [n_head] * n_layer,
        "n_kv_group_layerlist": [n_kv] * n_layer,
        "attention_variant_layerlist": ["infinite"] * n_layer,
        "mlp_variant": "swiglu",
        "activation_variant": "gelu",
        "norm_variant_attn": "rmsnorm",
        "norm_variant_output": "rmsnorm",
        "use_rotary_embeddings": True,
        "bias": False,
        "use_peri_ln": True,
        "use_pre_ln": True,
        "use_concat_heads": True,
    }
    
    sd = {}
    sd["transformer.wte.weight"] = torch.randn(vocab_size, n_embd)
    sd["transformer.ln_f.gain"] = torch.randn(n_embd)
    
    for i in range(n_layer):
        p = f"transformer.h.{i}."
        sd[p + "pre_ln_attn.gain"] = torch.randn(n_embd)
        sd[p + "peri_ln_attn.gain"] = torch.randn(n_embd)
        sd[p + "pre_ln_mlp.gain"] = torch.randn(n_embd)
        sd[p + "peri_ln_mlp.gain"] = torch.randn(n_embd)
        sd[p + "mlp.c_fc_in1.weight"] = torch.randn(mlp_hidden, n_embd)
        sd[p + "mlp.c_fc_in2.weight"] = torch.randn(mlp_hidden, n_embd)
        sd[p + "mlp.c_fc_out.weight"] = torch.randn(n_embd, mlp_hidden)
        sd[p + "attn.c_attn_q.weight"] = torch.randn(n_head * qk, n_embd)
        sd[p + "attn.c_attn_k.weight"] = torch.randn(n_kv * qk, n_embd)
        sd[p + "attn.c_attn_v.weight"] = torch.randn(n_kv * vd, n_embd)
        sd[p + "attn.c_proj.weight"] = torch.randn(n_embd, n_head * vd)
        
    torch.save({"model": sd, "model_args": ma}, LOCAL_CKPT)

def export_model():
    export_script = os.path.join(REPO_ROOT, "reallmforge", "export_reallm_hetero.py")
    cmd = ["python3", export_script, LOCAL_CKPT, LOCAL_RLM, "--version", "2"]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def build_prompt_for_tokens(target_prefill_tokens):
    base_corpus = (
        "Once upon a time in a sunny meadow near a quiet village, there lived a friendly little girl named Lily. "
        "Every afternoon, Lily loved to walk along the stone path to observe the birds singing in the tall trees. "
        "One day, she noticed a fluffy gray kitten sitting beside a wooden bench. "
        "The kitten was curious and gently rubbed against her shoes with a soft purr. "
        "Lily smiled warmly and brought a bowl of fresh milk to welcome her new little friend. "
        "From that day on, the two companions explored the forest trails together, sharing joy and laughter every day. "
        "Suddenly, they heard a strange rustling sound coming from behind the bushes, and when Lily looked closely, she saw a hidden golden box tucked under the roots of a giant tree. "
        "Curious about what was inside, Lily and the kitten decided to open the box together, and to their surprise, they found a map leading to"
    )
    words = base_corpus.split()
    target_words = max(1, int(round(target_prefill_tokens / 1.3)))
    repeated_words = (words * ((target_words // len(words)) + 2))[:target_words]
    prompt_str = " ".join(repeated_words)
    if not prompt_str.endswith("because") and not prompt_str.endswith("to"):
        prompt_str += " and then they decided to explore further because"
    return prompt_str



def run_benchmark_with_power(adb_base, prefill_tokens=128, decode_steps=64, sample_rate=10.0, pre_idle_sec=4.0, post_idle_sec=4.0):
    subprocess.run(adb_base + ["push", LOCAL_RLM, f"/data/local/tmp/{LOCAL_RLM}"], capture_output=True, check=True)
    
    prompt_str = build_prompt_for_tokens(prefill_tokens)
    total_steps = prefill_tokens + decode_steps
    interval_ms = max(5, int((1.0 / sample_rate) * 1000))
    total_est_sec = pre_idle_sec + (total_steps / 5.0) + post_idle_sec + 60.0

    remote_script = f"""#!/bin/sh
set -e
cd /data/local/tmp

input keyevent KEYCODE_WAKEUP 2>/dev/null || true
svc power stayon true 2>/dev/null || true
settings put system screen_off_timeout 86400000 2>/dev/null || true

sleep 0.8

if [ -f /sys/class/power_supply/battery/current_now ]; then
    NODE="/sys/class/power_supply/battery"
elif [ -f /sys/class/power_supply/sw5100_bms/current_now ]; then
    NODE="/sys/class/power_supply/sw5100_bms"
else
    echo "NO_NODE"
    exit 1
fi

CSV="/data/local/tmp/trace_raw.csv"
LOG="/data/local/tmp/infer_output.log"
rm -f "$CSV" "$LOG"

./power_sampler "$NODE" {total_est_sec:.1f} {interval_ms} "$CSV" &
SAMPLER_PID=$!

sleep {pre_idle_sec}

./runq_reallm {LOCAL_RLM} -g tokenizer_gpt2.bin -i '{prompt_str}' -t 0.8 -p 0.9 -n {total_steps} > "$LOG" 2>&1 || true

sleep {post_idle_sec}

kill -15 "$SAMPLER_PID" 2>/dev/null || true
wait "$SAMPLER_PID" 2>/dev/null || true

echo "DONE"
"""
    push_proc = subprocess.Popen(adb_base + ["shell", "cat > /data/local/tmp/run_trace.sh && chmod +x /data/local/tmp/run_trace.sh"],
                                 stdin=subprocess.PIPE, text=True)
    push_proc.communicate(input=remote_script)

    subprocess.run(adb_base + ["shell", "sh /data/local/tmp/run_trace.sh"], check=True)

    subprocess.run(adb_base + ["pull", "/data/local/tmp/trace_raw.csv", "temp_trace_raw.csv"], capture_output=True, check=True)
    subprocess.run(adb_base + ["pull", "/data/local/tmp/infer_output.log", "temp_infer_log.txt"], capture_output=True, check=True)
    subprocess.run(adb_base + ["shell", f"rm -f /data/local/tmp/{LOCAL_RLM} /data/local/tmp/trace_raw.csv /data/local/tmp/infer_output.log /data/local/tmp/run_trace.sh"], capture_output=True)

    with open("temp_infer_log.txt", "r", encoding="utf-8", errors="replace") as f:
        infer_output = f.read()

    timestamps, power_vals = [], []
    with open("temp_trace_raw.csv", "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                try:
                    t = float(row[0])
                    i_ua = float(row[1])
                    v_uv = float(row[2])
                    timestamps.append(t)
                    power_vals.append((abs(i_ua) * v_uv) / 1e12)
                except ValueError:
                    pass

    for tmp in ["temp_trace_raw.csv", "temp_infer_log.txt"]:
        if os.path.exists(tmp): os.remove(tmp)

    prefill_tokens_parsed = 0
    decode_tokens_parsed = 0
    prefill_ms = 0.0
    ttft_ms = 0.0
    decode_tok_s = 0.0
    total_tok_s = 0.0

    for line in infer_output.splitlines():
        if "prefill_ms:" in line:
            parts = line.split(",")
            for p in parts:
                kv = p.strip().split(":")
                if len(kv) == 2:
                    k, v = kv[0].strip(), kv[1].strip()
                    try:
                        if k == "prefill_tokens": prefill_tokens_parsed = int(v)
                        elif k == "prefill_ms": prefill_ms = float(v)
                        elif k == "ttft_ms": ttft_ms = float(v)
                        elif k == "decode_tokens": decode_tokens_parsed = int(v)
                        elif k == "decode_tok_s": decode_tok_s = float(v)
                        elif k == "achieved tok/s": total_tok_s = float(v)
                    except ValueError:
                        pass
        elif "achieved tok/s:" in line and total_tok_s == 0.0:
            try:
                total_tok_s = float(line.split(":")[-1].strip())
            except ValueError:
                pass

    tok_s = decode_tok_s if decode_tok_s > 0 else total_tok_s
    tpot_ms = (1000.0 / tok_s) if tok_s > 0 else 0.0

    t_infer_start = pre_idle_sec
    infer_duration = (prefill_ms / 1000.0) + (decode_tokens_parsed / tok_s) if (prefill_ms > 0 and tok_s > 0) else (timestamps[-1] - pre_idle_sec - post_idle_sec)
    t_infer_end = t_infer_start + infer_duration

    # Use median of the steady pre-idle window (excluding first 1s for stabilization)
    pre_powers = [p for t, p in zip(timestamps, power_vals) if 1.0 <= t < t_infer_start]
    active_powers = [p for t, p in zip(timestamps, power_vals) if t_infer_start <= t <= t_infer_end]

    mean_baseline_w = float(np.median(pre_powers)) if pre_powers else 0.0
    mean_active_w = float(np.mean(active_powers)) if active_powers else 0.0
    peak_active_w = float(np.max(active_powers)) if active_powers else 0.0
    dynamic_power_w = max(0.0, mean_active_w - mean_baseline_w)

    total_energy_j = mean_active_w * infer_duration
    dynamic_energy_j = dynamic_power_w * infer_duration
    dynamic_energy_per_tok_mj = (dynamic_energy_j / decode_steps) * 1000.0 if decode_steps > 0 else 0.0

    return {
        "tok_s": tok_s,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "duration_s": infer_duration,
        "baseline_power_w": mean_baseline_w,
        "active_power_w": mean_active_w,
        "peak_power_w": peak_active_w,
        "dynamic_power_w": dynamic_power_w,
        "total_energy_j": total_energy_j,
        "dynamic_energy_per_token_mj": dynamic_energy_per_tok_mj
    }

def get_existing_completed_config_ids(out_csv):
    completed = set()
    if os.path.exists(out_csv):
        with open(out_csv, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row:
                    completed.add(row[0].strip())
    return completed

def main():
    parser = argparse.ArgumentParser(description="Sweep LLM architecture configurations on physical Android device from CSV input.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to input CSV config file")
def measure_inter_run_idle_power(adb_base, duration_sec=3.0, sample_rate=10.0):
    interval_ms = max(5, int((1.0 / sample_rate) * 1000))
    remote_script = f"""#!/bin/sh
set -e
cd /data/local/tmp

input keyevent KEYCODE_WAKEUP 2>/dev/null || true
svc power stayon true 2>/dev/null || true

if [ -f /sys/class/power_supply/battery/current_now ]; then
    NODE="/sys/class/power_supply/battery"
elif [ -f /sys/class/power_supply/sw5100_bms/current_now ]; then
    NODE="/sys/class/power_supply/sw5100_bms"
else
    echo "NO_NODE"
    exit 1
fi

CSV="/data/local/tmp/inter_idle.csv"
rm -f "$CSV"

./power_sampler "$NODE" {duration_sec:.1f} {interval_ms} "$CSV" &
SAMPLER_PID=$!

sleep {duration_sec:.1f}

kill -15 "$SAMPLER_PID" 2>/dev/null || true
wait "$SAMPLER_PID" 2>/dev/null || true
echo "DONE"
"""
    push_proc = subprocess.Popen(adb_base + ["shell", "cat > /data/local/tmp/run_inter_idle.sh && chmod +x /data/local/tmp/run_inter_idle.sh"],
                                 stdin=subprocess.PIPE, text=True)
    push_proc.communicate(input=remote_script)

    subprocess.run(adb_base + ["shell", "sh /data/local/tmp/run_inter_idle.sh"], capture_output=True, check=True)
    subprocess.run(adb_base + ["pull", "/data/local/tmp/inter_idle.csv", "temp_inter_idle.csv"], capture_output=True, check=True)
    subprocess.run(adb_base + ["shell", "rm -f /data/local/tmp/inter_idle.csv /data/local/tmp/run_inter_idle.sh"], capture_output=True)

    power_vals, curr_vals, volt_vals = [], [], []
    with open("temp_inter_idle.csv", "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                try:
                    i_ua = float(row[1])
                    v_uv = float(row[2])
                    p_w = (abs(i_ua) * v_uv) / 1e12
                    power_vals.append(p_w)
                    curr_vals.append(abs(i_ua) / 1000.0)
                    volt_vals.append(v_uv / 1e6)
                except ValueError:
                    pass

    if os.path.exists("temp_inter_idle.csv"):
        os.remove("temp_inter_idle.csv")

    if not power_vals:
        return 0.0, 0.0, 0.0

    mean_p = float(np.median(power_vals))
    mean_i = float(np.median(curr_vals))
    mean_v = float(np.median(volt_vals))
    return mean_p, mean_i, mean_v

def main():
    parser = argparse.ArgumentParser(description="Sweep LLM architecture configurations on physical Android device from CSV input.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to input CSV config file")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT_CSV, help="Path to output CSV results file")
    parser.add_argument("--serial", type=str, default=os.environ.get("ANDROID_SERIAL", ""), help="Device serial or IP:port")
    parser.add_argument("--adb", type=str, default=find_adb(), help="Path to adb binary")
    parser.add_argument("--prefill-tokens", type=int, default=128, help="Target prefill prompt tokens (default: 128)")
    parser.add_argument("--steps", type=int, default=64, help="Decoding steps per config (default: 64)")
    parser.add_argument("--sample-rate", type=float, default=10.0, help="Power sampling frequency in Hz (default: 10.0)")
    parser.add_argument("--inter-run-idle-sec", "--idle-between-runs", type=float, default=0.0, help="Optional idle battery sampling duration in seconds between config runs (default: 0.0, disabled)")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found.")
        sys.exit(1)

    adb_base = [args.adb]
    if args.serial:
        adb_base += ["-s", args.serial]

    print("==========================================================")
    print("Executing LLM Architecture Sweep on Physical Device")
    print(f"Config CSV : {args.config}")
    print(f"Output CSV : {args.out}")
    print(f"Device     : {args.serial if args.serial else 'default connected device'}")
    print(f"Prefill    : ~{args.prefill_tokens} prompt tokens")
    print(f"Decode     : {args.steps} generated tokens")
    if args.inter_run_idle_sec > 0:
        print(f"Inter-Idle : {args.inter_run_idle_sec:.1f}s sampling between runs")
    print("==========================================================")

    arch = ensure_compiled_and_pushed(adb_base)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if not os.path.exists(args.out):
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "config_id", "category", "n_layer", "d_model", "n_h", "n_kv", "d_qk", "d_v", "d_mlp",
                "decode_tok_s", "ttft_ms", "tpot_ms", "duration_s", "baseline_power_w", "active_power_w", "peak_power_w",
                "dynamic_power_w", "total_energy_j", "dynamic_energy_per_token_mj", "notes"
            ])

    completed_ids = get_existing_completed_config_ids(args.out)

    configs = []
    with open(args.config, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
        f.seek(0)
        delimiter = "\t" if "\t" in first_line else ","
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            configs.append(row)

    total_runs = len(configs)
    print(f"Loaded {total_runs} sweep configurations from {args.config} (Already completed: {len(completed_ids)})")

    for idx, cfg in enumerate(configs, 1):
        config_id = cfg["config_id"].strip()
        category = cfg.get("category", "").strip()
        n_layer = int(cfg["n_layer"])
        d_model = int(cfg["d_model"])
        n_h = int(cfg["n_h"])
        n_kv = int(cfg["n_kv"])
        d_qk = int(cfg["d_qk"])
        d_v = int(cfg["d_v"])
        d_mlp = int(cfg["d_mlp"])
        notes = cfg.get("notes", "").strip()

        if config_id in completed_ids:
            print(f"[{idx}/{total_runs}] Skipping completed: {config_id}")
            continue

        if args.inter_run_idle_sec > 0:
            p_idle, i_idle, v_idle = measure_inter_run_idle_power(adb_base, duration_sec=args.inter_run_idle_sec, sample_rate=args.sample_rate)
            print(f"  [Inter-Run Idle] Battery power: {p_idle*1000.0:.1f} mW ({i_idle:.1f} mA @ {v_idle:.2f} V)")

        print(f"\n[{idx}/{total_runs}] Profiling Config '{config_id}' ({category})...")
        print(f"  Specs: n_layer={n_layer}, d_model={d_model}, n_h={n_h}, n_kv={n_kv}, d_qk={d_qk}, d_v={d_v}, d_mlp={d_mlp}")

        try:
            generate_mock_ckpt(n_h, n_kv, d_qk, d_v, d_mlp, n_layer, d_model)
            export_model()

            metrics = run_benchmark_with_power(
                adb_base,
                prefill_tokens=args.prefill_tokens,
                decode_steps=args.steps,
                sample_rate=args.sample_rate
            )

            with open(args.out, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    config_id, category, n_layer, d_model, n_h, n_kv, d_qk, d_v, d_mlp,
                    f"{metrics['tok_s']:.4f}", f"{metrics['ttft_ms']:.2f}", f"{metrics['tpot_ms']:.2f}", f"{metrics['duration_s']:.4f}",
                    f"{metrics['baseline_power_w']:.4f}", f"{metrics['active_power_w']:.4f}",
                    f"{metrics['peak_power_w']:.4f}", f"{metrics['dynamic_power_w']:.4f}",
                    f"{metrics['total_energy_j']:.4f}", f"{metrics['dynamic_energy_per_token_mj']:.4f}",
                    notes
                ])

            print(f"  -> decode tok/s: {metrics['tok_s']:.2f} | TTFT: {metrics['ttft_ms']:.1f}ms | TPOT: {metrics['tpot_ms']:.2f}ms/tok | active power: {metrics['active_power_w']*1000:.1f}mW | dynamic energy/tok: {metrics['dynamic_energy_per_token_mj']:.1f} mJ/tok")

        except Exception as e:
            print(f"  -> Error profiling config {config_id}: {e}")
        finally:
            if os.path.exists(LOCAL_CKPT):
                os.remove(LOCAL_CKPT)
            if os.path.exists(LOCAL_RLM):
                os.remove(LOCAL_RLM)

    print(f"\n==========================================================")
    print(f"Sweep complete! Results saved to: {args.out}")
    print("==========================================================")

if __name__ == "__main__":
    main()
