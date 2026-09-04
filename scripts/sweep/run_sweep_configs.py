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

def wake_and_keep_awake(adb_base):
    try:
        subprocess.run(adb_base + ["shell", "input keyevent KEYCODE_WAKEUP 2>/dev/null || true; svc power stayon true 2>/dev/null || true; settings put system screen_off_timeout 86400000 2>/dev/null || true"], capture_output=True, timeout=5)
    except Exception:
        pass

def adb_push_with_retry(adb_base, local_path, remote_path, max_retries=4):
    for attempt in range(1, max_retries + 1):
        res = subprocess.run(adb_base + ["push", local_path, remote_path], capture_output=True, text=True)
        if res.returncode == 0:
            return True
        print(f"⚠️ ADB push '{local_path}' -> '{remote_path}' failed (attempt {attempt}/{max_retries}): {res.stderr.strip()}")
        time.sleep(1.5)
        # Try reconnecting if wireless ADB target
        if "-s" in adb_base:
            s_idx = adb_base.index("-s") + 1
            serial = adb_base[s_idx]
            if ":" in serial:
                print(f"Reconnecting ADB to {serial}...")
                subprocess.run([adb_base[0], "disconnect", serial], capture_output=True)
                time.sleep(0.5)
                subprocess.run([adb_base[0], "connect", serial], capture_output=True)
                wake_and_keep_awake(adb_base)
                time.sleep(1.0)
    res.check_returncode()

def ensure_compiled_and_pushed(adb_base):
    wake_and_keep_awake(adb_base)
    arch = None
    for attempt in range(3):
        try:
            arch_res = subprocess.run(adb_base + ["shell", "uname -m"], capture_output=True, text=True, check=True)
            arch = arch_res.stdout.strip()
            if arch:
                break
        except Exception:
            if "-s" in adb_base:
                s_idx = adb_base.index("-s") + 1
                serial = adb_base[s_idx]
                if ":" in serial:
                    print(f"Connecting/Reconnecting ADB to {serial} (attempt {attempt+1}/3)...")
                    subprocess.run([adb_base[0], "disconnect", serial], capture_output=True)
                    time.sleep(0.5)
                    subprocess.run([adb_base[0], "connect", serial], capture_output=True)
                    wake_and_keep_awake(adb_base)
                    time.sleep(1.0)
            else:
                time.sleep(1.0)

    if not arch:
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

    print(f"Compiling runq_reallm for {arch} using {os.path.basename(compiler)} (OpenMP 4-core Batched GEMM + Cortex-A53 tuned)...")
    cmd = [compiler, "-O3", "-march=armv8-a", "-mcpu=cortex-a53", "-mfpu=neon-fp-armv8", "-ffast-math", "-fopenmp", "-static-openmp", "-Isrc", "-o", "runq_reallm_device", "src/runq_reallm.c", "-lm"]
    subprocess.run(cmd, check=True)
    
    if os.path.exists("src/power_sampler.c"):
        print("Compiling power_sampler for device...")
        cmd_ps = [compiler, "-O3", "-o", "power_sampler_device", "src/power_sampler.c"]
        subprocess.run(cmd_ps, check=True)
        adb_push_with_retry(adb_base, "power_sampler_device", "/data/local/tmp/power_sampler")
        subprocess.run(adb_base + ["shell", "chmod +x /data/local/tmp/power_sampler"], capture_output=True, check=True)
        if os.path.exists("power_sampler_device"):
            os.remove("power_sampler_device")

    print("Pushing runq_reallm engine and tokenizer to device...")
    adb_push_with_retry(adb_base, "runq_reallm_device", "/data/local/tmp/runq_reallm")
    subprocess.run(adb_base + ["shell", "chmod +x /data/local/tmp/runq_reallm"], capture_output=True, check=True)
    if os.path.exists("runq_reallm_device"):
        os.remove("runq_reallm_device")

    tok_path = find_tokenizer_gpt2()
    if tok_path:
        print(f"Pushing tokenizer ({tok_path}) -> /data/local/tmp/tokenizer_gpt2.bin...")
        adb_push_with_retry(adb_base, tok_path, "/data/local/tmp/tokenizer_gpt2.bin")
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
    adb_push_with_retry(adb_base, LOCAL_RLM, f"/data/local/tmp/{LOCAL_RLM}")
    
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
export OMP_NUM_THREADS=4

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

def get_device_telemetry(adb_base):
    cmd = """
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo 0;
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null || echo 0;
    cat /sys/class/thermal/thermal_zone17/temp 2>/dev/null || echo 0;
    cat /sys/class/thermal/thermal_zone6/temp 2>/dev/null || echo 0;
    cat /sys/class/thermal/cooling_device1/cur_state 2>/dev/null || cat /sys/class/thermal/cooling_device0/cur_state 2>/dev/null || echo 0;
    cat /sys/class/power_supply/battery/voltage_now 2>/dev/null || echo 0;
    cat /sys/class/power_supply/battery/current_now 2>/dev/null || echo 0;
    """
    try:
        res = subprocess.run(adb_base + ["shell", cmd], capture_output=True, text=True, timeout=10)
        lines = [l.strip() for l in res.stdout.strip().splitlines() if l.strip()]
        if len(lines) >= 7:
            freq_khz = int(lines[0]) if lines[0].isdigit() else 0
            max_freq_khz = int(lines[1]) if lines[1].isdigit() else 1900800
            cpu_temp = float(lines[2]) / 1000.0 if lines[2].isdigit() else 0.0
            skin_temp = float(lines[3]) / 1000.0 if lines[3].isdigit() else 0.0
            cdev = int(lines[4]) if lines[4].isdigit() else 0
            volt = float(lines[5]) / 1e6 if lines[5].isdigit() else 0.0
            curr = float(lines[6]) / 1000.0 if (lines[6].isdigit() or lines[6].startswith("-")) else 0.0
            return {
                "freq_khz": freq_khz,
                "freq_mhz": freq_khz / 1000.0,
                "max_freq_mhz": max_freq_khz / 1000.0,
                "cpu_temp_c": cpu_temp,
                "skin_temp_c": skin_temp,
                "cooling_state": cdev,
                "voltage_v": volt,
                "current_ma": curr
            }
    except Exception:
        pass
    return None

def wait_for_thermal_and_voltage_recovery(adb_base, target_cpu_temp=44.0, max_wait_sec=180.0, poll_interval=2.0, min_voltage_v=3.5):
    """
    Monitors device thermal, CPU frequency, and battery voltage.
    Prioritizes MAX CPU Frequency (1.90 GHz) and Cooling Device Level 0.
    As long as frequency is unclamped (1.90 GHz) and cooling state is 0,
    inference runs immediately without unnecessary thermal wait.
    """
    t0 = time.time()
    was_throttled = False

    while True:
        telem = get_device_telemetry(adb_base)
        if not telem:
            time.sleep(poll_interval)
            if time.time() - t0 > max_wait_sec:
                break
            continue

        cdev = telem["cooling_state"]
        freq_mhz = telem["freq_mhz"]
        cpu_temp = telem["cpu_temp_c"]
        volt = telem["voltage_v"]

        # Primary condition: CPU must be running at nominal max clock (1.90 GHz) with zero thermal mitigation
        is_freq_clamped = freq_mhz < 1850.0
        is_cdev_active = cdev > 0
        # Secondary safety ceiling (only if target_cpu_temp > 0)
        is_hot = (target_cpu_temp > 0.0) and (cpu_temp > target_cpu_temp)
        is_volt_sagging = volt > 0 and volt < min_voltage_v

        needs_cooldown = is_freq_clamped or is_cdev_active or is_hot or is_volt_sagging

        if not needs_cooldown:
            if was_throttled:
                time.sleep(1.0)
                telem_post = get_device_telemetry(adb_base) or telem
                print(f"\n  [Frequency Guard] ✅ Max frequency restored (1.90 GHz, CoolState: 0, CPU: {telem_post['cpu_temp_c']:.1f}°C, Volt: {telem_post['voltage_v']:.2f}V). Resuming sweep.")
            return telem

        was_throttled = True
        elapsed = time.time() - t0
        if elapsed > max_wait_sec:
            print(f"\n  [Frequency Guard] ⚠️ Wait timeout ({max_wait_sec}s). Resuming: CPU={cpu_temp:.1f}°C, Freq={freq_mhz:.0f}MHz, CoolState={cdev}, Volt={volt:.2f}V.")
            return telem

        reasons = []
        if is_freq_clamped: reasons.append(f"Freq={freq_mhz:.0f}MHz (clamped < 1900MHz)")
        if is_cdev_active: reasons.append(f"CoolState={cdev}")
        if is_hot: reasons.append(f"CPU={cpu_temp:.1f}°C > {target_cpu_temp:.1f}°C")
        if is_volt_sagging: reasons.append(f"Volt={volt:.2f}V < {min_voltage_v}V")
        reason_str = ", ".join(reasons)

        print(f"  [Frequency Guard] Throttling active ({reason_str})! Waiting for max frequency recovery... ({elapsed:.0f}s elapsed)", end="\r", flush=True)
        time.sleep(poll_interval)

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
    parser = argparse.ArgumentParser(description="Sweep LLM architecture configurations on physical Android device with Throttling & Thermal Recovery Protection.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to input CSV config file")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT_CSV, help="Path to output CSV results file")
    parser.add_argument("--serial", type=str, default=os.environ.get("ANDROID_SERIAL", ""), help="Device serial or IP:port")
    parser.add_argument("--adb", type=str, default=find_adb(), help="Path to adb binary")
    parser.add_argument("--prefill-tokens", type=int, default=48, help="Target prefill prompt tokens (default: 48)")
    parser.add_argument("--steps", type=int, default=16, help="Decoding steps per config (default: 16)")
    parser.add_argument("--sample-rate", type=float, default=10.0, help="Power sampling frequency in Hz (default: 10.0)")
    parser.add_argument("--inter-run-idle-sec", "--idle-between-runs", type=float, default=0.0, help="Optional idle battery sampling duration in seconds between config runs (default: 0.0, disabled)")
    parser.add_argument("--cool-down-temp", type=float, default=44.0, help="Target safety CPU temperature ceiling (°C). Set 0 to disable temp ceiling and rely purely on max clock (default: 44.0)")
    parser.add_argument("--min-cooldown-sec", type=float, default=2.0, help="Minimum pause between runs for voltage/thermal settling (default: 2.0s)")
    parser.add_argument("--max-cool-wait", type=float, default=180.0, help="Max seconds to wait for thermal/voltage recovery (default: 180.0)")
    parser.add_argument("--no-cooldown", action="store_true", help="Disable automatic thermal cooldown and voltage recovery wait")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found.")
        sys.exit(1)

    if not args.serial:
        res = subprocess.run([args.adb, "devices"], capture_output=True, text=True)
        active_devices = []
        offline_devices = []
        for line in res.stdout.strip().splitlines()[1:]:
            parts = line.strip().split()
            if len(parts) >= 2:
                if parts[1] == "device":
                    active_devices.append(parts[0])
                elif parts[1] == "offline":
                    offline_devices.append(parts[0])

        if not active_devices and offline_devices:
            for off in offline_devices:
                if ":" in off:
                    print(f"Device {off} is offline. Attempting automatic reconnection...")
                    subprocess.run([args.adb, "disconnect", off], capture_output=True)
                    time.sleep(0.5)
                    subprocess.run([args.adb, "connect", off], capture_output=True)
            time.sleep(1.0)
            res = subprocess.run([args.adb, "devices"], capture_output=True, text=True)
            for line in res.stdout.strip().splitlines()[1:]:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[1] == "device":
                    active_devices.append(parts[0])

        if active_devices:
            args.serial = active_devices[0]
        elif offline_devices:
            args.serial = offline_devices[0]

    adb_base = [args.adb]
    if args.serial:
        adb_base += ["-s", args.serial]

    print("==========================================================")
    print("Executing LLM Architecture Sweep on Physical Device")
    print(f"Config CSV   : {args.config}")
    print(f"Output CSV   : {args.out}")
    print(f"Device       : {args.serial if args.serial else 'default connected device'}")
    print(f"Prefill      : ~{args.prefill_tokens} prompt tokens")
    print(f"Decode       : {args.steps} generated tokens")
    print(f"Thermal Guard: Target CPU <= {args.cool_down_temp:.1f}°C (Cooldown protection: {'OFF' if args.no_cooldown else 'ON'})")
    if args.inter_run_idle_sec > 0:
        print(f"Inter-Idle   : {args.inter_run_idle_sec:.1f}s sampling between runs")
    print("==========================================================")

    arch = ensure_compiled_and_pushed(adb_base)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    csv_header_v2 = [
        "config_id", "category", "n_layer", "d_model", "n_h", "n_kv", "d_qk", "d_v", "d_mlp",
        "decode_tok_s", "ttft_ms", "tpot_ms", "duration_s", "baseline_power_w", "active_power_w", "peak_power_w",
        "dynamic_power_w", "total_energy_j", "dynamic_energy_per_token_mj",
        "temp_cpu_start_c", "temp_cpu_end_c", "cooling_state", "cpu_freq_mhz", "voltage_v", "notes"
    ]
    
    is_legacy_header = False
    if not os.path.exists(args.out):
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header_v2)
    else:
        with open(args.out, "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline()
            if "temp_cpu_start_c" not in first_line:
                is_legacy_header = True

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

        # 1. Thermal, Frequency & Voltage Recovery Guard
        if not args.no_cooldown:
            telem_start = wait_for_thermal_and_voltage_recovery(
                adb_base,
                target_cpu_temp=args.cool_down_temp,
                max_wait_sec=args.max_cool_wait
            )
        else:
            telem_start = get_device_telemetry(adb_base)

        if args.min_cooldown_sec > 0:
            time.sleep(args.min_cooldown_sec)

        if args.inter_run_idle_sec > 0:
            p_idle, i_idle, v_idle = measure_inter_run_idle_power(adb_base, duration_sec=args.inter_run_idle_sec, sample_rate=args.sample_rate)
            print(f"  [Inter-Run Idle] Battery power: {p_idle*1000.0:.1f} mW ({i_idle:.1f} mA @ {v_idle:.2f} V)")

        temp_start_c = telem_start["cpu_temp_c"] if telem_start else 0.0
        volt_start_v = telem_start["voltage_v"] if telem_start else 0.0
        print(f"\n[{idx}/{total_runs}] Profiling Config '{config_id}' ({category}) [CPU Temp: {temp_start_c:.1f}°C, Volt: {volt_start_v:.2f}V]...")
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

            # Sample ending telemetry
            telem_end = get_device_telemetry(adb_base)
            temp_end_c = telem_end["cpu_temp_c"] if telem_end else 0.0
            cooling_state_end = telem_end["cooling_state"] if telem_end else 0
            freq_end_mhz = telem_end["freq_mhz"] if telem_end else 0.0
            volt_end_v = telem_end["voltage_v"] if telem_end else 0.0

            with open(args.out, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if is_legacy_header:
                    writer.writerow([
                        config_id, category, n_layer, d_model, n_h, n_kv, d_qk, d_v, d_mlp,
                        f"{metrics['tok_s']:.4f}", f"{metrics['ttft_ms']:.2f}", f"{metrics['tpot_ms']:.2f}", f"{metrics['duration_s']:.4f}",
                        f"{metrics['baseline_power_w']:.4f}", f"{metrics['active_power_w']:.4f}",
                        f"{metrics['peak_power_w']:.4f}", f"{metrics['dynamic_power_w']:.4f}",
                        f"{metrics['total_energy_j']:.4f}", f"{metrics['dynamic_energy_per_token_mj']:.4f}",
                        notes
                    ])
                else:
                    writer.writerow([
                        config_id, category, n_layer, d_model, n_h, n_kv, d_qk, d_v, d_mlp,
                        f"{metrics['tok_s']:.4f}", f"{metrics['ttft_ms']:.2f}", f"{metrics['tpot_ms']:.2f}", f"{metrics['duration_s']:.4f}",
                        f"{metrics['baseline_power_w']:.4f}", f"{metrics['active_power_w']:.4f}",
                        f"{metrics['peak_power_w']:.4f}", f"{metrics['dynamic_power_w']:.4f}",
                        f"{metrics['total_energy_j']:.4f}", f"{metrics['dynamic_energy_per_token_mj']:.4f}",
                        f"{temp_start_c:.1f}", f"{temp_end_c:.1f}", cooling_state_end, f"{freq_end_mhz:.1f}", f"{volt_end_v:.3f}",
                        notes
                    ])

            print(f"  -> decode tok/s: {metrics['tok_s']:.2f} | TTFT: {metrics['ttft_ms']:.1f}ms | TPOT: {metrics['tpot_ms']:.2f}ms/tok | active power: {metrics['active_power_w']*1000:.1f}mW | CPU Temp: {temp_start_c:.1f}°C -> {temp_end_c:.1f}°C (CoolState: {cooling_state_end})")

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

