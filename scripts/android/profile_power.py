#!/usr/bin/env python3
import time
import subprocess
import argparse
import sys
import os

DEFAULT_ADB = os.environ.get("ADB", "/Users/xintingj/Library/Android/sdk/platform-tools/adb")

def read_battery_stats(adb_cmd, serial=None):
    """
    Reads current_now (uA) and voltage_now (uV) from Linux power supply sysfs.
    Supports standard battery and wear/watch power management ICs (e.g. sw5100_bms).
    """
    target = ["-s", serial] if serial else []
    sysfs_nodes = (
        "cat /sys/class/power_supply/battery/current_now /sys/class/power_supply/battery/voltage_now 2>/dev/null || "
        "cat /sys/class/power_supply/sw5100_bms/current_now /sys/class/power_supply/sw5100_bms/voltage_now 2>/dev/null"
    )
    cmd = [adb_cmd] + target + ["shell", sysfs_nodes]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = res.stdout.strip().split()
        if len(lines) >= 2:
            current_ua = float(lines[0])
            voltage_uv = float(lines[1])
            power_w = (abs(current_ua) * voltage_uv) / 1e12
            return power_w
    except Exception:
        pass
    return None

def main():
    parser = argparse.ArgumentParser(description="Profile real-time power draw and energy consumption of LLM on Android / Pixel Watch.")
    parser.add_argument("--engine", type=str, default="runq_tinystories_android", help="Name of executable binary in /data/local/tmp")
    parser.add_argument("--model", type=str, default="stories15M.q8.bin", help="Model filename in /data/local/tmp")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.bin", help="Tokenizer filename in /data/local/tmp")
    parser.add_argument("--tok-flag", type=str, default="-z", choices=["-z", "-g"], help="-z for sentencepiece/llama tokenizer, -g for GPT-2 BPE")
    parser.add_argument("--prompt", type=str, default="Once upon a time, a little girl named Lily", help="Generation prompt")
    parser.add_argument("--steps", type=int, default=64, help="Number of decoding steps")
    parser.add_argument("--serial", type=str, default=os.environ.get("ANDROID_SERIAL", ""), help="Device serial or IP:port")
    parser.add_argument("--adb", type=str, default=DEFAULT_ADB, help="Path to adb binary")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--topp", type=float, default=0.9, help="Top-p sampling")
    args = parser.parse_args()

    adb = args.adb
    serial = args.serial or None

    adb_base = [adb]
    if serial:
        adb_base += ["-s", serial]

    print("==========================================================")
    print("Starting On-Device Power & Throughput Profiling")
    print(f"Model       : {args.model}")
    print(f"Engine      : {args.engine}")
    print(f"Steps       : {args.steps}")
    print(f"Device      : {serial if serial else 'default connected device'}")
    print("==========================================================")

    run_cmd = adb_base + [
        "shell",
        f"cd /data/local/tmp && ./{args.engine} {args.model} {args.tok_flag} {args.tokenizer} -i '{args.prompt}' -t {args.temp} -p {args.topp} -n {args.steps}"
    ]

    start_time = time.time()
    proc = subprocess.Popen(run_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="ignore")

    power_samples = []
    timestamps = []

    # Poll battery power sensors at ~20Hz (every 50ms) while inference is running
    while proc.poll() is None:
        p = read_battery_stats(adb, serial)
        if p is not None:
            power_samples.append(p)
            timestamps.append(time.time() - start_time)
        time.sleep(0.05)

    stdout, stderr = proc.communicate()
    duration = time.time() - start_time

    # Parse tok/s from stderr or stdout
    tok_s = 0.0
    for line in (stderr + "\n" + stdout).splitlines():
        if "achieved tok/s:" in line:
            try:
                tok_s = float(line.split(":")[-1].strip())
            except ValueError:
                pass
            break

    print("\n--- MODEL OUTPUT ---")
    print(stdout.strip())
    print("--------------------\n")

    if not power_samples:
        print("Note: Could not read /sys/class/power_supply/ battery sensors on device (or running on emulator).")
        print(f"Inference Duration : {duration:.2f} s")
        if tok_s > 0:
            print(f"Throughput         : {tok_s:.2f} tok/s")
        return

    peak_power = max(power_samples)
    mean_power = sum(power_samples) / len(power_samples)

    # Numerical integration (trapezoidal rule)
    total_energy_j = 0.0
    for i in range(1, len(power_samples)):
        dt = timestamps[i] - timestamps[i-1]
        total_energy_j += 0.5 * (power_samples[i] + power_samples[i-1]) * dt

    energy_per_token_j = total_energy_j / args.steps if args.steps > 0 else 0.0

    print("=== POWER & ENERGY PROFILE ===")
    print(f"Inference Duration : {duration:.2f} seconds")
    if tok_s > 0:
        print(f"Throughput         : {tok_s:.2f} tok/s")
    print(f"Total Steps        : {args.steps}")
    print(f"Power Samples      : {len(power_samples)} samples (~20Hz)")
    print(f"Peak Power Draw    : {peak_power:.3f} W")
    print(f"Average Power Draw : {mean_power:.3f} W")
    print(f"Total Energy Spent : {total_energy_j:.3f} Joules")
    print(f"Energy per Token   : {energy_per_token_j * 1000:.3f} mJ/tok")
    print("==============================\n")

if __name__ == "__main__":
    main()
