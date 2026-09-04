#!/usr/bin/env python3
"""
Targeted Search Space Generator (1,000 Configurations) for Pixel Watch 5.
Designed to systematically evaluate specific hardware hypotheses and serve as training data
for learning an analytical/surrogate Hardware Performance & Energy Model.

Composition:
1. Suite A (L2/SLC Cache Residency Inflection): 120 samples
2. Suite B (Iso-Capacity Width vs. Depth Pareto Fronts): 180 samples
3. Suite C (Attention Head Asymmetry & KV-Sharing MQA/GQA/MHA): 150 samples
4. Suite D (MLP Expansion Ratio & FFN Geometry): 100 samples
5. Suite E (Latin Hypercube Global Uniform Exploration 1M - 80M): 450 samples
Total: Exactly 1,000 unique architectural configurations.
"""

import os
import math
import random
import numpy as np
import pandas as pd

VOCAB_SIZE = 50257
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calc_model_specs(n_layer, d_model, n_h, n_kv, d_qk, d_v, d_mlp, vocab_size=VOCAB_SIZE):
    embed_params = vocab_size * d_model
    attn_params = (d_model * n_h * d_qk) + (d_model * n_kv * d_qk) + (d_model * n_kv * d_v) + (n_h * d_v * d_model)
    mlp_params = 3 * d_model * d_mlp
    layer_params = attn_params + mlp_params
    total_params = embed_params + n_layer * layer_params
    
    layer_kb = (layer_params * 1.25) / 1024.0
    total_mb = (total_params * 1.25) / (1024.0 * 1024.0)
    return total_params, layer_params, total_mb, layer_kb

def generate_suite_a_cache_inflection():
    """Suite A: Fine-grained stepping across layer working sets (100 KB to 2000 KB)."""
    configs = []
    # Grid of widths and fine-grained MLP sizes
    for d in [96, 128, 160, 192, 224, 256, 320, 384]:
        for L in [4, 8]:
            nh = max(1, d // 32)
            nkv = 1
            dqk = 32
            dv = 32
            # 8 fine steps of d_mlp per (d, L) pair to cross cache boundaries
            mlp_multipliers = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
            for m_mul in mlp_multipliers:
                dmlp = int(round((d * m_mul) / 32.0) * 32)
                tot, l_param, tot_mb, l_kb = calc_model_specs(L, d, nh, nkv, dqk, dv, dmlp)
                configs.append({
                    "config_id": f"S1_CACHE_D{d}_L{L}_MLP{dmlp}",
                    "suite": "Hypothesis_1_Cache_Inflection",
                    "n_layer": L,
                    "d_model": d,
                    "n_h": nh,
                    "n_kv": nkv,
                    "d_qk": dqk,
                    "d_v": dv,
                    "d_mlp": dmlp,
                    "total_params_M": round(tot / 1e6, 3),
                    "layer_size_kb": round(l_kb, 1),
                    "mlp_ratio": round(dmlp / d, 2),
                    "kv_ratio": round(nkv / nh, 3)
                })
    return configs[:120]

def generate_suite_b_iso_capacity_width_depth():
    """Suite B: Iso-parameter slices (4M, 8M, 15M, 25M, 40M, 60M) exploring width vs. depth."""
    configs = []
    target_params = [4.0, 8.0, 15.0, 25.0, 40.0, 60.0] # in Millions
    
    for p_tgt in target_params:
        # Vary depth from shallow (2) to deep (28)
        for L in [2, 3, 4, 6, 8, 10, 12, 16, 20, 24]:
            # Solve for approximate d_model
            # Param ≈ 50257 * d + L * (4 * d^2 + 3 * d * (2.67 * d)) = 50257 * d + L * 12 * d^2
            # 12 * L * d^2 + 50257 * d - Target = 0
            a = 12 * L
            b = VOCAB_SIZE
            c = - (p_tgt * 1e6)
            disc = b**2 - 4 * a * c
            if disc < 0: continue
            d_exact = (-b + math.sqrt(disc)) / (2 * a)
            d = int(max(64, round(d_exact / 32.0) * 32))
            
            # Vary attention head sizes & mlp ratios around this point
            for mlp_r in [2.5, 3.5, 4.0]:
                dmlp = int(round((d * mlp_r) / 32.0) * 32)
                nh = max(2, d // 32)
                for nkv in [1, max(1, nh // 2)]:
                    tot, l_param, tot_mb, l_kb = calc_model_specs(L, d, nh, nkv, 32, 32, dmlp)
                    configs.append({
                        "config_id": f"S2_ISOP_T{int(p_tgt)}M_L{L}_D{d}_KV{nkv}_R{mlp_r}",
                        "suite": "Hypothesis_2_IsoParam_Width_Depth",
                        "n_layer": L,
                        "d_model": d,
                        "n_h": nh,
                        "n_kv": nkv,
                        "d_qk": 32,
                        "d_v": 32,
                        "d_mlp": dmlp,
                        "total_params_M": round(tot / 1e6, 3),
                        "layer_size_kb": round(l_kb, 1),
                        "mlp_ratio": round(dmlp / d, 2),
                        "kv_ratio": round(nkv / nh, 3)
                    })
    # Subsample or cap to exactly 180
    random.seed(42)
    random.shuffle(configs)
    return configs[:180]

def generate_suite_c_attention_geometry():
    """Suite C: Multi-Query, Grouped-Query vs Multi-Head & Asymmetric d_qk vs d_v."""
    configs = []
    for d in [128, 192, 256, 384, 512]:
        for L in [6, 12]:
            dmlp = int(round(d * 2.67 / 32.0) * 32)
            nh_candidates = [4, 8, 12, 16] if d >= 256 else [2, 4, 8]
            for nh in nh_candidates:
                # KV sharing ratios
                valid_n_kvs = sorted(list(set([1, max(1, nh // 4), max(1, nh // 2), nh])))
                for nkv in valid_n_kvs:
                    # Asymmetric head dimensions
                    for (dqk, dv) in [(32, 32), (64, 32), (32, 64), (64, 64)]:
                        tot, l_param, tot_mb, l_kb = calc_model_specs(L, d, nh, nkv, dqk, dv, dmlp)
                        configs.append({
                            "config_id": f"S3_ATTN_D{d}_L{L}_H{nh}_KV{nkv}_Q{dqk}_V{dv}",
                            "suite": "Hypothesis_3_Attention_Geometry",
                            "n_layer": L,
                            "d_model": d,
                            "n_h": nh,
                            "n_kv": nkv,
                            "d_qk": dqk,
                            "d_v": dv,
                            "d_mlp": dmlp,
                            "total_params_M": round(tot / 1e6, 3),
                            "layer_size_kb": round(l_kb, 1),
                            "mlp_ratio": round(dmlp / d, 2),
                            "kv_ratio": round(nkv / nh, 3)
                        })
    random.seed(42)
    random.shuffle(configs)
    return configs[:150]

def generate_suite_d_mlp_expansion():
    """Suite D: Isolating FFN compute intensity by sweeping expansion ratio from 1.0x to 6.0x."""
    configs = []
    for d in [128, 192, 256, 384, 512]:
        for L in [4, 8, 12, 16]:
            nh = max(2, d // 32)
            nkv = max(1, nh // 2)
            for r in [1.0, 1.5, 2.0, 2.67, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]:
                dmlp = int(round((d * r) / 32.0) * 32)
                tot, l_param, tot_mb, l_kb = calc_model_specs(L, d, nh, nkv, 32, 32, dmlp)
                configs.append({
                    "config_id": f"S4_MLP_D{d}_L{L}_R{r:.1f}",
                    "suite": "Hypothesis_4_MLP_Expansion",
                    "n_layer": L,
                    "d_model": d,
                    "n_h": nh,
                    "n_kv": nkv,
                    "d_qk": 32,
                    "d_v": 32,
                    "d_mlp": dmlp,
                    "total_params_M": round(tot / 1e6, 3),
                    "layer_size_kb": round(l_kb, 1),
                    "mlp_ratio": round(dmlp / d, 2),
                    "kv_ratio": round(nkv / nh, 3)
                })
    random.seed(42)
    random.shuffle(configs)
    return configs[:100]

def generate_suite_e_latin_hypercube(num_samples=450, seed=42):
    """Suite E: Stratified Latin Hypercube Sampling covering the continuous design space."""
    np.random.seed(seed)
    random.seed(seed)
    
    # Define bounds
    d_model_choices = list(range(64, 640 + 1, 32))
    d_qk_choices = [16, 32, 48, 64]
    d_v_choices = [16, 32, 48, 64]
    
    configs = []
    seen = set()
    
    # 450 stratified bins across parameter size (1M to 80M)
    param_bins = np.linspace(1.0, 80.0, num_samples + 1)
    
    for i in range(num_samples):
        p_min, p_max = param_bins[i], param_bins[i+1]
        
        # Sample with rejection until falling into current stratum
        attempts = 0
        found = False
        while attempts < 200:
            attempts += 1
            L = random.randint(2, 28)
            d = random.choice(d_model_choices)
            nh_max = max(1, d // 16)
            nh = random.choice([h for h in [1, 2, 4, 6, 8, 12, 16] if h <= nh_max])
            valid_kvs = [k for k in range(1, nh + 1) if nh % k == 0]
            nkv = random.choice(valid_kvs)
            dqk = random.choice(d_qk_choices)
            dv = random.choice(d_v_choices)
            mlp_r = random.uniform(1.2, 5.5)
            dmlp = int(round((d * mlp_r) / 32.0) * 32)
            
            key = (L, d, nh, nkv, dqk, dv, dmlp)
            if key in seen: continue
            
            tot, l_param, tot_mb, l_kb = calc_model_specs(L, d, nh, nkv, dqk, dv, dmlp)
            p_m = tot / 1e6
            
            # Check parameter range
            if p_min <= p_m <= p_max or (attempts > 50 and 1.0 <= p_m <= 80.0):
                seen.add(key)
                configs.append({
                    "config_id": f"S5_LHS_{len(configs)+1:03d}_L{L}_D{d}_H{nh}",
                    "suite": "Hypothesis_5_LatinHypercube_Global",
                    "n_layer": L,
                    "d_model": d,
                    "n_h": nh,
                    "n_kv": nkv,
                    "d_qk": dqk,
                    "d_v": dv,
                    "d_mlp": dmlp,
                    "total_params_M": round(p_m, 3),
                    "layer_size_kb": round(l_kb, 1),
                    "mlp_ratio": round(dmlp / d, 2),
                    "kv_ratio": round(nkv / nh, 3)
                })
                found = True
                break
                
        if not found:
            # Fallback random sample
            L = random.randint(4, 16)
            d = random.choice(d_model_choices)
            nh = max(2, d // 32)
            dmlp = d * 3
            tot, _, _, l_kb = calc_model_specs(L, d, nh, 1, 32, 32, dmlp)
            configs.append({
                "config_id": f"S5_LHS_FB_{len(configs)+1:03d}_L{L}_D{d}",
                "suite": "Hypothesis_5_LatinHypercube_Global",
                "n_layer": L,
                "d_model": d,
                "n_h": nh,
                "n_kv": 1,
                "d_qk": 32,
                "d_v": 32,
                "d_mlp": dmlp,
                "total_params_M": round(tot / 1e6, 3),
                "layer_size_kb": round(l_kb, 1),
                "mlp_ratio": 3.0,
                "kv_ratio": round(1.0 / nh, 3)
            })
            
    return configs[:num_samples]

def main():
    print("=" * 70)
    print("Generating Structured 1,000 Configuration Search Space for Pixel Watch 5")
    print("=" * 70)
    
    suite_a = generate_suite_a_cache_inflection()
    suite_b = generate_suite_b_iso_capacity_width_depth()
    suite_c = generate_suite_c_attention_geometry()
    suite_d = generate_suite_d_mlp_expansion()
    suite_e = generate_suite_e_latin_hypercube(num_samples=1000 - len(suite_a) - len(suite_b) - len(suite_c) - len(suite_d))
    
    all_configs = suite_a + suite_b + suite_c + suite_d + suite_e
    
    df = pd.DataFrame(all_configs)
    
    # Assign global 1-based index
    df["sample_idx"] = range(1, len(df) + 1)
    
    out_csv = os.path.join(OUTPUT_DIR, "watch5_targeted_1000_sweep.csv")
    df.to_csv(out_csv, index=False)
    
    print(f"Total Configurations Generated: {len(df)}")
    print(f" - Suite A (L2/SLC Cache Inflection)       : {len(suite_a)} samples")
    print(f" - Suite B (Iso-Capacity Width vs Depth)   : {len(suite_b)} samples")
    print(f" - Suite C (Attention Geometry & KV-Share) : {len(suite_c)} samples")
    print(f" - Suite D (MLP Expansion Ratio Geometry)  : {len(suite_d)} samples")
    print(f" - Suite E (Latin Hypercube Global Space)  : {len(suite_e)} samples")
    print(f"\n[OK] Saved dataset configuration to:\n     {out_csv}")
    
    # Print Distribution Summary
    print("\nSearch Space Summary Statistics:")
    print(f" - Depth Range (L)         : {df['n_layer'].min()} to {df['n_layer'].max()} layers")
    print(f" - Width Range (d_model)   : {df['d_model'].min()} to {df['d_model'].max()}")
    print(f" - Parameter Range         : {df['total_params_M'].min():.2f}M to {df['total_params_M'].max():.2f}M")
    print(f" - Layer Working Set (KB)  : {df['layer_size_kb'].min():.1f} KB to {df['layer_size_kb'].max():.1f} KB")

if __name__ == "__main__":
    main()
