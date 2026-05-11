import os
import subprocess
import time
from pathlib import Path


# CONFIG

MERGED_PATH = "outputs/Llama-3.1-8B-Instruct-Medical-merged-fp16"
OUTPUT_DIR  = "gguf"
LLAMA_CPP   = "llama.cpp"

QUANT_TARGETS = [
    ("q2_k",   "Q2_K"), # minimum size
    ("q3_k_m", "Q3_K_M"),  # low quality, lightweight CPU inference
    ("q4_k_m", "Q4_K_M"),  # sweet spot, recommended for Ollama/production
    ("q5_k_m", "Q5_K_M"),  # high quality, small GPU
    ("q6_k",   "Q6_K"),    # near fp16 quality
    ("q8_0",   "Q8_0"),    # best GGUF quality, benchmark reference
]


# STEP 1: CONVERT TO GGUF

def convert_to_gguf():
    assert Path(MERGED_PATH).exists(), (
        f"Merged model not found at: {MERGED_PATH}\n"
    )

    convert_script = Path(LLAMA_CPP) / "convert_hf_to_gguf.py"
    assert convert_script.exists(), (
        f"convert_hf_to_gguf.py not found at {convert_script}\n"
        f"Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp"
    )

    base_gguf = Path(OUTPUT_DIR) / "base_bf16.gguf"
    if base_gguf.exists():
        print(f"[CONVERT] Base GGUF already exists → {base_gguf}, skipping.")
        return str(base_gguf)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cmd = [
        "python3", str(convert_script),
        MERGED_PATH,
        "--outtype", "bf16",
        "--outfile", str(base_gguf),
    ]
    print(f"[CONVERT] Running: {' '.join(cmd)}")
    t0     = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"[ERROR] Conversion failed:\n{result.stderr}")
        raise RuntimeError("GGUF base conversion failed.")

    print(f"[CONVERT] Base GGUF saved at {base_gguf}  ({elapsed:.0f}s)\n")
    return str(base_gguf)


# STEP 2: QUANTIZE

def quantize(base_gguf):
    # Support different llama.cpp build paths
    for candidate in [
        Path(LLAMA_CPP) / "build" / "bin" / "llama-quantize",
        Path(LLAMA_CPP) / "llama-quantize",
        Path(LLAMA_CPP) / "quantize",
    ]:
        if candidate.exists():
            quantize_bin = candidate
            break
    else:
        raise FileNotFoundError(
            f"llama-quantize binary not found. Build llama.cpp first:\n"
            f"  cd {LLAMA_CPP} && cmake -B build && cmake --build build --config Release"
        )

    results = []
    total   = len(QUANT_TARGETS)

    for i, (quant_type, folder_name) in enumerate(QUANT_TARGETS, 1):
        out_dir  = Path(OUTPUT_DIR) / folder_name
        out_file = out_dir / f"medical-llama3.1-8b-{folder_name.lower()}.gguf"
        os.makedirs(out_dir, exist_ok=True)

        if out_file.exists():
            size_gb = out_file.stat().st_size / 1e9
            print(f"  [{i}/{total}] SKIP {folder_name} — already exists ({size_gb:.2f} GB)")
            results.append((folder_name, "skipped", size_gb, 0))
            continue

        cmd = [str(quantize_bin), base_gguf, str(out_file), quant_type.upper()]
        print(f"  [{i}/{total}] Quantizing → {folder_name} ({quant_type.upper()})...")

        t0     = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"    [ERROR] {folder_name} failed:\n{result.stderr[:300]}")
            results.append((folder_name, "failed", 0, elapsed))
            continue

        size_gb = out_file.stat().st_size / 1e9
        print(f"    [OK] {out_file.name}  {size_gb:.2f} GB  ({elapsed:.0f}s)")
        results.append((folder_name, "ok", size_gb, elapsed))

    return results


# STEP 3: SUMMARY

def print_summary(results):
    print("\n" + "═" * 62)
    print("  GGUF CONVERSION SUMMARY")
    print("═" * 62)
    print(f"  {'Format':<18} {'Status':<10} {'Size (GB)':<12} {'Time (s)'}")
    print("  " + "─" * 56)
    total_size = 0
    for name, status, size_gb, elapsed in results:
        icon = "✓" if status == "ok" else ("↷" if status == "skipped" else "✗")
        print(f"  {icon} {name:<16} {status:<10} {size_gb:<12.2f} {elapsed:.0f}s")
        total_size += size_gb
    print("  " + "─" * 56)
    print(f"  {'TOTAL':<18} {'':<10} {total_size:.2f} GB")
    print("═" * 62 + "\n")


# MAIN

def main():
    print("  GGUF Multi-Format Converter — Vietnamese Healthcare LLM")

    print(f"[INFO] Input  : {MERGED_PATH}")
    print(f"[INFO] Output : {OUTPUT_DIR}\n")

    # Step 1: Convert merged fp16 to base GGUF
    base_gguf = convert_to_gguf()

    # Step 2: Quantize to all target formats
    print(f"[QUANT] Starting {len(QUANT_TARGETS)} quantization jobs...\n")
    results = quantize(base_gguf)

    # Step 3: Summary
    print_summary(results)

    print(f"[INFO] All GGUF files saved under: {Path(OUTPUT_DIR).resolve()}")
    print("[INFO] Output structure:")
    for name, status, _, _ in results:
        if status in ("ok", "skipped"):
            print(f"  {OUTPUT_DIR}/{name}/medical-llama3.1-8b-{name.lower()}.gguf")


if __name__ == "__main__":
    main()