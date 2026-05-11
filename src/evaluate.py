import json
import math
import os
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from unsloth import FastLanguageModel
from src.model_loader import load_model


# ───────────────────────── CONFIG ─────────────────────────

MODEL_PATH = "./outputs/Llama-3.1-8B-Instruct-Medical-merged-fp16"
TEST_FILE  = "./data/processed/test.jsonl"
OUTPUT_DIR = "./fine_eval"

MAX_SEQ_LEN = 1024
MAX_NEW_TOK = 256
LIMIT = None

SYSTEM_PROMPT = """
You are a Vietnamese medical assistant.
Answer accurately, safely, and avoid making medical diagnoses
"""


# ───────────────────────── DATA ─────────────────────────

def load_jsonl(path, limit=None):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            obj = json.loads(line)
            data.append({
                "question": obj["question"].strip(),
                "reference": obj["answer"].strip()
            })
    print(f"[DATA] Loaded {len(data)} samples")
    return data


# ───────────────────────── PROMPT ─────────────────────────

def build_prompt(q, tokenizer):
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q},
    ]
    return tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True
    )


# ───────────────────────── INFER ─────────────────────────

def infer(model, tokenizer, q, device):
    prompt = build_prompt(q, tokenizer)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
    ).to(device)

    input_len = inputs["input_ids"].shape[-1]

    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOK,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = (time.perf_counter() - start) * 1000

    gen = out[0][input_len:]
    pred = tokenizer.decode(gen, skip_special_tokens=True).strip()

    tps = len(gen) / (latency / 1000 + 1e-9)

    return pred, latency, tps


# ───────────────────────── METRICS ─────────────────────────

def norm(t):
    return unicodedata.normalize("NFC", t).lower().strip()

def chars(t):
    return list(norm(t))


def lcs(a, b):
    dp = [[0] * (len(b)+1) for _ in range(len(a)+1)]
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            dp[i][j] = (
                dp[i-1][j-1] + 1
                if a[i-1] == b[j-1]
                else max(dp[i-1][j], dp[i][j-1])
            )
    return dp[-1][-1]


def rouge_l(pred, ref):
    p, r = chars(pred), chars(ref)
    if not p or not r:
        return 0.0

    l = lcs(p, r)
    prec = l / len(p)
    rec = l / len(r)
    return (2 * prec * rec) / (prec + rec + 1e-9) * 100


def bleu(pred, ref):
    hp, rf = chars(pred), chars(ref)

    bp = 1.0 if len(hp) >= len(rf) else math.exp(1 - len(rf) / (len(hp) + 1e-9))

    scores = []
    for n in range(1, 5):
        hyp = Counter(tuple(hp[i:i+n]) for i in range(len(hp)-n+1))
        refc = Counter(tuple(rf[i:i+n]) for i in range(len(rf)-n+1))

        match = sum(min(hyp[g], refc[g]) for g in hyp)
        total = sum(hyp.values()) + 1e-9
        scores.append((match + 1) / (total + 1))

    return bp * math.exp(sum(math.log(s) for s in scores) / 4) * 100


def ppl(model, tokenizer, q, r, device):
    prompt = build_prompt(q, tokenizer)
    full = prompt + r

    enc_p = tokenizer(prompt, add_special_tokens=False)
    enc_f = tokenizer(full, return_tensors="pt", add_special_tokens=False)

    ids = enc_f["input_ids"].to(device)
    pl = len(enc_p["input_ids"])

    labels = ids.clone()
    labels[:, :pl] = -100

    with torch.no_grad():
        loss = model(input_ids=ids, labels=labels).loss.item()

    return math.exp(loss) if loss < 100 else float("inf")


# ───────────────────────── MAIN ─────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    samples = load_jsonl(TEST_FILE, LIMIT)

    model, tokenizer = load_model(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f"[INFO] Device: {device}")

    results = []
    rouge_list, bleu_list, ppl_list = [], [], []
    lat_list, tps_list = [], []

    for i, s in enumerate(samples):
        q, r = s["question"], s["reference"]

        pred, lat, tps = infer(model, tokenizer, q, device)
        rl = rouge_l(pred, r)
        bl = bleu(pred, r)
        p = ppl(model, tokenizer, q, r, device)

        rouge_list.append(rl)
        bleu_list.append(bl)
        ppl_list.append(p)
        lat_list.append(lat)
        tps_list.append(tps)

        results.append({
            "idx": i,
            "question": q,
            "reference": r,
            "prediction": pred,
            "rouge_l": rl,
            "bleu": bl,
            "ppl": p,
            "latency_ms": lat,
            "tps": tps
        })

        if i % 50 == 0:
            print(f"[{i}] ROUGE-L: {rl:.2f} | BLEU: {bl:.2f} | PPL: {p:.2f}")

    # ── SAVE ─────────────────────────

    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "eval.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "rouge_l_mean": float(np.mean(rouge_list)),
        "bleu_mean": float(np.mean(bleu_list)),
        "ppl_mean": float(np.mean([x for x in ppl_list if np.isfinite(x)])),
        "latency_mean_ms": float(np.mean(lat_list)),
        "tps_mean": float(np.mean(tps_list)),
        "num_samples": len(results)
    }

    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[DONE] Saved eval.jsonl + summary.json")


if __name__ == "__main__":
    main()