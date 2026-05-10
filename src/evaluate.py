import json
import math
import os
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from unsloth import FastLanguageModel
from src.model_loader import load_model


# ─────────────────────────── CONFIG ─────────────────────────────────────────
# ← FILL these two paths before running

MODEL_PATH   = "./src/output/Llama-3.1-8B-Instruct-Medical"
TEST_FILE    = "./data/processed/test.jsonl"
OUTPUT_DIR   = "./fine_eval"

MAX_SEQ_LEN  = 1024
MAX_NEW_TOK  = 256
LOAD_4BIT    = True
LIMIT        = None

SYSTEM_PROMPT = "You are a customer care doctor. Be polite and answer all questions from the customer."

def load_jsonl(path, limit=None):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            assert "question" in obj and "answer" in obj, (
                f"Line {i}: expected 'question' + 'answer', got {list(obj.keys())}"
            )
            samples.append({
                "question":  obj["question"].strip(),
                "reference": obj["answer"].strip(),
            })
    print(f"[DATA] Loaded {len(samples)} samples from {path}")
    return samples


def build_prompt(question, tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ─────────────────────────── INFERENCE ──────────────────────────────────────

def infer(model, tokenizer, question, device):
    prompt    = build_prompt(question, tokenizer)
    inputs    = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[-1]

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOK,
            use_cache=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.perf_counter() - t0) * 1000

    gen_ids        = outputs[0][input_len:]
    num_tok        = len(gen_ids)
    prediction     = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    tokens_per_sec = num_tok / (latency_ms / 1000) if latency_ms > 0 else 0.0

    return prediction, latency_ms, tokens_per_sec


# ─────────────────────────── NORMALIZE ──────────────────────────────────────

def norm(text):
    return unicodedata.normalize("NFC", text).lower().strip()

def chars(text):
    return list(norm(text))


# ─────────────────────────── ROUGE-L ────────────────────────────────────────

def _lcs_len(x, y):
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = (
                dp[i-1][j-1] + 1
                if x[i-1] == y[j-1]
                else max(dp[i-1][j], dp[i][j-1])
            )
    return dp[m][n]


def rouge_l(pred, ref):
    """
    Char-level ROUGE-L F1.
    Precision = LCS / len(pred_chars)
    Recall    = LCS / len(ref_chars)
    F1        = harmonic mean
    """
    pt, rt = chars(pred), chars(ref)
    if not pt or not rt:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    lcs   = _lcs_len(pt, rt)
    prec  = lcs / len(pt)
    rec   = lcs / len(rt)
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "precision": round(prec * 100, 4),
        "recall":    round(rec  * 100, 4),
        "f1":        round(f1   * 100, 4),
    }


# ─────────────────────────── BLEU-4 ─────────────────────────────────────────

def _ngram(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

def _clipped(hyp, ref, n):
    hc, rc = _ngram(hyp, n), _ngram(ref, n)
    return sum(min(c, rc[ng]) for ng, c in hc.items()), max(sum(hc.values()), 0)

def bleu4(pred, ref):
    """
    Char-level BLEU-4 with add-1 smoothing and brevity penalty.
    Returns score in [0, 100].
    """
    hyp, ref_ = chars(pred), chars(ref)
    bp = 1.0 if len(hyp) >= len(ref_) else math.exp(1 - len(ref_) / max(len(hyp), 1))
    precisions = []
    for n in range(1, 5):
        m, t = _clipped(hyp, ref_, n)
        precisions.append((m + 1) / (t + 1))   # add-1 smoothing
    log_avg = sum(math.log(p) for p in precisions) / 4
    return round(bp * math.exp(log_avg) * 100, 4)


# ─────────────────────────── BERTSCORE ──────────────────────────────────────

def _embed(model, tokenizer, text, device):
    """
    Extract L2-normalized token embeddings from model's embed_tokens layer.
    Truncate to 512 tokens to avoid OOM on long responses.
    """
    ids = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=512
    ).to(device)
    with torch.no_grad():
        emb = model.model.embed_tokens(ids["input_ids"])   # (1, L, H)
    return F.normalize(emb[0], dim=-1)                     # (L, H)


def bertscore(model, tokenizer, pred, ref, device):
    """
    BERTScore using the model's own embedding space.
    Precision: avg max cosine sim of each pred token to ref tokens
    Recall:    avg max cosine sim of each ref token to pred tokens
    F1:        harmonic mean
    Score in [0, 100].
    """
    pred_emb = _embed(model, tokenizer, pred, device)   # (P, H)
    ref_emb  = _embed(model, tokenizer, ref,  device)   # (R, H)

    sim   = torch.mm(pred_emb, ref_emb.T)               # (P, R) cosine similarity matrix
    prec  = sim.max(dim=1).values.mean().item()          # each pred token → best ref match
    rec   = sim.max(dim=0).values.mean().item()          # each ref token  → best pred match
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "precision": round(prec * 100, 4),
        "recall":    round(rec  * 100, 4),
        "f1":        round(f1   * 100, 4),
    }


# ─────────────────────────── PERPLEXITY ─────────────────────────────────────

def perplexity(model, tokenizer, question, reference, device):
    """
    Perplexity of the reference answer conditioned on the question prompt.

    Steps:
      1. Tokenize: full = prompt_tokens + reference_tokens
      2. Labels:   -100 for prompt tokens (excluded from loss)
                   reference token ids as-is
      3. Forward pass → cross-entropy loss → PPL = exp(loss)

    Lower PPL = model is more confident about the reference answer.
    Typical range after fine-tuning: 1.5 – 8.0
    """
    prompt    = build_prompt(question, tokenizer)
    full_text = prompt + reference

    enc_prompt = tokenizer(prompt,    add_special_tokens=False)
    enc_full   = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")

    prompt_len = len(enc_prompt["input_ids"])
    input_ids  = enc_full["input_ids"].to(device)

    labels = input_ids.clone()
    labels[0, :prompt_len] = -100   # ignore prompt in loss

    with torch.no_grad():
        out  = model(input_ids=input_ids, labels=labels)
        loss = out.loss.item()

    ppl = math.exp(loss) if loss < 100 else float("inf")
    return round(ppl, 4)


# ─────────────────────────── AGGREGATE STATS ─────────────────────────────────

def stats(values, label):
    """Return mean/median/min/max/p10/p90 for a list of floats."""
    a = np.array([v for v in values if not math.isinf(v)])
    return {
        f"{label}_mean":   round(float(np.mean(a)),              4),
        f"{label}_median": round(float(np.median(a)),            4),
        f"{label}_p10":    round(float(np.percentile(a, 10)),    4),
        f"{label}_p90":    round(float(np.percentile(a, 90)),    4),
        f"{label}_min":    round(float(np.min(a)),               4),
        f"{label}_max":    round(float(np.max(a)),               4),
    }


def latency_stats(latencies_ms, tps_list):
    a = np.array(latencies_ms)
    return {
        "latency_mean_ms":   round(float(np.mean(a)),              2),
        "latency_p50_ms":    round(float(np.percentile(a, 50)),    2),
        "latency_p95_ms":    round(float(np.percentile(a, 95)),    2),
        "latency_p99_ms":    round(float(np.percentile(a, 99)),    2),
        "latency_min_ms":    round(float(np.min(a)),               2),
        "latency_max_ms":    round(float(np.max(a)),               2),
        "throughput_mean_tok_per_sec": round(float(np.mean(tps_list)), 2),
    }


# ─────────────────────────── MAIN ────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    samples = load_jsonl(TEST_FILE, limit=LIMIT)

    model, tokenizer = load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device.upper()}\n")

    # Per-sample storage
    records     = []
    rougeL_list = []
    bleu4_list  = []
    bs_f1_list  = []
    ppl_list    = []
    lat_list    = []
    tps_list    = []

    total = len(samples)

    for idx, sample in enumerate(samples):
        q   = sample["question"]
        ref = sample["reference"]

        # ── 1. Inference ────────────────────────────────────────────────────
        pred, lat_ms, tps = infer(model, tokenizer, q, device)

        # ── 2. ROUGE-L ──────────────────────────────────────────────────────
        rl = rouge_l(pred, ref)

        # ── 3. BLEU-4 ───────────────────────────────────────────────────────
        b4 = bleu4(pred, ref)

        # ── 4. BERTScore ────────────────────────────────────────────────────
        bs = bertscore(model, tokenizer, pred, ref, device)

        # ── 5. Perplexity ───────────────────────────────────────────────────
        ppl = perplexity(model, tokenizer, q, ref, device)

        # ── Collect ─────────────────────────────────────────────────────────
        rougeL_list.append(rl["f1"])
        bleu4_list.append(b4)
        bs_f1_list.append(bs["f1"])
        ppl_list.append(ppl)
        lat_list.append(lat_ms)
        tps_list.append(tps)

        record = {
            "idx":        idx,
            "question":   q,
            "reference":  ref,
            "prediction": pred,
            "rouge_l":    rl,                  # {precision, recall, f1}
            "bleu4":      b4,                  # float
            "bertscore":  bs,                  # {precision, recall, f1}
            "perplexity": ppl,                 # float
            "latency_ms": round(lat_ms, 2),
            "tokens_per_sec": round(tps, 2),
        }
        records.append(record)

        # ── Progress ─────────────────────────────────────────────────────────
        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(
                f"  [{idx+1:>4}/{total}] "
                f"ROUGE-L: {np.mean(rougeL_list):.2f} | "
                f"BLEU-4: {np.mean(bleu4_list):.2f} | "
                f"BERTScore F1: {np.mean(bs_f1_list):.2f} | "
                f"PPL: {np.mean([p for p in ppl_list if not math.isinf(p)]):.2f} | "
                f"Lat: {np.mean(lat_list):.0f}ms"
            )

    # ── Aggregate summary ────────────────────────────────────────────────────
    print("\n[INFO] Building summary...")
    summary = {
        "num_samples": total,
        "model_path":  MODEL_PATH,
        **stats(rougeL_list, "rouge_l_f1"),
        **stats(bleu4_list,  "bleu4"),
        **stats(bs_f1_list,  "bertscore_f1"),
        **stats(ppl_list,    "perplexity"),
        **latency_stats(lat_list, tps_list),
    }

    # ── Print final results ──────────────────────────────────────────────────
    print("\n" + "═" * 62)
    print("  FINE EVAL RESULTS — Vietnamese Healthcare Assistant")
    print("═" * 62)
    print(f"  Samples          : {total}")
    print(f"  ROUGE-L  F1      : mean={summary['rouge_l_f1_mean']:.2f}  median={summary['rouge_l_f1_median']:.2f}  p10={summary['rouge_l_f1_p10']:.2f}  p90={summary['rouge_l_f1_p90']:.2f}")
    print(f"  BLEU-4           : mean={summary['bleu4_mean']:.2f}  median={summary['bleu4_median']:.2f}  p10={summary['bleu4_p10']:.2f}  p90={summary['bleu4_p90']:.2f}")
    print(f"  BERTScore F1     : mean={summary['bertscore_f1_mean']:.2f}  median={summary['bertscore_f1_median']:.2f}  p10={summary['bertscore_f1_p10']:.2f}  p90={summary['bertscore_f1_p90']:.2f}")
    print(f"  Perplexity       : mean={summary['perplexity_mean']:.2f}  median={summary['perplexity_median']:.2f}  p10={summary['perplexity_p10']:.2f}  p90={summary['perplexity_p90']:.2f}")
    print(f"  Latency          : mean={summary['latency_mean_ms']}ms  p95={summary['latency_p95_ms']}ms  p99={summary['latency_p99_ms']}ms")
    print(f"  Throughput       : {summary['throughput_mean_tok_per_sec']} tok/sec")
    print("═" * 62 + "\n")

    # ── Save all outputs ─────────────────────────────────────────────────────
    out = Path(OUTPUT_DIR)

    # summary.json
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # per_sample.jsonl — all records
    with open(out / "per_sample.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # low_rouge.jsonl — bottom 10% ROUGE-L for error analysis
    threshold_rouge = float(np.percentile(rougeL_list, 10))
    low_rouge = [r for r in records if r["rouge_l"]["f1"] <= threshold_rouge]
    with open(out / "low_rouge.jsonl", "w", encoding="utf-8") as f:
        for r in sorted(low_rouge, key=lambda x: x["rouge_l"]["f1"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # low_bertscore.jsonl — bottom 10% BERTScore F1
    threshold_bs = float(np.percentile(bs_f1_list, 10))
    low_bs = [r for r in records if r["bertscore"]["f1"] <= threshold_bs]
    with open(out / "low_bertscore.jsonl", "w", encoding="utf-8") as f:
        for r in sorted(low_bs, key=lambda x: x["bertscore"]["f1"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[INFO] summary.json       → {out / 'summary.json'}")
    print(f"[INFO] per_sample.jsonl   → {out / 'per_sample.jsonl'}  ({total} records)")
    print(f"[INFO] low_rouge.jsonl    → {out / 'low_rouge.jsonl'}   ({len(low_rouge)} samples, ROUGE-L ≤ {threshold_rouge:.2f})")
    print(f"[INFO] low_bertscore.jsonl→ {out / 'low_bertscore.jsonl'}({len(low_bs)} samples, BERTScore ≤ {threshold_bs:.2f})")


if __name__ == "__main__":
    main()