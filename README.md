# 🏥 vietnam-med-llm

 Fine-tuned LLaMA 3.1 8B for Vietnamese medical Q&A using LoRA + Unsloth. Trained on 9,000 curated samples with structured response format and full GGUF export pipeline for local deployment via llama.cpp.

---

## 📌 Overview

**vietnam-med-llm** adapts LLaMA 3.1 8B Instruct for Vietnamese-language medical question answering. The primary motivation is the lack of high-quality, open-source medical LLMs for Vietnamese — a low-resource language significantly underrepresented in medical NLP benchmarks.

The system targets patients seeking reliable symptom, diagnosis, and medication information in Vietnamese, where general-purpose LLMs frequently hallucinate or produce clinically unsafe outputs.

| Attribute | Detail |
|---|---|
| 🤖 **Base Model** | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` |
| 🔧 **Fine-tuning Method** | LoRA (r=16, alpha=16, RSLoRA) via Unsloth |
| 🌐 **Language** | Vietnamese 🇻🇳 |
| 🩺 **Domain** | Medical Q&A (symptoms, diagnosis, medication) |
| 📊 **Training Data** | Vietnamese Medical QA — 9,000 samples |
| 🖥️ **Training Hardware** | Tesla V100-SXM2 32GB |
| 📦 **Export** | GGUF (Q2\_K → Q8\_0) via llama.cpp |

---

## 🧠 Approach

### 1. 🎯 Base Model Selection

LLaMA 3.1 8B Instruct was selected for its strong multilingual token coverage and instruction-following capability at the 8B scale. Unsloth's optimized fork was used over the vanilla HuggingFace checkpoint to leverage custom CUDA kernels — achieving ~2x training throughput on the same hardware.

### 2. ⚙️ LoRA Fine-Tuning

Full fine-tuning of an 8B model is computationally prohibitive. LoRA adapters were injected across **all projection layers** (attention + MLP) to give the model sufficient capacity for domain adaptation:

```python
r              = 16
lora_alpha     = 16
lora_dropout   = 0.5 # regularization for 9K sample dataset
use_rslora     = True
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # full attention
    "gate_proj", "up_proj", "down_proj",        # full MLP/FFN
]
```

Training ran in **fp16** (V100 does not support bf16 natively).

### 5. 📦 GGUF Export

Post-training, LoRA adapters were merged into base model weights (`merged_16bit`) then quantized to 6 GGUF formats via llama.cpp:

| Format | Size | Use case |
|---|---|---|
| Q2\_K | ~3.2 GB | 🪶 RAM-limited / quick test |
| Q3\_K\_M | ~4.0 GB | 💻 CPU-only inference |
| Q4\_K\_M | ~4.9 GB | ✅ Recommended — Ollama/production |
| Q5\_K\_M | ~5.7 GB | 🎯 High quality, small GPU |
| Q6\_K | ~6.6 GB | 💎 Near fp16 quality |
| Q8\_0 | ~8.5 GB | 🏆 Best quality, benchmark |

---

## 🚀 Quickstart

Try the model instantly on **HuggingFace Spaces** — no setup required:

👉 **[Run on HuggingFace Space](https://huggingface.co/spaces/cngchis/vietnam-medical)**

---

## ⚠️ Limitations

- **Dataset coverage** — 9,000 samples covers common conditions but underperforms on rare diseases and complex drug interactions (PPL max=56.6).
- **No RAG** — model relies entirely on parametric knowledge; does not retrieve from live medical references or updated clinical guidelines.
- **Vietnamese only** — code-switched or English queries may produce degraded outputs.
- **Not a medical device** — outputs should not substitute professional medical advice. No clinical validation has been performed.

---

## 🔭 Future Work

- [ ] **Structured retraining** — enforce 3-part response format (greet → explain → recommend) to reduce BLEU variance
- [ ] **RAG pipeline** — retrieval over medical knowledge base to reduce hallucination
- [ ] **DPO alignment** — preference optimization using clinician-rated response pairs
- [ ] **Dataset expansion** — clinical notes, drug leaflets, Vietnamese medical literature
- [ ] **Safety classifier** — detect and refuse high-risk queries (e.g. medication dosage for self-harm)
- [ ] **Bilingual extension** — handle Vietnamese–English code-switched inputs

---

## 📜 Citation

```bibtex
@misc{vietnam-med-llm,
  author    = {cngchis},
  title     = {vietnam-med-llm: LoRA Fine-tuned LLaMA 3.1 for Vietnamese Medical Q&A},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/cngchis/vietnam-med-llm}
}
```

---

## ⚠️ Disclaimer

This model is intended for **research and educational purposes only**. It is not a certified medical device and has not been validated for clinical use. Always consult a qualified healthcare professional for medical decisions.

---

## 📜 License

[MIT](LICENSE) — model weights derived from LLaMA 3.1 are subject to [Meta's LLaMA Community License](https://llama.meta.com/llama-downloads/).