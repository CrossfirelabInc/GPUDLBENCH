# GPU AI Benchmark Tests — Explanations

> This document explains what each test in the Crossfirelab GPU Benchmark suite does, what it measures, and what it means in practice. Written for YouTube viewers and tech enthusiasts.

---

## 1. Vision Training (Training Vision) — ResNet-50 / ResNet-101

**What does it do?**
Image classification models (ResNet-50 and ResNet-101) are trained on the GPU. The number of images processed per second is measured at FP32, FP16, and BF16 precisions.

**What does it mean in practice?**
- If you want to train your own image classification model (e.g. cat/dog detection, medical image analysis, product recognition), this test shows how fast your GPU will handle it.
- Higher result = shorter training time. The difference between 500 img/s and 1000 img/s means cutting a 10-hour training run down to 5 hours.
- FP16/BF16 results show how much speed you gain from mixed-precision training — on modern GPUs this can be a **2–3× speedup**.
- Comparison charts show each GPU's **peak throughput** and **same batch size** comparisons separately.

---

## 2. NLP Training (Training NLP) — BERT-Base / BERT-Large

**What does it do?**
Natural language processing (NLP) models BERT-base and BERT-large are trained on the GPU. The number of text samples processed per second is measured.

**What does it mean in practice?**
- If you want to train language models — the foundation of ChatGPT-like systems — this test matters.
- Sentiment analysis, text classification, and question-answering systems all use this type of training.
- BERT-large is **3× larger** than BERT-base — GPU performance differences become more visible with larger models.
- Researchers and companies use these results to make GPU investment decisions.
- Comparison charts show **maximum throughput** and **same batch size** graphs separately.

---

## 3. Vision Inference (Inference Vision) — ResNet-50 / ResNet-101

**What does it do?**
Pre-trained image models run predictions (inference). The number of images classified per second is measured.

**What does it mean in practice?**
- For **real-time** applications like security camera analysis, license plate recognition, and factory quality control, this number is critical.
- 1000 img/s = classifying 1000 photos per second. A security camera streams at 30 FPS, so 1000 img/s means you can analyse **33 cameras simultaneously**.
- For web services: each API request classifies one image — higher throughput = more concurrent users.
- Comparison charts show **peak throughput at optimal batch size**.

---

## 4. NLP Inference (Inference NLP) — BERT-Base / BERT-Large

**What does it do?**
Measures how many text samples per second can be processed using trained NLP models.

**What does it mean in practice?**
- In chatbots, search engines, translation services, and spam filters, every user request is an inference call.
- 500 samples/s = 500 text analyses per second. For example, instant sentiment analysis of product reviews on an e-commerce site.
- Higher inference speed = lower server costs. You can serve more users with the same GPU.
- Check the **BS=** labels in the charts to see whether performance differences come from VRAM or raw compute power.

---

## 5. LLM Token Generation (LLM Tokens per Second)

**What does it do?**
Large language models (DeepSeek, Phi-4, Gemma, QwQ, etc.) are run on the GPU via llama.cpp. The number of tokens (word fragments) generated per second and time-to-first-token (TTFT) are measured.

**What does it mean in practice?**
- **This shows how fast AI assistants like ChatGPT/Claude would run locally on your computer.**
- 50 tokens/s ≈ ~35 words per second = a smooth chat experience.
- 10 tokens/s = a slow, word-by-word experience.
- Time-to-first-token (TTFT) = how long you wait for the first response after asking a question. Under 100ms is ideal.
- Larger models (27B, 32B) are smarter but slower — GPU power is the deciding factor.

---

## 6. VRAM Limits (VRAM Limits)

**What does it do?**
Tests the largest model that fits in GPU memory (VRAM) and the maximum context length achievable.

**What does it mean in practice?**
- **VRAM = your GPU's RAM.** More VRAM = ability to run larger, smarter models.
- 24GB VRAM ≈ 13B parameter model (Llama-2-13B level)
- 48GB VRAM ≈ 30B+ parameter model
- Context length = how many words the model can "remember" at once. Long document analysis and book summarisation require long context.
- **If a model doesn't fit in VRAM, you either need a smaller model or quantisation (compression).**

---

## 7. Object Detection Training (Object Detection) — Faster R-CNN / Mask R-CNN

**What does it do?**
Models that detect and localise objects in images (Faster R-CNN, Mask R-CNN) are trained on the GPU.

**What does it mean in practice?**
- Autonomous vehicles: detecting pedestrians, vehicles, and road signs.
- Medical imaging: tumour detection in X-ray/MRI scans.
- Security: detecting suspicious objects/people in camera feeds.
- Retail: shelf analysis, customer counting.
- **A much heavier workload than ResNet classification** — GPU memory and compute are tested simultaneously.
- Comparison charts show **maximum throughput** and **same batch size** graphs separately.

---

## 8. Multi-GPU Scaling (Multi-GPU Scaling)

**What does it do?**
Runs the same workload on 1 GPU and 2 GPUs to measure how much benefit the second GPU provides. Three models are tested: ResNet-50 (vision/CNN), BERT-base (NLP encoder), and GPT-2 Large (~774M parameters, LLM decoder). For each model, single-GPU baseline, DDP, and FSDP ZeRO-2 methods are compared. Scaling efficiency (%) = 2-GPU throughput / (2 × 1-GPU throughput) × 100.

**What does it mean in practice?**
- **Is buying a second GPU worth it?** This test answers exactly that.
- 100% efficiency = 2 GPUs give exactly 2× speed. In practice this is nearly impossible because inter-GPU communication takes time.
- 85–95% efficiency = excellent scaling. Your second GPU investment pays off.
- 70–85% efficiency = good scaling. Significant benefit from the second GPU.
- Below 50% = poor scaling. The second GPU isn't fully earning its keep.
- **Training generally scales well**, inference is harder to scale because batch sizes are smaller.

### Methods Used

**DDP (DistributedDataParallel)**
Each GPU holds a full copy of the model. Each GPU processes its own batch, then gradients are gathered and synchronised via NCCL all-reduce. The most common and simplest multi-GPU training method. No VRAM savings — each GPU separately holds the full model and optimiser states. Best performance (%95+ efficiency) with small to medium models.

**FSDP ZeRO-2 (Fully Sharded Data Parallel — SHARD_GRAD_OP)**
The PyTorch-native equivalent of Microsoft DeepSpeed ZeRO Stage-2. Unlike DDP, it shards gradients and optimiser states across GPUs. Each GPU holds only 1/N of this data — saving VRAM and enabling larger models/batches to fit. Model parameters are kept in full on each GPU during forward and backward passes (this differentiates it from ZeRO-3). Can scale better than DDP for large models (like GPT-2 Large) thanks to VRAM efficiency.

---

## Precision Types Explained

| Abbreviation | Description | Use Case |
|-------------|-------------|----------|
| **FP32** | 32-bit floating point (full precision) | Scientific computing, training reference |
| **FP16** | 16-bit floating point (half precision) | Training/inference acceleration |
| **BF16** | Brain Float 16 (Google format) | Modern AI training (wider value range) |
| **FP8** | 8-bit floating point | Next-gen GPU ultra-fast inference |
| **FP64** | 64-bit double precision | Scientific simulation, HPC |

> **General rule:** Lower precision = faster but potentially less accurate. In modern AI, FP16/BF16 is the "gold standard" — the speed gain is large, and accuracy loss is negligible.

---

## Comparison Charts: Maximum Throughput and Same Batch Size

Benchmarks 1–4 and 7 (vision/NLP training, inference, and object detection) run at multiple batch sizes. Each GPU automatically scales up to the largest batch size it can support. Therefore, GPUs with different VRAM capacities may achieve different batch sizes.

This is shown in the comparison charts with **two separate chart types**:

### Maximum Throughput
- Shows each GPU's **peak throughput** at whichever batch size achieved it.
- Chart labels include `(BS=64)` annotations — showing which batch size was used.
- **Question:** "At what batch size does this GPU deliver its best performance?"
- Available for all training and inference tests.

### Same Batch Size (Batch-Normalised)
- The **smallest common batch size** across all GPUs is selected, and all GPUs' throughput at that batch size is compared.
- This completely removes the VRAM advantage: every GPU processes the same workload.
- Chart labels include `BS=16` annotations — showing which batch size the comparison uses.
- **Question:** "At the same workload, which GPU is faster?"
- Ideal for raw compute power comparison.

> **Which should you look at?**
> - For real-world performance → **Maximum Throughput** chart (in practice every GPU runs at its optimal batch size)
> - For raw GPU compute power → **Same Batch Size** chart (VRAM advantage removed, equal workload)

---

## GPU Scorecard

The **GPU Scorecard** table at the end of the comparison charts summarises all benchmark results at a glance. Each row represents a metric, each column a GPU. The GPU with the best result in each category is highlighted with a green border and ★ BEST label.

Key metrics shown on the scorecard:
- **Training/inference throughput** — peak values
- **LLM speed** — token/s for each model, with VRAM loadable status (✅ loaded / ⚠️ partial / ❌ too large)
- **VRAM capacity** — largest loadable model and context length
- **LLMFit recommendations** — best-fit, largest Dense and largest MoE LLM for your VRAM (explained below)
- **DLPerf composite score** — combined CNN + Transformer + LLM performance normalised to RTX PRO 5000 Blackwell = 100
- **Power consumption & temperature** — lower is better

---

## Dense vs MoE (Mixture of Experts) Architecture

The scorecard includes "Largest Dense LLM" and "Largest MoE LLM" rows. The difference between these two architectures directly impacts GPU choice in modern AI:

### What is a Dense Model?
Traditional deep learning architecture. **All parameters are actively used** for every token.

- Examples: Llama-3.1-70B, Qwen2.5-72B, Falcon-40B
- 70B Dense model = 70 billion parameters computed for every token generated
- **Advantage:** Simple architecture, predictable performance, low latency
- **Disadvantage:** High VRAM requirement — 70B Q4 ≈ ~40 GB VRAM needed
- **VRAM rule:** All parameters must be loaded into GPU memory

### What is a MoE (Mixture of Experts) Model?
The model's total parameter count is very large (e.g. 671B), but only a **small "expert" subset** is active per token.

- Examples: DeepSeek-R1 (671B total, ~37B active), Llama-4-Maverick (400B total, ~17B active), Qwen3-235B-A22B (235B total, ~22B active)
- **Advantage:** Massive knowledge capacity used with far less computation. A 671B MoE model runs at roughly ~37B Dense speed.
- **Disadvantage:** Total parameter count must fit in VRAM (or be partially offloaded to RAM). Routing mechanism adds extra latency.
- **VRAM rule:** Active expert parameters stay in GPU memory; the rest can be offloaded to RAM (but this reduces speed)

### What Does This Mean in Practice?

| Feature | Dense (e.g. 70B) | MoE (e.g. DeepSeek-R1 671B) |
|---------|-------------------|------------------------------|
| Total parameters | 70B | 671B |
| Active parameters per token | 70B | ~37B |
| VRAM requirement (Q4_K_M) | ~40 GB | ~37 GB (active) + RAM offload |
| Token/s speed | Medium | High (active portion is small) |
| Knowledge capacity | 70B level | 671B level (much broader) |

> **Summary:** MoE models provide access to massive parameter counts with limited VRAM. A 24 GB VRAM GPU can run at most ~30B Dense models, but can run the active portion of MoE models with 400B total parameters. However, the full model size must still fit in system RAM.

---

## How to Read the Results?

1. **Training tests** → "How fast can I train my model with this GPU?"
2. **Inference tests** → "How many users/cameras can I serve simultaneously with a trained model?"
3. **LLM test** → "How smooth will my local ChatGPT experience be?"
4. **VRAM test** → "How large a model can I run?"
5. **Watt/Performance** → "How little energy do I need for the same performance?"
6. **Relative performance** → "How many times faster is this GPU compared to the baseline?"
7. **Object detection** → "Which GPU has the advantage in detection model training?"
8. **Dual-GPU test** → "Is the second GPU actually worth it?"
9. **Max throughput vs Same BS** → "Is the difference from VRAM or raw compute?"
10. **Scorecard** → "Across all tests, which GPU stands out?"
11. **Dense vs MoE** → "What's the largest model that fits in my VRAM?"
