# GPU Deep Learning Benchmark Suite

**By Crossfirelab** — Vibe Coded 🎯

> **[Türkçe](#türkçe)** | **[English](#english)**

---

<a name="türkçe"></a>
## 🇹🇷 Türkçe

NVIDIA GPU'lar için yapay zeka odaklı benchmark paketi. Eğitim hızı, çıkarım performansı, LLM token üretimi, VRAM kapasitesi ve çift-GPU ölçeklemeyi tek seferde ölçer. PyTorch + llama.cpp üzerine kurulu — Docker gerektirmez.

### Hızlı Başlangıç

#### 1. Klonla ve Kur

```bash
git clone <repo-url>
cd GPUDLBENCH
python3 install.py
```

Kurulum otomatik olarak sanal ortam oluşturur, GPU'nuza uygun PyTorch'u yükler, llama.cpp'yi derler ve test modellerini indirir (~57 GB).

#### 2. Ortamı Aktifleştir

```bash
source venv/bin/activate
```

#### 3. Benchmark'ları Çalıştır

```bash
python run_benchmarks.py
```

Tüm testleri sırayla çalıştırır (~2–4 saat). Sonuçlar `results/` altında zaman damgalı bir klasöre kaydedilir.

#### 4. Karşılaştırma Grafikleri Oluştur

Birden fazla GPU'yu test ettikten sonra (ayrı oturumlarda) GPU karşılaştırma grafikleri oluşturun:

```bash
python utils/generate_comparison.py --lang tr     # Türkçe
python utils/generate_comparison.py               # İngilizce (varsayılan)
```

Grafikler `results/comparison_charts/` klasörüne kaydedilir.

### Ne Test Ediliyor?

| # | Benchmark | Ne Ölçüyor |
|---|-----------|------------|
| 1 | CNN Eğitimi (ResNet-50/101) | Eğitim hızı (görüntü/sn) |
| 2 | Transformer Eğitimi (BERT-Base/Large) | Eğitim hızı (örnek/sn) |
| 3 | CNN Çıkarımı (ResNet-50/101) | Çıkarım gecikmesi ve hızı |
| 4 | Transformer Çıkarımı (BERT-Base/Large) | NLP çıkarım hızı |
| 5 | LLM Token Üretimi (llama.cpp) | Token üretim hızı (token/sn) |
| 6 | VRAM Kapasite Testi | En büyük yüklenebilir model, maks. bağlam uzunluğu |
| 7 | GEMM Hesaplama Stresi | FP64/FP32/FP16/BF16/FP8 zirve TFLOPS |
| 8 | Nesne Algılama Eğitimi (Faster/Mask R-CNN) | Algılama modeli eğitim hızı |
| 9 | GPU Temelleri | Bellek bant genişliği, PCIe, gecikme, FFT |
| 10 | Çift-GPU Ölçekleme (DDP + FSDP) | 2 GPU ile ölçekleme verimi |

### Gereksinimler

- **Linux** (Ubuntu 20.04+ önerilir)
- **NVIDIA GPU** — CUDA 11.8+
- **Python 3.9+**
- **16 GB RAM** minimum
- **İnternet bağlantısı** (ilk model indirmeleri için)

### Çalıştırma Seçenekleri

```bash
python run_benchmarks.py --demo              # ~5 dk hızlı test
python run_benchmarks.py --skip 5 6 10       # belirli testleri atla
python benchmarks/1_training_vision.py       # tek bir benchmark çalıştır
```

### Kurulum Seçenekleri

```bash
python3 install.py --skip-llama        # llama.cpp'yi atla
python3 install.py --skip-models       # model indirmeyi atla
python3 install.py --model-set popular # daha küçük model seti (~38 GB)
```

### Sonuçlar ve Raporlar

Her çalışma `results/` altında zaman damgalı bir klasör oluşturur.

```bash
python utils/generate_report.py                   # oturum özet raporu
python utils/generate_comparison.py --lang tr      # Türkçe karşılaştırma grafikleri
python utils/generate_comparison.py --skip-charts  # sadece metrik, grafik yok
```

### Çift-GPU Kurulumu

Sisteminizde 2 aynı GPU varsa, benchmark 10 (çift-GPU ölçekleme) otomatik çalışır. Farklı GPU'larınız varsa hangisini test edeceğiniz sorulur ve çift-GPU testi atlanır.

### Sorun Giderme

**"no kernel image is available"** → Kurulum tekrar çalıştırıldığında GPU'nuza uygun PyTorch otomatik yüklenir:
```bash
python3 install.py
```

**GPU değiştirdiniz mi?** → Kurulumu tekrar çalıştırın. Yeni kartı algılar, PyTorch'u günceller ve llama.cpp'yi yeniden derler.

> Detaylı benchmark açıklamaları için → [BENCHMARKS_TR.md](BENCHMARKS_TR.md)

---

<a name="english"></a>
## 🇬🇧 English

AI-focused benchmark suite for NVIDIA GPUs. Measures training throughput, inference speed, LLM token generation, VRAM capacity, and dual-GPU scaling in a single run. Built on PyTorch + llama.cpp — no Docker required.

### Quick Start

#### 1. Clone & Install

```bash
git clone <repo-url>
cd GPUDLBENCH
python3 install.py
```

The installer automatically creates a virtual environment, installs the correct PyTorch for your GPU, builds llama.cpp, and downloads test models (~57 GB).

#### 2. Activate the Environment

```bash
source venv/bin/activate
```

#### 3. Run Benchmarks

```bash
python run_benchmarks.py
```

Runs all benchmarks sequentially (~2–4 hours). Results are saved to a timestamped folder under `results/`.

#### 4. Generate Comparison Charts

After benchmarking multiple GPUs (in separate sessions), generate cross-GPU comparison charts:

```bash
python utils/generate_comparison.py
```

Charts are saved to `results/comparison_charts/`. English is the default language; use `--lang tr` for Turkish.

### What's Being Tested?

| # | Benchmark | What It Measures |
|---|-----------|-----------------|
| 1 | CNN Training (ResNet-50/101) | Training throughput (images/sec) |
| 2 | Transformer Training (BERT-Base/Large) | Training throughput (samples/sec) |
| 3 | CNN Inference (ResNet-50/101) | Inference latency & throughput |
| 4 | Transformer Inference (BERT-Base/Large) | NLP inference throughput |
| 5 | LLM Token Generation (llama.cpp) | Token generation speed (tokens/sec) |
| 6 | VRAM Capacity Test | Largest loadable model, max context length |
| 7 | GEMM Compute Stress | Peak TFLOPS (FP64/FP32/FP16/BF16/FP8) |
| 8 | Object Detection Training (Faster/Mask R-CNN) | Detection model training throughput |
| 9 | GPU Fundamentals | Memory BW, PCIe, latency, FFT |
| 10 | Dual-GPU Scaling (DDP + FSDP) | Scaling efficiency with 2 identical GPUs |

### Requirements

- **Linux** (Ubuntu 20.04+ recommended)
- **NVIDIA GPU** with CUDA 11.8+
- **Python 3.9+**
- **16 GB RAM** minimum
- **Internet connection** (for initial model downloads)

### Run Options

```bash
python run_benchmarks.py --demo              # ~5-min smoke test
python run_benchmarks.py --skip 5 6 10       # skip specific benchmarks
python benchmarks/1_training_vision.py       # run a single benchmark
```

### Install Options

```bash
python3 install.py --skip-llama        # skip llama.cpp build
python3 install.py --skip-models       # skip model downloads
python3 install.py --model-set popular # smaller model set (~38 GB)
```

### Results & Reports

Each run creates a timestamped folder under `results/`.

```bash
python utils/generate_report.py                   # per-session summary
python utils/generate_comparison.py               # English comparison charts
python utils/generate_comparison.py --lang tr      # Turkish
python utils/generate_comparison.py --skip-charts  # metrics only, no PNGs
```

### Dual-GPU Setup

If your system has 2 identical GPUs, benchmark 10 (dual-GPU scaling) runs automatically. If your GPUs are different, you'll be prompted to choose which one to benchmark and the dual-GPU test is skipped.

### Troubleshooting

**"no kernel image is available"** → Re-run the installer. It auto-detects your GPU and installs the correct PyTorch build:
```bash
python3 install.py
```

**Switched GPU?** → Re-run the installer. It detects the new card, updates PyTorch, and rebuilds llama.cpp.

### HuggingFace Token (Optional)

Some gated models require a HuggingFace token. Create a `.credentials` file:

```
HF_TOKEN=hf_yourTokenHere
```

Or pass directly: `python3 install.py --hf-token hf_yourTokenHere`

Get a token at https://huggingface.co/settings/tokens (Read access is sufficient). The suite works without a token — gated model downloads will simply be skipped.

---

## License

MIT
