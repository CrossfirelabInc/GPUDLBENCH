# GPU Yapay Zeka Benchmark Paketi

**Crossfirelab** tarafindan gelistirildi — Vibe Coded 🎯

NVIDIA ekran kartlari icin yapay zeka odakli benchmark (performans testi) paketi. Egitim hizi, cikarim hizi, LLM (buyuk dil modeli) token uretimi, VRAM kapasitesi ve cift GPU olcekleme gibi testleri tek komutla calistirir. PyTorch + llama.cpp kullanir, Docker gerektirmez.

---

## Ne Test Ediliyor?

| No | Test | Ne Olcuyor |
|----|------|-------------|
| 1 | CNN Egitim (ResNet-50/101) | Saniyede kac goruntu ile egitim yapilabilir |
| 2 | Transformer Egitim (BERT-Base/Large) | Saniyede kac metin ornegi islenebilir |
| 3 | CNN Cikarim (ResNet-50/101) | Egitilmis modelle saniyede kac tahmin yapilir |
| 4 | Transformer Cikarim (BERT-Base/Large) | NLP modeliyle saniyede kac metin analiz edilir |
| 5 | LLM Token Uretimi (llama.cpp) | Buyuk dil modelleri saniyede kac kelime uretir |
| 6 | VRAM Kapasite Testi | En buyuk hangi model yuklenebilir, maks baglam uzunlugu |
| 8 | Nesne Algilama Egitimi (Faster/Mask R-CNN) | Nesne tespiti modellerinin egitim hizi |
| 10 | Cift GPU Olcekleme (DDP + FSDP) | 2. GPU ne kadar hiz kazandiriyor |

---

## Gereksinimler

- **Linux** (Ubuntu 20.04+ onerilir)
- **NVIDIA ekran karti** (CUDA 11.8 veya uzeri)
- **Python 3.9+**
- **16 GB RAM** minimum
- **Internet baglantisi** (ilk kurulumda model indirme icin)

---

## Kurulum

### 1. Projeyi indirin

```bash
git clone <repo-url>
cd GPUDLBENCH
```

### 2. HuggingFace Token olusturun

Bazi modelleri indirmek icin HuggingFace hesabiniz gerekiyor:

1. https://huggingface.co/settings/tokens adresine gidin
2. Yeni bir token olusturun (Read yetkisi yeterli)
3. `.credentials` dosyasini olusturun:

```bash
nano .credentials
```

Icine sunu yazin:

```
HF_TOKEN=hf_sizinTokenBuraya
```

### 3. Kurulumu baslatin

```bash
python3 install.py
```

Bu komut otomatik olarak:
- Python sanal ortami (venv) olusturur
- GPU'nuza uygun PyTorch versiyonunu kurar
- llama.cpp'yi derler (LLM testi icin)
- Test modellerini indirir (~57 GB)

Kurulum bittikten sonra sanal ortami aktiflestirin:

```bash
source venv/bin/activate
```

### Kurulum secenekleri

```bash
python3 install.py --skip-llama        # llama.cpp kurulumunu atla (test 5 calismaz)
python3 install.py --skip-models       # model indirmeyi atla
python3 install.py --model-set popular # daha kucuk model seti (~38 GB)
```

---

## Testleri Calistirma

### Tum testleri calistir

```bash
python run_benchmarks.py
```

Bu komut sirasiyla tum testleri calistirir ve sonuclari `results/` klasorune kaydeder. Tum testler yaklasik 2-4 saat surer.

### Hizli deneme modu

Ilk kez deniyorsaniz veya hizli bir test yapmak istiyorsaniz:

```bash
python run_benchmarks.py --demo
```

Demo modu yaklasik 5 dakikada biter — tam sonuclar vermez ama her seyin calistigini dogrulamaniz icin idealdir.

### Belirli testleri atlama

```bash
python run_benchmarks.py --skip 5 6 10    # LLM, VRAM, cift GPU testlerini atla
```

### Tek bir testi calistirma

```bash
python benchmarks/1_training_vision.py     # sadece goruntu egitim testi
python benchmarks/5_llm_tokens_per_sec.py  # sadece LLM testi
```

---

## Sonuclari Goruntuleme

Her test calistirmasinda `results/` klasorunde tarih damgali bir klasor olusturulur. En son calistirma `results/latest` kisayolundan erisilebilir.

### Ozet rapor olusturma

```bash
python utils/generate_report.py
```

### GPU karsilastirma grafikleri olusturma

Birden fazla GPU test ettiyseniz (farkli oturumlarda), karsilastirma grafikleri olusturabilirsiniz:

```bash
python utils/generate_comparison.py              # Turkce grafikler (varsayilan)
python utils/generate_comparison.py --lang en     # Ingilizce grafikler
```

Grafikler `results/comparison_charts/` klasorune kaydedilir.

---

## Cift GPU Kurulumu

Sisteminizde 2 ayni GPU varsa test 10 (cift GPU olcekleme) otomatik olarak calisir. Eger GPU'lariniz farkliysa, hangi GPU'yu test etmek istediginiz sorulur ve cift GPU testi atlanir.

---

## Ayarlar

Batch boyutlari, iterasyon sayilari, model listeleri gibi ayarlari degistirmek isterseniz:

```bash
nano benchmarks/config.py
```

---

## Sik Karsilasilan Sorunlar

### "no kernel image is available" hatasi

GPU'nuzun mimarisi icin uygun PyTorch kurulu degil. Cozum:

```bash
python3 install.py
```

Kurulum scripti GPU'nuzu otomatik algilar ve uygun surumu kurar. Blackwell GPU'lar (RTX 5090, 5080 vb.) icin gerekirse nightly surum otomatik denenir.

### Derleme hatalari (`_Float64x`, `CMake` hatalari)

C++ derleyiciniz eski olabilir. Cozum:

```bash
sudo apt-get update
sudo apt-get install -y gcc-13 g++-13
python3 install.py
```

### GPU degisikligi yaptim

Ekran kartini fiziksel olarak degistirdikten sonra:

```bash
python3 install.py
```

Kurulum scripti yeni GPU'yu algilar, PyTorch'u gunceller ve llama.cpp'yi yeniden derler.

---

## Uretilen Grafikler

Karsilastirma araci su grafikleri olusturur:

1. **CNN Egitim Verimi** — ResNet-50/101
2. **Transformer Egitim Verimi** — BERT-Base/Large
3. **CNN Cikarim Verimi** — ResNet-50/101
4. **Transformer Cikarim Verimi** — BERT-Base/Large
5. **LLM Performansi** — Token uretim hizi
6. **Nesne Algilama** — Faster/Mask R-CNN
7. **Watt Basina Performans** — Enerji verimliligi
8. **Gorece Performans** — GPU'lar arasi kat farki
9. **CNN vs Transformer** — Mimari karsilastirma
10. **Cift GPU Olcekleme** — DDP/FSDP hizlanma
11. **GPU Karsilastirma Karti** — Tum sonuclarin ozet tablosu

---

## Lisans

MIT
