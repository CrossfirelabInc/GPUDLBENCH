# GPU Yapay Zeka Benchmark Testleri — Açıklamalar

> Bu belge, Crossfirelab GPU Benchmark paketindeki her testin ne yaptığını, neyi ölçtüğünü ve gerçek hayatta ne anlama geldiğini sade bir dille açıklar. YouTube izleyicileri ve teknoloji meraklıları için hazırlanmıştır.

---

## 1. Görüntü Eğitimi (Training Vision) — ResNet-50 / ResNet-101

**Ne yapıyor?**
Fotoğraf sınıflandırma modelleri (ResNet-50 ve ResNet-101) GPU üzerinde eğitiliyor. FP32, FP16 ve BF16 hassasiyetlerinde saniyede kaç görüntü işlendiği ölçülüyor.

**Gerçek hayatta ne anlama geliyor?**
- Kendi fotoğraf sınıflandırma modelinizi (örneğin kedi/köpek ayırma, tıbbi görüntü analizi, ürün tanıma) eğitmek istiyorsanız, bu test GPU'nuzun bu işi ne kadar hızlı yapacağını gösterir.
- Yüksek sonuç = daha kısa eğitim süresi. Örneğin 500 img/s ile 1000 img/s arasındaki fark, 10 saatlik bir eğitimi 5 saate düşürmek demektir.
- FP16/BF16 sonuçları, "mixed precision" eğitim ile ne kadar hız kazandığınızı gösterir — modern GPU'larda bu **2-3 kat** hız farkı yaratabilir.
- Karşılaştırma grafiklerinde her GPU'nun **en yüksek throughput'u** ve **aynı batch boyutu** karşılaştırması ayrı ayrı gösterilir.

---

## 2. NLP Eğitimi (Training NLP) — BERT-Base / BERT-Large

**Ne yapıyor?**
Doğal dil işleme (NLP) modelleri olan BERT-base ve BERT-large, GPU üzerinde eğitiliyor. Saniyede kaç metin örneği işlendiği ölçülüyor.

**Gerçek hayatta ne anlama geliyor?**
- ChatGPT benzeri modellerin temelini oluşturan dil modellerini eğitmek istiyorsanız bu test önemli.
- Duygu analizi, metin sınıflandırma, soru-cevap sistemi gibi projeler bu tür eğitim kullanır.
- BERT-large, BERT-base'den **3 kat** daha büyük — büyük modellerde GPU farkı daha belirgin olur.
- Araştırmacılar ve şirketler bu sonuçlara bakarak GPU yatırım kararı verir.
- Karşılaştırma grafiklerinde **maksimum verim** ve **aynı batch boyutu** grafikleri ayrı sunulur.

---

## 3. Görüntü Çıkarımı (Inference Vision) — ResNet-50 / ResNet-101

**Ne yapıyor?**
Önceden eğitilmiş görüntü modelleri ile tahmin (inference) yapılıyor. Saniyede kaç fotoğrafın sınıflandırılabildiği ölçülüyor.

**Gerçek hayatta ne anlama geliyor?**
- Güvenlik kamerası analizi, otomatik araç plaka tanıma, fabrika kalite kontrol gibi **gerçek zamanlı** uygulamalarda bu sayı kritiktir.
- 1000 img/s = saniyede 1000 fotoğrafı sınıflandırabilme. Bir güvenlik kamerası 30 FPS yayın yapar, yani 1000 img/s ile **33 kamerayı aynı anda** analiz edebilirsiniz.
- Web servisleri için: her API isteğinde bir fotoğraf sınıflandırılır, yüksek throughput = daha fazla eşzamanlı kullanıcı.
- Karşılaştırmada **en iyi batch boyutundaki zirve throughput** grafikleri mevcuttur.

---

## 4. NLP Çıkarımı (Inference NLP) — BERT-Base / BERT-Large

**Ne yapıyor?**
Eğitilmiş NLP modelleriyle saniyede kaç metin örneğinin işlenebileceği ölçülüyor.

**Gerçek hayatta ne anlama geliyor?**
- Chatbot, arama motoru, otomatik çeviri, spam filtresi gibi servislerde her kullanıcı isteği bir inference çağrısıdır.
- 500 sample/s = saniyede 500 metin analizi. Örneğin bir e-ticaret sitesinde ürün yorumlarının anlık duygu analizi.
- Yüksek inference hızı = daha düşük sunucu maliyeti. Aynı GPU ile daha fazla kullanıcıya hizmet verebilirsiniz.
- GPU'lar arası farkın VRAM mı yoksa saf hesaplama gücünden mi geldiğini görmek için grafiklerdeki **BS=** etiketlerine dikkat edin.

---

## 5. LLM Token Üretimi (LLM Tokens per Second)

**Ne yapıyor?**
Büyük dil modelleri (DeepSeek, Phi-4, Gemma, QwQ gibi) llama.cpp ile GPU üzerinde çalıştırılıyor. Saniyede kaç token (kelime parçası) üretildiği ve ilk tokenin ne kadar sürede geldiği ölçülüyor.

**Gerçek hayatta ne anlama geliyor?**
- **Bu, ChatGPT/Claude gibi yapay zeka asistanlarının yerel bilgisayarınızda ne kadar hızlı çalışacağını gösterir.**
- 50 token/s ≈ saniyede ~35 kelime = akıcı bir sohbet deneyimi.
- 10 token/s = kelime kelime gelen, yavaş bir deneyim.
- İlk token süresi (TTFT) = soruyu sorduktan sonra ilk cevabın gelmesi için bekleme süresi. 100ms altı idealdir.
- Büyük modeller (27B, 32B) daha akıllı ama daha yavaş — GPU gücü burada belirleyici.

---

## 6. VRAM Limitleri (VRAM Limits)

**Ne yapıyor?**
GPU belleğine (VRAM) sığan en büyük modeli ve en uzun bağlam uzunluğunu (context length) test ediyor.

**Gerçek hayatta ne anlama geliyor?**
- **VRAM = GPU'nuzun RAM'i.** Daha fazla VRAM = daha büyük ve daha akıllı modelleri çalıştırabilme.
- 24GB VRAM ≈ 13B parametreli model (Llama-2-13B seviyesi)
- 48GB VRAM ≈ 30B+ parametreli model
- Bağlam uzunluğu = modelin aynı anda kaç kelimeyi "hatırlayabileceği". Uzun doküman analizi, kitap özetleme gibi işler uzun bağlam gerektirir.
- **Eğer bir model VRAM'e sığmıyorsa, ya daha küçük model kullanmanız ya da quantization (sıkıştırma) yapmanız gerekir.**

---

## 7. GEMM Hesaplama Stresi (GEMM Compute Stress)

**Ne yapıyor?**
GPU'nun saf matematik hesaplama gücünü ölçüyor. Büyük matris çarpımları (GEMM = General Matrix Multiply) yaparak her hassasiyet türünde (FP64, FP32, FP16, BF16, FP8) zirve TFLOPS değerini buluyor.

**Gerçek hayatta ne anlama geliyor?**
- **Her yapay zeka modeli temelde matris çarpımıdır.** Bu test, GPU'nun teorik maksimum hesaplama kapasitesini gösterir.
- FP32 TFLOPS = bilimsel simülasyon, mühendislik yazılımları için kritik.
- BF16/FP16 TFLOPS = yapay zeka eğitim/çıkarım hızının temel belirleyicisi.
- FP8 = yeni nesil GPU'larda 2x daha hızlı çıkarım potansiyeli.
- Bu test, GPU'nun "kas gücünü" ölçer — yazılım optimizasyonundan bağımsız, saf donanım performansı.

---

## 8. Nesne Algılama Eğitimi (Object Detection) — Faster R-CNN / Mask R-CNN

**Ne yapıyor?**
Fotoğraftaki nesneleri tespit eden ve konumlarını belirleyen modeller (Faster R-CNN, Mask R-CNN) eğitiliyor.

**Gerçek hayatta ne anlama geliyor?**
- Otonom araç teknolojisi: yayaları, araçları, tabelaları tespit etme.
- Tıbbi görüntüleme: X-ray/MRI'da tümör tespiti.
- Güvenlik: kamera görüntülerinde şüpheli nesne/kişi algılama.
- Perakende: raf analizi, müşteri sayma.
- **ResNet sınıflandırmasından çok daha ağır bir iş yükü** — GPU belleği ve hesaplama gücü aynı anda test edilir.
- Karşılaştırmada **maksimum verim** ve **aynı batch boyutu** grafikleri ayrı sunulur.

---

## 9. GPU Temelleri (GPU Fundamentals)

**Ne yapıyor?**
GPU'nun alt seviye donanım performansını ölçüyor:
- **Bellek bant genişliği:** GPU belleğinden veri okuma/yazma hızı (GB/s)
- **PCIe bant genişliği:** CPU ↔ GPU arası veri transfer hızı
- **FFT:** Hızlı Fourier Dönüşümü performansı
- **Kernel başlatma gecikmesi:** GPU'ya komut gönderme hızı
- **SpMM:** Seyrek matris çarpımı

**Gerçek hayatta ne anlama geliyor?**
- **Bellek bant genişliği**, büyük modellerde "bottleneck" (darboğaz) oluşturur. Yüksek BW = daha hızlı model yükleme ve çıkarım.
- **PCIe hızı**, CPU'dan GPU'ya veri aktarımını etkiler. Veri pipeline'ı yavaşsa, GPU boşa bekler.
- **Kernel gecikmesi**, küçük işlemlerde (örneğin gerçek zamanlı inference) önemlidir — düşük gecikme = daha duyarlı sistem.
- Bu testler, GPU'nun "temel sağlık kontrolü" gibidir. Kartın üretici spesifikasyonlarına ulaşıp ulaşmadığını gösterir.

---

## 10. Çoklu GPU Ölçekleme (Multi-GPU Scaling)

**Ne yapıyor?**
Aynı iş yükünü 1 GPU ve 2 GPU ile çalıştırarak, ikinci GPU'nun ne kadar fayda sağladığını ölçüyor. Üç farklı model test edilir: ResNet-50 (görüntü/CNN), BERT-base (NLP encoder) ve GPT-2 Large (~774M parametre, LLM decoder). Her model için tek GPU baseline, DDP ve FSDP ZeRO-2 yöntemleri karşılaştırılır. Ölçekleme verimi (%) = 2 GPU throughput / (2 × 1 GPU throughput) × 100 formülüyle hesaplanır.

**Gerçek hayatta ne anlama geliyor?**
- **İkinci bir GPU almaya değer mi?** Bu test tam olarak bunu gösterir.
- %100 verim = 2 GPU, tam olarak 2 kat hız. Gerçekte bu neredeyse imkansızdır çünkü GPU'lar arası haberleşme zaman alır.
- %85-95 arası verim = çok iyi ölçekleme. İkinci GPU yatırımınıza değer.
- %70-85 arası verim = iyi ölçekleme. İkinci GPU'dan büyük fayda var.
- %50 altı = zayıf ölçekleme. İkinci GPU parayı tam hak etmiyor.
- **Eğitim genellikle iyi ölçeklenir**, çıkarım (inference) genellikle daha zor ölçeklenir çünkü batch size küçüktür.

### Kullanılan Yöntemler

**DDP (DistributedDataParallel)**
Her GPU'da modelin tam bir kopyası bulunur. Her GPU kendi batch'ini işler, sonra gradyanlar NCCL all-reduce ile toplanır ve senkronize edilir. En yaygın ve en basit çoklu GPU eğitim yöntemidir. Ek VRAM tasarrufu sağlamaz — her GPU, modelin tamamını ve optimizer state'lerini ayrı ayrı tutar. Küçük-orta modellerde (%95+ verimlilik ile) en iyi performansı verir.

**FSDP ZeRO-2 (Fully Sharded Data Parallel — SHARD_GRAD_OP)**
Microsoft DeepSpeed ZeRO Stage-2'nin PyTorch-native karşılığıdır. DDP'den farklı olarak gradyanları ve optimizer state'lerini GPU'lar arasında parçalar (shard). Bu sayede her GPU, bu verilerin yalnızca 1/N'ini tutar — VRAM tasarrufu sağlar ve daha büyük modellerin/batch'lerin sığmasını mümkün kılar. Model parametreleri forward ve backward arasında her GPU'da tam olarak tutulur (ZeRO-3'ten farkı budur). Büyük modellerde (GPT-2 Large gibi) VRAM verimliliği sayesinde DDP'den daha iyi ölçeklenebilir.

---

## Hassasiyet Türleri Nedir?

| Kısaltma | Açıklama | Kullanım Alanı |
|----------|----------|----------------|
| **FP32** | 32-bit kayan nokta (tam hassasiyet) | Bilimsel hesaplama, eğitim referansı |
| **FP16** | 16-bit kayan nokta (yarı hassasiyet) | Eğitim/çıkarım hızlandırma |
| **BF16** | Brain Float 16 (Google formatı) | Modern AI eğitimi (daha geniş değer aralığı) |
| **FP8** | 8-bit kayan nokta | Yeni nesil GPU'larda ultra-hızlı çıkarım |
| **FP64** | 64-bit çift hassasiyet | Bilimsel simülasyon, HPC |

> **Genel kural:** Düşük hassasiyet = daha hızlı ama potansiyel olarak daha az doğru. Modern AI'da FP16/BF16 "altın standart" olarak kabul edilir — hız kazancı büyük, doğruluk kaybı ihmal edilebilir düzeydedir.

---

## Karşılaştırma Grafikleri: Maksimum Verim ve Aynı Batch Boyutu

Benchmark testleri 1-4 ve 8 (görüntü/NLP eğitim, çıkarım ve nesne algılama) birden fazla batch boyutunda çalışır. Her GPU, destekleyebildiği en büyük batch boyutuna kadar otomatik olarak ölçeklenir. Bu nedenle farklı VRAM kapasitesine sahip GPU'lar farklı batch boyutlarına ulaşabilir.

Karşılaştırma grafiklerinde bu durum **iki ayrı grafik türü** ile gösterilir:

### Maksimum Verim (Max Throughput)
- Her GPU'nun **en yüksek throughput'u** hangi batch boyutunda elde edildiyse o değer gösterilir.
- Grafik etiketlerinde `(BS=64)` gibi açıklamalar yer alır — hangi batch boyutunun kullanıldığı görülür.
- **Soru:** "Bu GPU pratikte en iyi performansını hangi batch boyutunda veriyor?"
- Eğitim ve çıkarım testlerinin tamamı için mevcuttur.

### Aynı Batch Boyutu (Same Batch Size)
- Tüm GPU'ların çalıştırabildiği **en küçük ortak batch boyutu** seçilir ve tüm GPU'ların o batch boyutundaki throughput'ları karşılaştırılır.
- Bu sayede VRAM avantajı tamamen devre dışı kalır: her GPU aynı iş yükünü işler.
- Grafik etiketlerinde `BS=16` gibi açıklamalar yer alır — karşılaştırmanın hangi batch boyutunda yapıldığı görülür.
- **Soru:** "Aynı iş yükünde hangi GPU daha hızlı?"
- Saf hesaplama gücü karşılaştırması için idealdir.

> **Hangisine bakmalı?**
> - Gerçek dünya performansı için → **Maksimum Verim** grafiği (pratikte her GPU en uygun batch boyutunda çalışır)
> - Saf GPU hesaplama gücü için → **Aynı Batch Boyutu** grafiği (VRAM avantajı devre dışı, eşit iş yükü)

---

## GPU Skor Kartı (GPU Scorecard)

Karşılaştırma grafiklerinin en sonunda yer alan **GPU Scorecard** tablosu, tüm benchmark sonuçlarını tek bir bakışta özetler. Her satır bir metriği, her sütun bir GPU'yu temsil eder. Her kategoride en iyi sonucu alan GPU yeşil çerçeve ve ★ BEST etiketi ile vurgulanır.

Skor kartında gösterilen başlıca metrikler:
- **Eğitim/çıkarım throughput** — en yüksek değerler
- **LLM hızı** — en iyi modelin token/s değeri
- **VRAM kapasitesi** — en büyük yüklenebilir model ve bağlam uzunluğu
- **LLMFit önerileri** — VRAM'e göre en iyi, en büyük Dense ve en büyük MoE LLM önerileri (aşağıda açıklanmıştır)
- **Güç tüketimi & sıcaklık** — düşük olan daha iyi

---

## Dense vs MoE (Mixture of Experts) Model Mimarisi

Scorecard'da "En Büyük Dense LLM" ve "En Büyük MoE LLM" satırları yer alır. Bu iki mimari arasındaki fark, modern yapay zekada GPU seçimini doğrudan etkiler:

### Dense Model Nedir?
Geleneksel derin öğrenme mimarisidir. Modeldeki **tüm parametreler her token için aktif olarak** kullanılır.

- Örnek: Llama-3.1-70B, Qwen2.5-72B, Falcon-40B
- 70B Dense model = her token üretilirken 70 milyar parametre hesaplanır
- **Avantaj:** Basit mimari, öngörülebilir performans, düşük gecikme
- **Dezavantaj:** VRAM ihtiyacı yüksek — 70B Q4 ≈ ~40 GB VRAM gerektirir
- **VRAM kuralı:** Tüm parametreler GPU belleğine yüklenmelidir

### MoE (Mixture of Experts) Model Nedir?
Modelin toplam parametreleri çok büyüktür (örneğin 671B), ancak her token için yalnızca **küçük bir "uzman" alt kümesi** aktif olur.

- Örnek: DeepSeek-R1 (671B toplam, ~37B aktif), Llama-4-Maverick (400B toplam, ~17B aktif), Qwen3-235B-A22B (235B toplam, ~22B aktif)
- **Avantaj:** Devasa bilgi kapasitesi çok daha az hesaplama ile kullanılır. 671B MoE model, hız olarak ~37B Dense model gibi çalışır.
- **Dezavantaj:** Toplam parametre sayısı VRAM'e sığmalıdır (veya kısmen RAM'e "offload" edilmelidir). Routing mekanizması ek gecikme ekler.
- **VRAM kuralı:** Aktif uzman parametreleri GPU belleğinde tutulur, geri kalanı RAM'e aktarılabilir (ancak bu durumda hız düşer)

### Pratikte Ne Anlama Geliyor?

| Özellik | Dense (örn. 70B) | MoE (örn. DeepSeek-R1 671B) |
|---------|-------------------|------------------------------|
| Toplam parametre | 70B | 671B |
| Token başına aktif parametre | 70B | ~37B |
| VRAM ihtiyacı (Q4_K_M) | ~40 GB | ~37 GB (aktif) + RAM offload |
| Token/s hızı | Orta | Yüksek (aktif kısım küçük) |
| Bilgi kapasitesi | 70B düzeyinde | 671B düzeyinde (çok daha geniş) |

> **Özet:** MoE modeller, sınırlı VRAM ile devasa parametrelere erişim sağlar. 24 GB VRAM'li bir GPU, Dense mimaride en fazla ~30B model çalıştırabilirken, MoE mimaride 400B parametreli bir modelin aktif kısmını çalıştırabilir. Ancak toplam model boyutu yine de sistem RAM'ine sığmalıdır.

---

## Sonuçları Nasıl Okumalı?

1. **Eğitim testleri** → "Bu GPU ile modelimi ne kadar hızlı eğitirim?"
2. **Çıkarım testleri** → "Eğitilmiş modeli kaç kullanıcıya/kameraya aynı anda servis edebilirim?"
3. **LLM testi** → "Yerel ChatGPT deneyimim ne kadar akıcı olur?"
4. **VRAM testi** → "Ne kadar büyük bir model çalıştırabilirim?"
5. **GEMM testi** → "GPU'nun saf hesaplama gücü ne?"
6. **Watt/Performans** → "Aynı performansı ne kadar az enerjiyle elde ediyorum?"
7. **Görece performans** → "Bu GPU, en zayıf rakibinden kaç kat hızlı?"
8. **Çift GPU testi** → "İkinci GPU gerçekten işe yarıyor mu?"
9. **Maks. verim vs Aynı BS** → "Fark VRAM'den mi yoksa saf güçten mi geliyor?"
10. **Skor kartı** → "Tüm testlerde hangi GPU öne çıkıyor?"
11. **Dense vs MoE** → "VRAM'ime en büyük hangi model sığar?"
