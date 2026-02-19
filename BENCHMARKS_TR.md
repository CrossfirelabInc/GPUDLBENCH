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

---

## 2. NLP Eğitimi (Training NLP) — BERT-Base / BERT-Large

**Ne yapıyor?**
Doğal dil işleme (NLP) modelleri olan BERT-base ve BERT-large, GPU üzerinde eğitiliyor. Saniyede kaç metin örneği işlendiği ölçülüyor.

**Gerçek hayatta ne anlama geliyor?**
- ChatGPT benzeri modellerin temelini oluşturan dil modellerini eğitmek istiyorsanız bu test önemli.
- Duygu analizi, metin sınıflandırma, soru-cevap sistemi gibi projeler bu tür eğitim kullanır.
- BERT-large, BERT-base'den **3 kat** daha büyük — büyük modellerde GPU farkı daha belirgin olur.
- Araştırmacılar ve şirketler bu sonuçlara bakarak GPU yatırım kararı verir.

---

## 3. Görüntü Çıkarımı (Inference Vision) — ResNet-50 / ResNet-101

**Ne yapıyor?**
Önceden eğitilmiş görüntü modelleri ile tahmin (inference) yapılıyor. Saniyede kaç fotoğrafın sınıflandırılabildiği ölçülüyor.

**Gerçek hayatta ne anlama geliyor?**
- Güvenlik kamerası analizi, otomatik araç plaka tanıma, fabrika kalite kontrol gibi **gerçek zamanlı** uygulamalarda bu sayı kritiktir.
- 1000 img/s = saniyede 1000 fotoğrafı sınıflandırabilme. Bir güvenlik kamerası 30 FPS yayın yapar, yani 1000 img/s ile **33 kamerayı aynı anda** analiz edebilirsiniz.
- Web servisleri için: her API isteğinde bir fotoğraf sınıflandırılır, yüksek throughput = daha fazla eşzamanlı kullanıcı.

---

## 4. NLP Çıkarımı (Inference NLP) — BERT-Base / BERT-Large

**Ne yapıyor?**
Eğitilmiş NLP modelleriyle saniyede kaç metin örneğinin işlenebileceği ölçülüyor.

**Gerçek hayatta ne anlama geliyor?**
- Chatbot, arama motoru, otomatik çeviri, spam filtresi gibi servislerde her kullanıcı isteği bir inference çağrısıdır.
- 500 sample/s = saniyede 500 metin analizi. Örneğin bir e-ticaret sitesinde ürün yorumlarının anlık duygu analizi.
- Yüksek inference hızı = daha düşük sunucu maliyeti. Aynı GPU ile daha fazla kullanıcıya hizmet verebilirsiniz.

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
Aynı iş yükünü 1 GPU ve 2 GPU ile çalıştırarak, ikinci GPU'nun ne kadar fayda sağladığını ölçüyor. Ölçekleme verimi (%) hesaplanıyor.

**Gerçek hayatta ne anlama geliyor?**
- **İkinci bir GPU almaya değer mi?** Bu test tam olarak bunu gösterir.
- %100 verim = 2 GPU, tam olarak 2 kat hız. Gerçekte bu neredeyse imkansızdır çünkü GPU'lar arası haberleşme zaman alır.
- %70-85 arası verim = iyi ölçekleme. İkinci GPU'dan büyük fayda var.
- %50 altı = zayıf ölçekleme. İkinci GPU parayı tam hak etmiyor.
- **Eğitim genellikle iyi ölçeklenir**, çıkarım (inference) genellikle daha zor ölçeklenir çünkü batch size küçüktür.
- DataParallel (tek makine, tek işlem) kullanılır — en yaygın çoklu GPU senaryosu.

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

## Sonuçları Nasıl Okumalı?

1. **Eğitim testleri** → "Bu GPU ile modelimi ne kadar hızlı eğitirim?"
2. **Çıkarım testleri** → "Eğitilmiş modeli kaç kullanıcıya/kameraya aynı anda servis edebilirim?"
3. **LLM testi** → "Yerel ChatGPT deneyimim ne kadar akıcı olur?"
4. **VRAM testi** → "Ne kadar büyük bir model çalıştırabilirim?"
5. **GEMM testi** → "GPU'nun saf hesaplama gücü ne?"
6. **Watt/Performans** → "Aynı performansı ne kadar az enerjiyle elde ediyorum?"
7. **Görece performans** → "Bu GPU, en zayıf rakibinden kaç kat hızlı?"
8. **Çift GPU testi** → "İkinci GPU gerçekten işe yarıyor mu?"
