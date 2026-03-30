# Feature Matching System

Bu proje, farklı **görüntü özellik çıkarımı ve eşleştirme** yaklaşımlarını karşılaştırmak için hazırlanmış bir benchmark ve deney ortamıdır. Depoda ORB, ALIKE, ALIKED + LightGlue ve XFeat tabanlı eşleştirme scriptleri yer almaktadır. Proje yapısında `data/` klasörü altında örnek giriş görselleri, `scripts/` klasörü altında ise karşılaştırma scriptleri bulunmaktadır. Ayrıca depoda örnek bir çıktı görseli olarak `sp_lightglue_matches.png` dosyası da bulunmaktadır. :contentReference[oaicite:0]{index=0}

## Projenin Amacı

Bu çalışmanın amacı, farklı feature matching yöntemlerini hem **doğruluk benzeri kalite ölçümleri** hem de **çalışma süresi / frekans** açısından karşılaştırmaktır. Scriptlerde eşleşme kalitesi için homography tabanlı RANSAC analizi kullanılır; inlier sayısı, inlier ratio ve reprojection error gibi metrikler hesaplanır. Performans tarafında ise extraction, matching ve toplam süre ölçülerek ortalama çalışma frekansı elde edilir. :contentReference[oaicite:1]{index=1}

## İçerik

### Klasör Yapısı

```bash
feature-matching-system/
├── data/
│   ├── 000100.png
│   ├── 000108.png
│   ├── img1.png
│   ├── img2.png
│   ├── ref.png
│   ├── sacre_coeur1.jpg
│   ├── sacre_coeur2.jpg
│   └── tgt.png
├── scripts/
│   ├── alike_benchmark.py
│   ├── aliked_lightglue_benchmark.py
│   ├── orb_test.py
│   └── xfeat_benchmark.py
├── sp_lightglue_matches.png
└── .gitignore
```

## Kullanılan Yöntemler

**1. ORB + BFMatcher**

orb_test.py scripti OpenCV tabanlı klasik bir yaklaşım kullanır. ORB ile keypoint ve descriptor çıkarımı yapılır, ardından BFMatcher ile eşleşmeler bulunur. Sonuçlar RANSAC ile filtrelenir ve inlier eşleşmeler görsel olarak kaydedilir. Script içinde benchmark parametreleri arasında NFEATURES=500, WARMUP_RUNS=5, BENCH_RUNS=50 ve RANSAC_THRESH=3.0 yer almaktadır. Çıktı dosyası results/orb_matches_ransac.png olarak üretilir.

**2. ALIKE Benchmark**

alike_benchmark.py scripti ALIKE tabanlı feature extraction yapar. Kod içinde ALIKE deposu harici bir klasörden içe aktarılmaktadır ve model olarak alike-t kullanılmıştır. Benchmark tarafında RESIZE_SCALE=0.75, DEVICE="cpu", SCORES_TH=0.35, N_LIMIT=200, SIM_TH=0.90, RUNS=10 gibi ayarlar bulunmaktadır. Elde edilen eşleşmeler için yine RANSAC tabanlı kalite ölçümü yapılır ve çıktı results/alike_matches_optimized.png olarak kaydedilir.

**3. ALIKED + LightGlue**

aliked_lightglue_benchmark.py scripti, ALIKED çıkarıcı ile LightGlue eşleştiriciyi birlikte kullanır. Kodda TOP_K=200, RESIZE=768, RUNS=10 gibi parametreler tanımlanmıştır. Girdi görselleri LightGlue örnek varlıklarından alınmaktadır. Script, eşleşme skorlarını, inlier oranını ve ortalama çalışma süresini raporlar; sonuç görselini results/aliked_lightglue_matches.png dosyasına yazar.

**4. XFeat Benchmark**

xfeat_benchmark.py scripti XFeat yaklaşımını kullanır. Kodda XFeat modülü harici bir repodan içe aktarılır. TOP_K=200, RESIZE_SCALE=0.75, RUNS=10 ve RANSAC_THRESH=3.0 gibi ayarlar bulunmaktadır. Eşleşme kalitesi yine homography + RANSAC üzerinden değerlendirilir ve çıktı results/xfeat_matches_v2.png olarak kaydedilir.

## Ölçülen Metrikler

Projede scriptler genel olarak aşağıdaki metrikleri raporlamaktadır:

- Keypoint sayısı
- Toplam match sayısı
- Gösterilen inlier match sayısı
- Minimum / maksimum / ortalama confidence
- Homography bulunup bulunmadığı
- Inlier sayısı
- Inlier ratio
- Reprojection error
- Ortalama extraction süresi
- Ortalama matching süresi
- Ortalama toplam süre
- Ortalama çalışma frekansı (Hz)

Bu metrikler scriptlerin çıktı bölümlerinde açıkça hesaplanıp yazdırılmaktadır.

## Gereksinimler

Projede kullanılan scriptlere göre temel olarak aşağıdaki Python kütüphanelerine ihtiyaç vardır:

- Python 3
- OpenCV
- NumPy
- PyTorch

Ayrıca bazı scriptler harici repo bağımlılıkları kullanmaktadır:

- ALIKE
- LightGlue
- accelerated_features / XFeat

Kod içinde bu bağımlılıkların yerel klasörlerden sys.path.append(...) ile eklendiği görülmektedir. Bu nedenle scriptleri çalıştırmadan önce ilgili repoların uygun klasör yapısında indirilmiş olması gerekir.


## Kurulum

Örnek kurulum adımları:

```bash
git clone https://github.com/merveakbey/feature-matching-system.git
cd feature-matching-system

python -m venv venv
source venv/bin/activate
pip install opencv-python numpy torch
```

Not: ALIKE, LightGlue ve XFeat tabanlı scriptler için ilgili harici repoların da ayrıca kurulması gerekir. Kod yapısında bu repoların ~/feature_matching_task/repos/... altında tutulduğu görülmektedir. Bu nedenle kendi sisteminizde klasör yollarını scriptlere göre düzenlemeniz gerekebilir.

## Çalışma Mantığı

Genel akış şu şekildedir:

- İki giriş görüntüsü yüklenir.
- Seçilen yöntem ile keypoint ve descriptor çıkarılır.
- Eşleşmeler bulunur.
- RANSAC ile geometrik doğrulama yapılır.
- Inlier eşleşmeler görsel olarak çizdirilir.
- Süre ve kalite metrikleri hesaplanır.
- Sonuçlar terminale ve çıktı görsellerine kaydedilir.

Bu yapı sayesinde klasik ve öğrenme tabanlı feature matching yöntemleri aynı mantıkta değerlendirilebilir.
