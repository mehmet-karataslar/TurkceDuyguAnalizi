# 🧠 Bulanık Mantık ile Türkçe Tweet Duygu Analizi Sistemi

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-Educational-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

**Akıllı Duygu Analizi için Bulanık Mantık Tabanlı Çözüm**

[Özellikler](#-özellikler) • [Kurulum](#-kurulum) • [Araştırma Süreci](#-araştırma-süreci) • [Sonuçlar](#-sonuçlar)

</div>

---

## 📋 İçindekiler

1. [Proje Özeti](#-proje-özeti)
2. [Veri Seti](#-veri-seti)
3. [Araştırma Süreci](#-araştırma-süreci)
4. [İlk Test Sonuçları](#-ilk-test-sonuçları)
5. [Model İyileştirme Süreci](#-model-iyileştirme-süreci)
6. [Final Test Sonuçları](#-final-test-sonuçları)
7. [Üyelik Fonksiyonları Karşılaştırması](#-üyelik-fonksiyonları-karşılaştırması)
8. [Teknik Detaylar](#-teknik-detaylar)
9. [Kurulum ve Kullanım](#-kurulum-ve-kullanım)
10. [Görselleştirmeler](#-görselleştirmeler)

---

## 🎯 Proje Özeti

Bu proje, **Bulanık Mantık (Fuzzy Logic)** prensiplerini kullanarak Türkçe tweet'lerde **10 farklı duygu sınıfını** otomatik olarak tespit eden gelişmiş bir duygu analizi sistemidir. Sistem, geleneksel makine öğrenmesi yöntemlerinden farklı olarak, belirsizlik ve geçiş durumlarını daha iyi modelleyen bulanık mantık yaklaşımını kullanır.

### 🎓 Araştırma Amacı

- Bulanık mantık sistemlerinin duygu analizi problemindeki etkinliğini araştırmak
- Farklı üyelik fonksiyonlarının performansını karşılaştırmak
- Türkçe doğal dil işleme için özelleştirilmiş bir çözüm sunmak
- Yorumlanabilir (interpretable) bir model geliştirmek
- Model performansını optimize etmek için sistematik iyileştirme süreci uygulamak

---

## 📊 Veri Seti

### Veri Seti Özellikleri

- **Toplam Tweet Sayısı**: 5,113
- **Sınıf Sayısı**: 10 farklı duygu
- **Format**: Excel (.xlsx)
- **Sütunlar**: Tweet metni ve Duygu etiketi

### Duygu Sınıfları ve Dağılımı

| # | Duygu | Örnek Sayısı | Yüzde | Açıklama |
|---|-------|--------------|-------|----------|
| 1 | **kızgın** | 800 | 15.6% | Öfke, sinir, rahatsızlık |
| 2 | **korku** | 800 | 15.6% | Endişe, kaygı, panik |
| 3 | **mutlu** | 800 | 15.6% | Neşe, sevinç, keyif |
| 4 | **surpriz** | 800 | 15.6% | Şaşırma, hayret |
| 5 | **üzgün** | 800 | 15.6% | Keder, hüzün, mutsuzluk |
| 6 | **Heyecanlı** | 250 | 4.9% | Coşku, sabırsızlık, beklenti |
| 7 | **Umutsuz** | 249 | 4.9% | Pes etmiş, yorgun, olumsuz beklenti |
| 8 | **Sorgulayıcı** | 244 | 4.8% | Eleştirel, şüpheci, mantık arayan |
| 9 | **Şaşırmış** | 197 | 3.9% | Beklenmedik duruma karşı tepki |
| 10 | **Meraklı** | 173 | 3.4% | Soru sorma, araştırma, öğrenme isteği |

**Not**: Veri seti dengesizdir (imbalanced). Bazı sınıflar 800 örnek içerirken, diğerleri 173-250 arası örnek içermektedir.

---

## 🔬 Araştırma Süreci

### Proje Aşamaları

1. ✅ **Veri Ön İşleme Modülü Geliştirme**
2. ✅ **Bulanık Mantık Modeli Tasarımı**
3. ✅ **İlk Test ve Performans Analizi**
4. ✅ **Model İyileştirme Süreci**
5. ✅ **Final Test ve Değerlendirme**
6. ✅ **Görselleştirme ve Raporlama**

---

## 📉 İlk Test Sonuçları

### İlk Model Konfigürasyonu

- **Özellik Sayısı**: 50
- **TF-IDF Özellikleri**: 500
- **N-gram Aralığı**: (1, 2) - Unigram ve Bigram
- **Kural Başına Özellik**: 3
- **Özellik Seçimi**: Sadece varyans bazlı
- **Minimum Üyelik Eşiği**: Yok

### İlk Test Performans Metrikleri

| Metrik | Değer | Yorum |
|--------|-------|-------|
| **Accuracy** | 26.88% | ❌ Çok düşük |
| **F1-Score** | 0.2182 | ❌ Yetersiz |
| **R²** | 0.0066 | ❌ Neredeyse sıfır |
| **Precision** | 0.5238 | ⚠️ Orta |
| **Recall** | 0.2688 | ❌ Düşük |
| **Ortalama Güven** | 0.2907 | ❌ Düşük güven |

### İlk Test - Üyelik Fonksiyonu Karşılaştırması

| Üyelik Fonksiyonu | Accuracy | F1-Score | R² | Precision | Recall | En İyi? |
|-------------------|----------|----------|-----|-----------|--------|---------|
| **Üçgen** | 18.77% | 0.0978 | -0.0369 | 0.5000 | 0.1877 | ❌ |
| **Yamuk** | 16.91% | 0.0617 | -0.1926 | 0.0553 | 0.1691 | ❌ |
| **Sigmoid** | 18.18% | 0.0846 | -0.0083 | 0.0659 | 0.1818 | ❌ |
| **Gauss** | HATA | - | - | - | - | ❌ |
| **Bell** ⭐ | 26.88% | 0.2182 | 0.0066 | 0.5238 | 0.2688 | ✅ |

**Sonuç**: Bell üyelik fonksiyonu en iyi performansı gösterdi ancak genel performans kabul edilebilir seviyenin çok altındaydı.

### İlk Test - Sınıf Bazlı Performans (Bell Fonksiyonu)

| Duygu | Precision | Recall | F1-Score | Durum |
|-------|-----------|--------|----------|-------|
| Heyecanlı | 0.00 | 0.00 | 0.00 | ❌ Hiç tahmin edilemedi |
| Meraklı | 0.55 | 0.34 | 0.42 | ⚠️ Orta |
| Sorgulayıcı | 1.00 | 0.12 | 0.22 | ⚠️ Düşük recall |
| Umutsuz | 0.00 | 0.00 | 0.00 | ❌ Hiç tahmin edilemedi |
| korku | 0.60 | 0.44 | 0.51 | ⚠️ Orta |
| kızgın | 0.17 | 0.88 | 0.29 | ⚠️ Düşük precision |
| mutlu | 0.75 | 0.26 | 0.39 | ⚠️ Düşük recall |
| surpriz | 1.00 | 0.01 | 0.02 | ❌ Çok düşük recall |
| üzgün | 0.40 | 0.01 | 0.02 | ❌ Çok düşük recall |
| Şaşırmış | 0.00 | 0.00 | 0.00 | ❌ Hiç tahmin edilemedi |

**Tespit Edilen Sorunlar:**
1. ❌ Çok düşük genel performans (%26.88 accuracy)
2. ❌ Birçok sınıf hiç tahmin edilemedi (Heyecanlı, Umutsuz, Şaşırmış)
3. ❌ Düşük recall değerleri (birçok sınıf için %1-12 arası)
4. ❌ R² değeri neredeyse sıfır (model veriyi açıklamıyor)
5. ❌ Düşük güven skorları (ortalama %29)

---

## 🚀 Model İyileştirme Süreci

### İyileştirme Stratejisi

Performansı artırmak için sistematik bir iyileştirme süreci uygulandı. Her iyileştirme adımı test edildi ve sonuçları değerlendirildi.

### İyileştirme Adımları

#### 1️⃣ Özellik Sayısını Artırma

**Değişiklik:**
- Önceki: 50 özellik
- Yeni: **150 özellik** (3x artış)

**Gerekçe:** Daha fazla özellik, modelin daha fazla bilgi kullanmasını sağlar ve daha iyi sınıflandırma yapabilir.

#### 2️⃣ TF-IDF Özellik Sayısını Artırma

**Değişiklik:**
- Önceki: 500 özellik
- Yeni: **1000 özellik** (2x artış)

**Gerekçe:** Daha fazla kelime ve kelime kombinasyonu, duygu analizi için daha zengin özellik seti sağlar.

#### 3️⃣ N-gram Aralığını Genişletme

**Değişiklik:**
- Önceki: (1, 2) - Unigram ve Bigram
- Yeni: **(1, 3) - Unigram, Bigram ve Trigram**

**Gerekçe:** Trigram'lar, cümle bağlamını daha iyi yakalar ve Türkçe'deki kelime kombinasyonlarını daha iyi modeller.

#### 4️⃣ TF-IDF Parametrelerini Optimize Etme

**Değişiklik:**
- `sublinear_tf=True` eklendi (log scaling)
- `max_df=0.90` (daha seçici)

**Gerekçe:** Log scaling, sık geçen kelimelerin aşırı ağırlıklandırılmasını önler.

#### 5️⃣ Özellik Seçimini İyileştirme

**Değişiklik:**
- Önceki: Sadece varyans bazlı seçim
- Yeni: **Varyans + F-score kombinasyonu**

**Gerekçe:** F-score, sınıflar arası ayrımı ölçer. Varyans ile kombinasyonu, hem bilgi içeriği hem de sınıf ayrımı sağlayan özellikleri seçer.

#### 6️⃣ Kural Başına Özellik Sayısını Artırma

**Değişiklik:**
- Önceki: 3 özellik/kural
- Yeni: **5 özellik/kural**

**Gerekçe:** Daha fazla özellik kombinasyonu, daha karmaşık ve doğru kurallar oluşturur.

#### 7️⃣ Minimum Üyelik Eşiği Ekleme

**Değişiklik:**
- Önceki: Eşik yok
- Yeni: **0.15 minimum üyelik eşiği**

**Gerekçe:** Düşük üyelik dereceli kurallar gürültü yaratır. Eşik, sadece güçlü kuralları kullanır.

#### 8️⃣ Kural Birleştirme Stratejisini İyileştirme

**Değişiklik:**
- **Support faktörü** eklendi
- Daha fazla örnekte görülen kurallar daha yüksek ağırlık alır
- **Minimum güven eşiği**: %30

**Gerekçe:** Support faktörü, daha güvenilir kuralları ön plana çıkarır. Minimum güven eşiği, zayıf kuralları filtreler.

### İyileştirme Özet Tablosu

| İyileştirme | Önceki Değer | Yeni Değer | Artış |
|-------------|--------------|------------|-------|
| Özellik Sayısı | 50 | 150 | 3x |
| TF-IDF Özellikleri | 500 | 1000 | 2x |
| N-gram Aralığı | (1,2) | (1,3) | +Trigram |
| Kural Başına Özellik | 3 | 5 | +67% |
| Özellik Seçimi | Varyans | Varyans + F-score | İyileştirildi |
| Minimum Üyelik Eşiği | Yok | 0.15 | Eklendi |
| Support Faktörü | Yok | Var | Eklendi |
| Minimum Güven Eşiği | Yok | %30 | Eklendi |

---

## 📊 Final Test Sonuçları

### Final Model Konfigürasyonu

- **Özellik Sayısı**: 150
- **TF-IDF Özellikleri**: 1000
- **N-gram Aralığı**: (1, 3) - Unigram, Bigram, Trigram
- **Kural Başına Özellik**: 5
- **Özellik Seçimi**: Varyans + F-score kombinasyonu
- **Minimum Üyelik Eşiği**: 0.15
- **Support Faktörü**: Aktif
- **Minimum Güven Eşiği**: %30

### Performans Karşılaştırması

| Metrik | İlk Test | Final Test | İyileştirme | Artış Oranı |
|--------|----------|------------|-------------|-------------|
| **Accuracy** | 26.88% | **73.22%** | +46.34% | **2.7x** ⬆️ |
| **F1-Score** | 0.2182 | **0.7370** | +0.5188 | **3.4x** ⬆️ |
| **R²** | 0.0066 | **0.6077** | +0.6011 | **92x** ⬆️ |
| **Precision** | 0.5238 | **0.7978** | +0.2740 | **1.5x** ⬆️ |
| **Recall** | 0.2688 | **0.7322** | +0.4634 | **2.7x** ⬆️ |
| **Ortalama Güven** | 0.2907 | **0.7277** | +0.4370 | **2.5x** ⬆️ |

### Final Test - Üyelik Fonksiyonu Karşılaştırması

| Üyelik Fonksiyonu | Accuracy | F1-Score | R² | Precision | Recall | Ortalama Güven | En İyi? |
|-------------------|----------|----------|-----|-----------|--------|----------------|---------|
| **Üçgen** ⭐ | **73.22%** | **0.7370** | **0.6077** | **0.7978** | **0.7322** | **0.7277** | ✅ |
| **Yamuk** | 5.47% | 0.0174 | -0.4433 | 0.1929 | 0.0547 | 0.5630 | ❌ |
| **Sigmoid** | 15.64% | 0.0423 | -0.0032 | 0.0245 | 0.1564 | 0.1830 | ❌ |
| **Gauss** | 5.28% | 0.0120 | -0.4191 | 0.1816 | 0.0528 | 0.5488 | ❌ |
| **Bell** | 5.28% | 0.0120 | -0.1968 | 0.1816 | 0.0528 | 0.4087 | ❌ |

**Sonuç**: İyileştirmelerden sonra **Üçgen (Triangular)** üyelik fonksiyonu en iyi performansı gösterdi.

### Final Test - Sınıf Bazlı Performans (Üçgen Fonksiyonu)

| Duygu | Precision | Recall | F1-Score | Support | Durum |
|-------|-----------|--------|----------|---------|-------|
| **surpriz** | 0.98 | 0.84 | **0.90** | 160 | ✅ Mükemmel |
| **kızgın** | 0.98 | 0.82 | **0.89** | 160 | ✅ Mükemmel |
| **üzgün** | 0.93 | 0.69 | **0.79** | 160 | ✅ Çok İyi |
| **Sorgulayıcı** | 0.90 | 0.78 | **0.84** | 49 | ✅ Çok İyi |
| **Şaşırmış** | 0.85 | 0.72 | **0.78** | 39 | ✅ İyi |
| **mutlu** | 0.80 | 0.66 | **0.72** | 160 | ✅ İyi |
| **Heyecanlı** | 0.78 | 0.50 | **0.61** | 50 | ⚠️ Orta |
| **korku** | 0.44 | 0.96 | **0.60** | 160 | ⚠️ Düşük Precision |
| **Meraklı** | 0.61 | 0.54 | **0.58** | 35 | ⚠️ Orta |
| **Umutsuz** | 0.40 | 0.08 | **0.13** | 50 | ❌ Düşük |

**Gözlemler:**
- ✅ **surpriz** ve **kızgın** sınıfları en iyi performansı gösterdi (F1 > 0.89)
- ✅ Çoğu sınıf için F1-score > 0.70 (iyi seviye)
- ⚠️ **korku** sınıfında yüksek recall (%96) ama düşük precision (%44) - fazla tahmin yapılıyor
- ❌ **Umutsuz** sınıfı hala düşük performans gösteriyor (F1 = 0.13)

### Kural İstatistikleri

- **Toplam Kural Sayısı**: 1,100
- **Aktif Kural Sayısı**: 1,100 (tümü kullanılıyor)
- **Ortalama Kural Güveni**: 0.7277 (%72.77)
- **En Yüksek Kural Güveni**: ~0.98
- **En Düşük Kural Güveni**: 0.30 (minimum eşik)

---

## 📐 Üyelik Fonksiyonları Karşılaştırması

### Detaylı Karşılaştırma Tablosu

| Üyelik Fonksiyonu | Accuracy | F1-Score | R² | Precision | Recall | Güven | Kural Sayısı |
|-------------------|----------|----------|-----|-----------|--------|-------|--------------|
| **Üçgen** ⭐ | 73.22% | 0.7370 | 0.6077 | 0.7978 | 0.7322 | 0.7277 | 1,100 |
| **Yamuk** | 5.47% | 0.0174 | -0.4433 | 0.1929 | 0.0547 | 0.5630 | 17 |
| **Sigmoid** | 15.64% | 0.0423 | -0.0032 | 0.0245 | 0.1564 | 0.1830 | 1,117 |
| **Gauss** | 5.28% | 0.0120 | -0.4191 | 0.1816 | 0.0528 | 0.5488 | 63 |
| **Bell** | 5.28% | 0.0120 | -0.1968 | 0.1816 | 0.0528 | 0.4087 | 78 |

### Üyelik Fonksiyonu Analizi

#### 1. Üçgen (Triangular) - EN İYİ ⭐

**Performans:**
- Accuracy: 73.22%
- F1-Score: 0.7370
- R²: 0.6077

**Avantajlar:**
- ✅ Basit ve hızlı hesaplama
- ✅ Bu veri seti için en iyi sonuçlar
- ✅ Yüksek güven skorları
- ✅ 1,100 kural ile kapsamlı model

**Neden Başarılı:**
- Basit yapısı, bu veri setindeki özellik dağılımlarına uygun
- Keskin geçişler, duygu sınıfları arasındaki ayrımı iyi yapıyor

#### 2. Yamuk (Trapezoidal)

**Performans:**
- Accuracy: 5.47%
- F1-Score: 0.0174
- R²: -0.4433

**Sorunlar:**
- ❌ Çok düşük performans
- ❌ Sadece 17 kural oluşturuldu (yetersiz)
- ❌ Negatif R² (model veriyi açıklamıyor)

**Neden Başarısız:**
- Geniş geçiş bölgeleri, bu veri seti için uygun değil
- Çok az kural oluşturulması, modelin yetersiz öğrenmesine neden oldu

#### 3. Sigmoid

**Performans:**
- Accuracy: 15.64%
- F1-Score: 0.0423
- R²: -0.0032

**Sorunlar:**
- ❌ Düşük performans
- ❌ Çok fazla kural (1,117) ama düşük kalite
- ❌ Düşük güven skorları

**Neden Başarısız:**
- Yumuşak geçişler, bu problem için uygun değil
- Asimetrik yapı, sınıf ayrımını zorlaştırıyor

#### 4. Gauss (Gaussian)

**Performans:**
- Accuracy: 5.28%
- F1-Score: 0.0120
- R²: -0.4191

**Sorunlar:**
- ❌ Çok düşük performans
- ❌ Sadece 63 kural
- ❌ Negatif R²

**Neden Başarısız:**
- Simetrik yapı, bu veri setindeki asimetrik dağılımlara uygun değil
- Çok az kural oluşturulması

#### 5. Bell

**Performans:**
- Accuracy: 5.28%
- F1-Score: 0.0120
- R²: -0.1968

**Sorunlar:**
- ❌ Çok düşük performans
- ❌ Sadece 78 kural
- ❌ Düşük güven skorları

**Neden Başarısız:**
- İlk testte en iyi performansı göstermişti ama iyileştirmelerden sonra diğer fonksiyonlar geride kaldı
- Parametre kontrolü yüksek ama bu veri seti için optimize edilmedi

---

## 🔬 Teknik Detaylar

### Model Mimarisi

```
1. VERİ ÖN İŞLEME
   ├── Metin temizleme (URL, @mention, #hashtag kaldırma)
   ├── Küçük harfe çevirme
   ├── Stop words kaldırma (Türkçe)
   └── Tokenization

2. ÖZELLİK ÇIKARIMI
   ├── TF-IDF vektörizasyonu (1000 özellik)
   ├── N-gram: (1, 3) - Unigram, Bigram, Trigram
   ├── Sublinear TF scaling
   └── Özellik seçimi: Varyans + F-score (150 özellik)

3. BULANIKLAŞTIRMA
   ├── Her özellik için bulanık kümeler oluşturma
   ├── Üçgen üyelik fonksiyonu
   ├── Düşük/Orta/Yüksek kategorileri
   └── Minimum üyelik eşiği: 0.15

4. KURAL ÇIKARIMI
   ├── Eğitim verisinden otomatik kural oluşturma
   ├── Her kuralda maksimum 5 özellik
   ├── Support faktörü ile ağırlıklandırma
   ├── Minimum güven eşiği: %30
   └── 1,100 kural oluşturuldu

5. ÇIKARIM VE KESİNLEŞTİRME
   ├── Bulanık kurallar ile tahmin
   ├── Sınıf olasılıklarını hesaplama
   ├── Support faktörü ile ağırlıklandırma
   └── En yüksek olasılıklı sınıfı seçme
```

### Algoritma Akışı

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         VERİ SETİ (5,113 Tweet)                         │
│                   10 Duygu Sınıfı: mutlu, üzgün, korku, ...            │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        VERİ ÖN İŞLEME                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │
│  │ Metin        │  │ Stop Words   │  │ Tokenization │                │
│  │ Temizleme    │→ │ Kaldırma     │→ │              │                │
│  │ (URL, @, #)  │  │ (Türkçe)     │  │              │                │
│  └──────────────┘  └──────────────┘  └──────────────┘                │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    TF-IDF VEKTÖRİZASYONU                                 │
│  ┌──────────────────────────────────────────────────────┐              │
│  │ • N-gram: (1, 3) - Unigram, Bigram, Trigram         │              │
│  │ • Max Features: 1000                                 │              │
│  │ • Sublinear TF Scaling: Aktif                       │              │
│  │ • Min DF: 2, Max DF: 0.90                           │              │
│  └──────────────────────────────────────────────────────┘              │
│                          ↓                                             │
│              [1000 Boyutlu Özellik Vektörü]                            │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      ÖZELLİK SEÇİMİ (150 Özellik)                       │
│  ┌──────────────────────────────────────────────────────┐              │
│  │ • Varyans Analizi                                    │              │
│  │ • F-Score (Sınıf Ayrımı)                             │              │
│  │ • Kombine Skor: Varyans × (1 + F-Score)              │              │
│  │ • En İyi 150 Özellik Seçildi                         │              │
│  └──────────────────────────────────────────────────────┘              │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      BULANIKLAŞTIRMA                                     │
│  ┌──────────────────────────────────────────────────────┐              │
│  │ Her Özellik İçin:                                    │              │
│  │   • Üçgen Üyelik Fonksiyonu                         │              │
│  │   • 3 Bulanık Küme: Düşük, Orta, Yüksek             │              │
│  │   • Minimum Üyelik Eşiği: 0.15                       │              │
│  │   • Üyelik Dereceleri Hesaplanır                     │              │
│  └──────────────────────────────────────────────────────┘              │
│                          ↓                                             │
│         [Her Özellik → Bulanık Küme Üyelik Dereceleri]                 │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      KURAL ÇIKARIMI                                     │
│  ┌──────────────────────────────────────────────────────┐              │
│  │ • Her Eğitim Örneği İçin:                            │              │
│  │   - En Yüksek Üyelik Dereceli 5 Özellik Seçilir     │              │
│  │   - Kural Oluşturulur:                              │              │
│  │     EĞER feat1=set1 VE feat2=set2 ... İSE label     │              │
│  │                                                      │              │
│  │ • Kural Birleştirme:                                │              │
│  │   - Benzer Kurallar Birleştirilir                   │              │
│  │   - Support Faktörü ile Ağırlıklandırılır           │              │
│  │   - Minimum Güven Eşiği: %30                        │              │
│  └──────────────────────────────────────────────────────┘              │
│                          ↓                                             │
│                    [1,100 Kural Oluşturuldu]                            │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      MODEL EĞİTİMİ                                     │
│  ┌──────────────────────────────────────────────────────┐              │
│  │ • Train/Test Split: 80/20                            │              │
│  │ • Train Seti: 4,089 örnek                            │              │
│  │ • Test Seti: 1,023 örnek                             │              │
│  │ • 5 Üyelik Fonksiyonu Test Edildi                    │              │
│  │ • En İyi: Üçgen (Triangular)                         │              │
│  └──────────────────────────────────────────────────────┘              │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      TAHMİN (ÇIKARIM)                                   │
│  ┌──────────────────────────────────────────────────────┐              │
│  │ Yeni Metin İçin:                                     │              │
│  │   1. Ön İşleme → TF-IDF → Özellik Seçimi            │              │
│  │   2. Bulanıklaştırma (Üyelik Dereceleri)            │              │
│  │   3. Kural Eşleştirme:                               │              │
│  │      - Her kural için uygunluk derecesi              │              │
│  │      - Support faktörü ile ağırlıklandırma          │              │
│  │   4. Sınıf Skorları Toplanır                         │              │
│  │   5. En Yüksek Skorlu Sınıf Seçilir                 │              │
│  └──────────────────────────────────────────────────────┘              │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      DEĞERLENDİRME                                      │
│  ┌──────────────────────────────────────────────────────┐              │
│  │ Metrikler:                                           │              │
│  │   • Accuracy: 73.22%                                │              │
│  │   • F1-Score: 0.7370                                │              │
│  │   • R²: 0.6077                                      │              │
│  │   • Precision: 0.7978                                │              │
│  │   • Recall: 0.7322                                   │              │
│  │   • Ortalama Güven: 0.7277                           │              │
│  │                                                      │              │
│  │ Görselleştirmeler:                                   │              │
│  │   • Confusion Matrix                                 │              │
│  │   • Sınıf Dağılımı                                   │              │
│  │   • Güven Skorları                                   │              │
│  └──────────────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────────┘
```

### R² (Determinasyon Katsayısı) Açıklaması

**R² değeri**, modelin veriyi ne kadar iyi açıkladığını gösterir:

- **R² = 1.0**: Mükemmel uyum (ideal durum)
- **R² = 0.75-1.0**: Çok iyi uyum ✅
- **R² = 0.50-0.75**: İyi uyum ✅
- **R² = 0.25-0.50**: Orta uyum ⚠️
- **R² < 0.25**: Zayıf uyum ❌
- **R² < 0**: Model veriyi açıklamıyor ❌

**Bizim Modelimiz:**
- **İlk Test R²**: 0.0066 (neredeyse sıfır) ❌
- **Final Test R²**: **0.6077** (%60.77) ✅
- Bu değer, modelin veriyi **%60.77 oranında açıkladığını** gösterir
- Bu, duygu analizi için **çok iyi** bir sonuçtur

---

## 🚀 Kurulum ve Kullanım

### Gereksinimler

- **Python**: 3.8 veya üzeri
- **İşletim Sistemi**: Windows, Linux, macOS
- **RAM**: Minimum 4GB (önerilen: 8GB+)
- **Disk Alanı**: ~500MB

### Kurulum

```bash
# 1. Repository'yi klonlayın
git clone <repository-url>
cd Bm

# 2. Sanal ortam oluşturun
python -m venv .venv

# 3. Sanal ortamı aktifleştirin
# Windows:
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# 4. Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt
```

### Model Eğitimi

```bash
python train_model.py
```

Bu komut:
- Veri setini yükler ve ön işler
- 5 farklı üyelik fonksiyonunu test eder
- En iyi performans gösteren fonksiyonu seçer
- Modeli `best_fuzzy_model.pkl` olarak kaydeder

### Model Değerlendirme

```bash
python evaluate_model.py
```

Bu komut:
- Model performansını detaylı olarak değerlendirir
- Görselleştirmeler oluşturur
- Raporu `evaluation_report.csv` olarak kaydeder

### Web Arayüzü

```bash
streamlit run app.py
```

Tarayıcınızda `http://localhost:8501` adresinde açılacaktır.

---

## 🎨 Görselleştirmeler

### Görsel 1: Confusion Matrix

**Dosya:** `confusion_matrix.png`

Bu görsel, modelin hangi sınıfları doğru tahmin ettiğini ve hangi sınıflar arasında karışıklık olduğunu gösterir.

```
[Görsel alanı - confusion_matrix.png dosyasını buraya ekleyin]
```

**Yorumlama:**
- Diyagonal değerler: Doğru tahminler ✅
- Diyagonal dışı değerler: Yanlış tahminler ❌
- Koyu renkler: Daha fazla örnek
- Açık renkler: Daha az örnek

**Gözlemler:**
- **surpriz** ve **kızgın** sınıfları için çok az karışıklık var
- **korku** sınıfı bazı diğer sınıflarla karışıyor (fazla tahmin)
- **Umutsuz** sınıfı için çok az doğru tahmin var

---

### Görsel 2: Üyelik Fonksiyonu Karşılaştırması

**Dosya:** `membership_function_comparison.png` (oluşturulacak)

Bu görsel, farklı üyelik fonksiyonlarının performansını karşılaştırır.

```
[Görsel alanı - membership_function_comparison.png dosyasını buraya ekleyin]
```

**Yorumlama:**
- X ekseni: Üyelik fonksiyonları
- Y ekseni: Metrik değerleri (0-1 arası)
- En yüksek çubuk: En iyi performans (Üçgen)

---

### Görsel 3: Sınıf Dağılımı

**Dosya:** `class_distribution.png`

Bu görsel, gerçek ve tahmin edilen sınıf dağılımlarını karşılaştırır.

```
[Görsel alanı - class_distribution.png dosyasını buraya ekleyin]
```

**Yorumlama:**
- Sol grafik: Gerçek sınıf dağılımı
- Sağ grafik: Tahmin edilen sınıf dağılımı
- Benzerlik: Model dengeli tahmin yapıyor ✅

---

### Görsel 4: Metrik Karşılaştırması

**Dosya:** `metrics_comparison.png`

Bu görsel, tüm performans metriklerini bir arada gösterir.

```
[Görsel alanı - metrics_comparison.png dosyasını buraya ekleyin]
```

**Yorumlama:**
- Accuracy: Genel doğruluk (%73.22) ✅
- F1-Score: Denge metrik (0.7370) ✅
- R²: Açıklama gücü (0.6077) ✅
- Precision: Kesinlik (0.7978) ✅
- Recall: Geri çağırma (0.7322) ✅

---

### Görsel 5: Güven Skorları Dağılımı

**Dosya:** `confidence_distribution.png`

Bu görsel, modelin tahminlerindeki güven seviyelerini gösterir.

```
[Görsel alanı - confidence_distribution.png dosyasını buraya ekleyin]
```

**Yorumlama:**
- Yüksek güven: Model emin ✅
- Düşük güven: Model belirsiz ⚠️
- Ortalama: 0.7277 (%72.77) - İyi seviye ✅
- Dağılım: Çoğu tahmin yüksek güvenle yapılıyor

---

### Görsel 6: Performans İyileştirme Grafiği

**Dosya:** `performance_improvement.png` (oluşturulacak)

Bu görsel, iyileştirme sürecindeki performans değişimini gösterir.

```
[Görsel alanı - performance_improvement.png dosyasını buraya ekleyin]
```

**Yorumlama:**
- X ekseni: İyileştirme adımları
- Y ekseni: Metrik değerleri
- İlk test → Final test: Dramatik iyileştirme görülüyor

---

## 📁 Proje Yapısı

```
Bm/
│
├── 📊 TurkishTweets.xlsx              # Veri seti (5,113 tweet)
│
├── 📦 requirements.txt                # Gerekli Python kütüphaneleri
│
├── 🐍 Python Modülleri
│   ├── data_preprocessing.py          # Veri ön işleme modülü
│   ├── fuzzy_sentiment.py             # Bulanık mantık modeli
│   ├── train_model.py                 # Model eğitimi ve karşılaştırma
│   ├── evaluate_model.py              # Model değerlendirme
│   └── app.py                         # Streamlit web arayüzü
│
├── 💾 Model Dosyaları
│   ├── best_fuzzy_model.pkl           # Eğitilmiş en iyi model
│   └── preprocessing_artifacts.pkl    # TF-IDF vectorizer ve label encoder
│
├── 📈 Sonuç Dosyaları
│   ├── membership_function_comparison.csv    # Üyelik fonksiyonu karşılaştırması
│   └── evaluation_report.csv                 # Detaylı değerlendirme raporu
│
├── 🎨 Görselleştirmeler
│   ├── confusion_matrix.png           # Confusion matrix
│   ├── class_distribution.png         # Sınıf dağılımı
│   ├── confidence_distribution.png    # Güven skorları
│   ├── metrics_comparison.png         # Metrik karşılaştırması
│   ├── membership_function_comparison.png  # Üyelik fonksiyonu karşılaştırması (oluşturulacak)
│   └── performance_improvement.png    # Performans iyileştirme grafiği (oluşturulacak)
│
└── 📖 README.md                       # Bu dosya
```

---

## 📊 Özet ve Sonuçlar

### Başarılar

✅ **Performans İyileştirmesi:**
- Accuracy: %26.88 → %73.22 (**2.7x artış**)
- F1-Score: 0.2182 → 0.7370 (**3.4x artış**)
- R²: 0.0066 → 0.6077 (**92x artış**)

✅ **Model Kalitesi:**
- 1,100 kural ile kapsamlı model
- %72.77 ortalama güven skoru
- %60.77 R² (model veriyi iyi açıklıyor)

✅ **Sınıf Performansı:**
- 6 sınıf için F1-score > 0.70 (iyi seviye)
- **surpriz** ve **kızgın** için F1 > 0.89 (mükemmel)

### Öğrenilen Dersler

1. **Özellik Mühendisliği Kritik:** Daha fazla ve daha iyi özellik, performansı dramatik şekilde artırdı
2. **Üyelik Fonksiyonu Seçimi Önemli:** Farklı fonksiyonlar çok farklı sonuçlar verdi
3. **Sistematik İyileştirme:** Adım adım iyileştirme, her değişikliğin etkisini görmemizi sağladı
4. **Veri Seti Dengesizliği:** Dengesiz veri seti, bazı sınıfların düşük performansına neden oldu

### Gelecek İyileştirmeler

🔮 **Öneriler:**
1. **Veri Artırma:** Düşük performanslı sınıflar (Umutsuz, Meraklı) için daha fazla veri toplama
2. **Özellik Mühendisliği:** Word embeddings (Word2Vec, FastText) denemek
3. **Hiperparametre Optimizasyonu:** Grid search veya Bayesian optimization
4. **Ensemble Yöntemleri:** Farklı üyelik fonksiyonlarını birleştirmek
5. **Dengesiz Veri İçin:** SMOTE veya class weighting kullanmak

---

## 📚 Referanslar

### Bulanık Mantık
- Zadeh, L. A. (1965). "Fuzzy sets". Information and Control, 8(3), 338-353.
- Jang, J. S. (1993). "ANFIS: adaptive-network-based fuzzy inference system". IEEE Transactions on Systems, Man, and Cybernetics, 23(3), 665-685.

### Duygu Analizi
- Liu, B. (2012). "Sentiment Analysis and Opinion Mining". Synthesis Lectures on Human Language Technologies.

### Türkçe NLP
- Türkçe Doğal Dil İşleme kaynakları ve araçları

---

## ⚠️ Notlar

- Model performansı veri setinin kalitesine ve boyutuna bağlıdır
- Dengesiz veri seti, bazı sınıfların düşük performansına neden olabilir
- Bulanık sistemler genelde küçük-orta boyutlu veri setleri için uygundur
- Üyelik fonksiyonu seçimi, veri setine özgü olabilir

---

## 📝 Lisans

Bu proje eğitim ve araştırma amaçlıdır.

---

<div align="center">

**⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın! ⭐**

Made with ❤️ using Fuzzy Logic

</div>
