# Teknik Notlar — DuyguAsistan

Genel kurulum ve kullanım: [`README.md`](README.md)

---

## Mimari (2026)

### 1. Ön işleme (`data_preprocessing.py`)

- Küçük harf, URL / @mention temizliği, noktalama sadeleştirme
- Türkçe stop-words
- TF-IDF: `max_features=2500`, n-gram `(1,3)`, `sublinear_tf=True`

### 2. Dual model eğitimi (`train_model.py`)

| Yol | Açıklama | Çıktı |
|-----|----------|-------|
| **LinearSVC (calibrated)** | `class_weight=balanced` + `CalibratedClassifierCV` | `best_sklearn_model.pkl` (**aktif**) |
| Logistic Regression | Karşılaştırma adayı | — |
| **Fuzzy triangular** | Yorumlanabilir kural tabanı | `best_fuzzy_model.pkl` |

Sohbet arayüzü (`app.py`) önce sklearn modelini yükler; yoksa fuzzy’ye düşer.

### 3. Bulanık yol (`fuzzy_sentiment.py`)

1. Varyans + F-score → özellik seçimi  
2. Düşük / orta / yüksek üyelik kümeleri  
3. Kural çıkarımı (`min_membership_threshold`, support)  
4. Ateşleme skoru → sınıf olasılığı  

Desteklenen üyelikler: `triangular`, `trapezoidal`, `sigmoid`, `gaussian`, `bell`

---

## Korpus (`build_expanded_dataset.py`)

| Çıktı | İçerik |
|-------|--------|
| `data/turkish_emotion_corpus.csv` | Geniş kütüphane (~22.5k) |
| `TurkishTweets.xlsx` | Curated eğitim (~11k) |
| `data/TurkishTweets_original_backup.xlsx` | Orijinal yedek (~5.1k) |

**Etiket map (HF 8-class → proje):**

| HF | Proje |
|----|-------|
| mutluluk, minnet | mutlu |
| uzuntu | üzgün |
| korku | korku |
| ofke, igrenme | kızgın |
| saskinlik | surpriz |
| pismanlik | Umutsuz |

TRSAv1 ürün yorumları domain kayması nedeniyle **hariç**.

---

## Komutlar

```bash
python build_expanded_dataset.py
python train_model.py
streamlit run app.py
```

### Çıktı dosyaları

| Dosya | İçerik |
|-------|--------|
| `best_sklearn_model.pkl` | Aktif LinearSVC + label encoder |
| `best_fuzzy_model.pkl` | Üçgen üyelik + kurallar |
| `preprocessing_artifacts.pkl` | Vectorizer + label encoder |
| `membership_function_comparison.csv` | Fuzzy vs Sklearn metrikleri |
| `evaluation_report.csv` | Final / aktif model özeti |

---

## Performans (kayıtlı)

| Model | Accuracy | F1 |
|-------|---------:|---:|
| LinearSVC calibrated | **78.2%** | **0.781** |
| Logistic Regression | 77.8% | 0.779 |
| Fuzzy triangular | 53.9% | 0.557 |

---

## Hiperparametreler

| Parametre | Değer |
|-----------|------:|
| TF-IDF `max_features` | 2500 |
| Fuzzy `n_features` | 150 |
| Fuzzy `min_membership_threshold` | 0.14 |
| Fuzzy `max_features_per_rule` | 5 |
| Fuzzy dengeli örnek / sınıf | ~850–1000 |
| LinearSVC `C` | 1.0 |
| Calibration CV | 3 |

---

## Bağımlılıklar

`requirements.txt`: pandas, numpy, scikit-learn, scikit-fuzzy, nltk, streamlit, openpyxl, matplotlib, seaborn, joblib, networkx
