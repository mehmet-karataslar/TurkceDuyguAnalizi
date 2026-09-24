"""
Yuksek kaliteli egitim seti + genis veri kutuphanesi.

- data/turkish_emotion_corpus.csv : genis kutuphane (~22k+, UI icin)
- TurkishTweets.xlsx : fuzzy model egitimi icin curated set
  (orijinal + sinif basina sinirli, benzer uzunlukta HF ornekleri)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
BACKUP = ROOT / "data" / "TurkishTweets_original_backup.xlsx"
CORPUS_CSV = ROOT / "data" / "turkish_emotion_corpus.csv"
OUT_XLSX = ROOT / "TurkishTweets.xlsx"
META = ROOT / "data" / "dataset_meta.txt"

EIGHT_CLASS_MAP = {
    # Sadece net 1-1 eşleşmeler (gürültülü: igrenme/minnet/pismanlik eğitimde YOK)
    "mutluluk": "mutlu",
    "uzuntu": "üzgün",
    "korku": "korku",
    "ofke": "kızgın",
    "saskinlik": "surpriz",
}

ALLOWED_HF_SOURCE = set(EIGHT_CLASS_MAP.keys())

# Egitim setine eklenecek HF ornek ust siniri (kalite icin dusuk tutuldu)
TRAIN_CAP = {
    "mutlu": 700,
    "üzgün": 700,
    "kızgın": 700,
    "korku": 700,
    "surpriz": 500,
}

# Kutuphane icin daha genis
LIBRARY_CAP = {
    "mutlu": 3000,
    "üzgün": 3000,
    "kızgın": 3000,
    "korku": 3000,
    "surpriz": 2500,
}

# Nadir sınıflar için yüksek kaliteli tohum cümleler
RARE_SEEDS = [
    ("Yarın sabah erkenden yola çıkıyoruz, heyecandan uyku tutmuyor!", "Heyecanlı"),
    ("Uzun zamandır beklediğim konser günü geldi çattı, yerimde duramıyorum.", "Heyecanlı"),
    ("Sınav sonucu açıklandı, heyecanlıyım ve sabırsızlanıyorum!", "Heyecanlı"),
    ("İlk kez uçağa bineceğim, heyecanlanıyorum!", "Heyecanlı"),
    ("Proje finali yarın, heyecandan midem bulanıyor.", "Heyecanlı"),
    ("Bu yeni diziyi merak ediyorum, acaba nasıl devam edecek?", "Meraklı"),
    ("İnsanlar neden sabahları daha üretken oluyor, bilimsel bir açıklaması var mı?", "Meraklı"),
    ("Rüyalarımızı gerçekten biz mi kontrol edebiliyoruz yoksa tamamen rastgele mi?", "Meraklı"),
    ("Bu kahvenin tadını merak ediyorum, denemek istiyorum.", "Meraklı"),
    ("Uzayda yaşam olup olmadığını merak ediyorum.", "Meraklı"),
    ("Herkes erken kalkan yol alır diyor ama gece çalışanlar daha başarılı değil mi aslında?", "Sorgulayıcı"),
    ("Neden herkes 8 bardak su içmemiz gerektiğini söylüyor, kim yaptı bu araştırmayı?", "Sorgulayıcı"),
    ("Gerçekten kahvaltı en önemli öğün mü yoksa bunu firmalar mı uydurdu?", "Sorgulayıcı"),
    ("Başarıyı sadece diplomayla mı ölçmeliyiz, bunu sorguluyorum.", "Sorgulayıcı"),
    ("Sosyal medya bizi mutlu mu ediyor yoksa yalnızlaştırıyor mu, tartışmaya açık.", "Sorgulayıcı"),
    ("Ne kadar çabalarsam çabalayayım sonucun değişmeyeceğini bilmek beni bitiriyor.", "Umutsuz"),
    ("Artık hayal kurmaya bile gücüm kalmadı, her şey anlamsız geliyor.", "Umutsuz"),
    ("Bu tünelin ucu bir yere çıkmıyor, karanlıkta kaybolduk gittik.", "Umutsuz"),
    ("Umudumu tamamen kaybettim, hiçbir şey düzelmeyecek gibi.", "Umutsuz"),
    ("Çaresizim, çıkış yolu göremiyorum.", "Umutsuz"),
    ("Hayretler içinde kaldım, ne diyeceğimi bilemedim.", "Şaşırmış"),
    ("Afalladım, bu kadarını beklemiyordum doğrusu.", "Şaşırmış"),
    ("Donakaldım, olanları bir türlü idrak edemedim.", "Şaşırmış"),
    ("Şaşırmış durumdayım, açıklama bekliyorum.", "Şaşırmış"),
]


def load_original() -> pd.DataFrame:
    df0 = pd.read_excel(BACKUP if BACKUP.exists() else OUT_XLSX)
    df = df0.rename(columns={df0.columns[0]: "Tweet", df0.columns[-1]: "Etiket"})
    return pd.DataFrame(
        {
            "Tweet": df["Tweet"].astype(str),
            "Etiket": df["Etiket"].astype(str),
            "Kaynak": "original_turkish_tweets",
        }
    )


def load_eight_class(caps: dict[str, int], seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(RAW / "turkish-8class-emotion-dataset.csv", sep=";", engine="python", on_bad_lines="skip")
    emotion_cols = [c for c in df.columns if c != "text"]
    for col in emotion_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df = df[df[emotion_cols].sum(axis=1) == 1].copy()
    # Gürültülü sınıfları (igrenme/minnet/pismanlik) ele
    source_emotion = df[emotion_cols].idxmax(axis=1)
    df = df[source_emotion.isin(ALLOWED_HF_SOURCE)].copy()
    df["Etiket"] = source_emotion[df.index].map(EIGHT_CLASS_MAP)
    df = df.dropna(subset=["Etiket"])
    # Tweet benzeri uzunluk filtresi (cok kisa/uzun sentetikleri ele)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len().between(20, 180)]

    parts = []
    for label, cap in caps.items():
        subset = df[df["Etiket"] == label]
        if subset.empty:
            continue
        parts.append(subset.sample(n=min(cap, len(subset)), random_state=seed))
    picked = pd.concat(parts, ignore_index=True)
    return pd.DataFrame(
        {
            "Tweet": picked["text"],
            "Etiket": picked["Etiket"].astype(str),
            "Kaynak": "hf_nihalenc_8class",
        }
    )


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Tweet"] = df["Tweet"].astype(str).str.strip()
    df = df[df["Tweet"].str.len() >= 5]
    df["_k"] = df["Tweet"].str.lower().str.split().str.join(" ")
    return df.drop_duplicates(subset=["_k", "Etiket"]).drop(columns=["_k"]).reset_index(drop=True)


def rare_seed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Tweet": [t for t, _ in RARE_SEEDS],
            "Etiket": [e for _, e in RARE_SEEDS],
            "Kaynak": "curated_rare_seeds",
        }
    )


def main() -> None:
    original = load_original()
    seeds = rare_seed_frame()
    # Tohumları çoğalt (nadir sınıfları güçlendir)
    seeds_boosted = pd.concat([seeds] * 8, ignore_index=True)

    library = dedupe(pd.concat([original, load_eight_class(LIBRARY_CAP), seeds], ignore_index=True))
    train_set = dedupe(
        pd.concat([original, load_eight_class(TRAIN_CAP, seed=7), seeds_boosted], ignore_index=True)
    )

    CORPUS_CSV.parent.mkdir(parents=True, exist_ok=True)
    library.to_csv(CORPUS_CSV, index=False, encoding="utf-8-sig")
    train_set[["Tweet", "Etiket"]].to_excel(OUT_XLSX, index=False)

    META.write_text(
        "\n".join(
            [
                "Veri kutuphanesi + curated egitim seti (kalite odaklı 2026)",
                f"Kutuphane: {len(library)}",
                f"Egitim seti (TurkishTweets.xlsx): {len(train_set)}",
                "",
                "Not: HF igrenme/minnet/pismanlik egitime alinmadi (gurultu).",
                "Nadir siniflar icin curated tohum cumleler eklendi.",
                "",
                "Kutuphane etiketleri:",
                library["Etiket"].value_counts().to_string(),
                "",
                "Egitim etiketleri:",
                train_set["Etiket"].value_counts().to_string(),
                "",
                "Kaynaklar: original TurkishTweets + HF 8class (temiz esleme) + rare seeds",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Kutuphane: {len(library)} | Egitim: {len(train_set)}")
    print(train_set["Etiket"].value_counts())


if __name__ == "__main__":
    main()
