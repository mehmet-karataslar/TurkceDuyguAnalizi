"""
Genişletilmiş Türkçe duygu korpusu oluşturucu.

Kaynaklar:
1) TurkishTweets.xlsx — mevcut 10 sınıflı tweet seti
2) HuggingFace nihalenc/turkish-8class-emotion-dataset (~42k)
3) HuggingFace maydogan/Turkish_SentimentAnalysis_TRSAv1
   (örneklenmiş Positive/Negative → mutlu/üzgün güçlendirmesi)
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT_XLSX = ROOT / "TurkishTweets.xlsx"
OUT_CSV = ROOT / "data" / "turkish_emotion_corpus.csv"
BACKUP = ROOT / "data" / "TurkishTweets_original_backup.xlsx"
META = ROOT / "data" / "dataset_meta.txt"

# 8-sınıf HF etiketleri → proje etiketleri
EIGHT_CLASS_MAP = {
    "mutluluk": "mutlu",
    "uzuntu": "üzgün",
    "korku": "korku",
    "ofke": "kızgın",
    "saskinlik": "surpriz",
    "igrenme": "kızgın",  # iğrenme → en yakın: kızgın
    "minnet": "mutlu",
    "pismanlik": "Umutsuz",
}

# TRSA polarity → duygu (kontrollü örnekleme)
TRSA_MAP = {
    "Positive": "mutlu",
    "Negative": "üzgün",
}


def load_original() -> pd.DataFrame:
    src = ROOT / "TurkishTweets.xlsx"
    if not (ROOT / "data" / "TurkishTweets_original_backup.xlsx").exists() and src.exists():
        # ilk çalıştırmada yedekle
        df0 = pd.read_excel(src)
        BACKUP.parent.mkdir(parents=True, exist_ok=True)
        df0.to_excel(BACKUP, index=False)
    elif BACKUP.exists():
        df0 = pd.read_excel(BACKUP)
    else:
        df0 = pd.read_excel(src)

    df = df0.rename(columns={df0.columns[0]: "Tweet", df0.columns[-1]: "Etiket"})
    df = df[["Tweet", "Etiket"]].copy()
    df["Kaynak"] = "original_turkish_tweets"
    return df


def load_eight_class() -> pd.DataFrame:
    path = RAW / "turkish-8class-emotion-dataset.csv"
    df = pd.read_csv(path, sep=";", engine="python", on_bad_lines="skip")
    emotion_cols = [c for c in df.columns if c != "text"]
    for col in emotion_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df = df[df[emotion_cols].sum(axis=1) == 1].copy()
    df["raw_label"] = df[emotion_cols].idxmax(axis=1)
    df["Etiket"] = df["raw_label"].map(EIGHT_CLASS_MAP)
    df = df.dropna(subset=["Etiket"])
    out = pd.DataFrame(
        {
            "Tweet": df["text"].astype(str),
            "Etiket": df["Etiket"].astype(str),
            "Kaynak": "hf_nihalenc_8class",
        }
    )
    return out


def load_trsa_sample(n_pos: int = 12000, n_neg: int = 12000, seed: int = 42) -> pd.DataFrame:
    path = RAW / "TRSAv1.csv"
    df = pd.read_csv(path)
    parts = []
    for score, label, n in [
        ("Positive", "mutlu", n_pos),
        ("Negative", "üzgün", n_neg),
    ]:
        subset = df[df["score"] == score].sample(n=min(n, (df["score"] == score).sum()), random_state=seed)
        parts.append(
            pd.DataFrame(
                {
                    "Tweet": subset["review"].astype(str),
                    "Etiket": label,
                    "Kaynak": "hf_TRSAv1_sample",
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def normalize_text_key(text: str) -> str:
    return " ".join(str(text).lower().split())


def build_corpus() -> pd.DataFrame:
    RAW.mkdir(parents=True, exist_ok=True)
    frames = [load_original(), load_eight_class(), load_trsa_sample()]
    combined = pd.concat(frames, ignore_index=True)
    combined["Tweet"] = combined["Tweet"].astype(str).str.strip()
    combined = combined[combined["Tweet"].str.len() >= 5]
    combined["_key"] = combined["Tweet"].map(normalize_text_key)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["_key", "Etiket"]).drop(columns=["_key"])
    after = len(combined)
    print(f"Birlesik satir (tekillestirme oncesi/sonrasi): {before} -> {after}")
    print("\nEtiket dağılımı:")
    print(combined["Etiket"].value_counts())
    print("\nKaynak dağılımı:")
    print(combined["Kaynak"].value_counts())
    return combined.reset_index(drop=True)


def main() -> None:
    corpus = build_corpus()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    # Eğitim pipeline Excel bekliyor
    corpus[["Tweet", "Etiket"]].to_excel(OUT_XLSX, index=False)

    meta = [
        "Türkçe Duygu Korpusu — otomatik birleştirme",
        f"Toplam örnek: {len(corpus)}",
        "Kaynaklar:",
        "  1) TurkishTweets.xlsx (orijinal 10 sınıf)",
        "  2) HuggingFace nihalenc/turkish-8class-emotion-dataset (~42k)",
        "  3) HuggingFace maydogan/TRSAv1 (12k Positive→mutlu, 12k Negative→üzgün)",
        "",
        "Etiket dağılımı:",
        corpus["Etiket"].value_counts().to_string(),
        "",
        "Kaynak dağılımı:",
        corpus["Kaynak"].value_counts().to_string(),
    ]
    META.write_text("\n".join(meta), encoding="utf-8")
    print(f"\nKaydedildi: {OUT_XLSX}")
    print(f"Kaydedildi: {OUT_CSV}")
    print(f"Meta: {META}")


if __name__ == "__main__":
    main()
