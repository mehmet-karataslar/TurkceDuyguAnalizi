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
    "mutluluk": "mutlu",
    "uzuntu": "üzgün",
    "korku": "korku",
    "ofke": "kızgın",
    "saskinlik": "surpriz",
    "igrenme": "kızgın",
    "minnet": "mutlu",
    "pismanlik": "Umutsuz",
}

# Egitim setine eklenecek HF ornek ust siniri (kalite icin dusuk tutuldu)
TRAIN_CAP = {
    "mutlu": 1200,
    "üzgün": 1200,
    "kızgın": 1200,
    "korku": 1200,
    "surpriz": 1000,
    "Umutsuz": 800,
}

# Kutuphane icin daha genis
LIBRARY_CAP = {
    "mutlu": 3500,
    "üzgün": 3500,
    "kızgın": 3500,
    "korku": 3500,
    "surpriz": 3000,
    "Umutsuz": 2500,
}


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
    df["Etiket"] = df[emotion_cols].idxmax(axis=1).map(EIGHT_CLASS_MAP)
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


def main() -> None:
    original = load_original()
    library = dedupe(pd.concat([original, load_eight_class(LIBRARY_CAP)], ignore_index=True))
    train_set = dedupe(pd.concat([original, load_eight_class(TRAIN_CAP, seed=7)], ignore_index=True))

    CORPUS_CSV.parent.mkdir(parents=True, exist_ok=True)
    library.to_csv(CORPUS_CSV, index=False, encoding="utf-8-sig")
    train_set[["Tweet", "Etiket"]].to_excel(OUT_XLSX, index=False)

    META.write_text(
        "\n".join(
            [
                "Veri kutuphanesi + curated egitim seti",
                f"Kutuphane: {len(library)}",
                f"Egitim seti (TurkishTweets.xlsx): {len(train_set)}",
                "",
                "Kutuphane etiketleri:",
                library["Etiket"].value_counts().to_string(),
                "",
                "Egitim etiketleri:",
                train_set["Etiket"].value_counts().to_string(),
                "",
                "Kaynaklar: original TurkishTweets + HuggingFace nihalenc/turkish-8class-emotion-dataset",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Kutuphane: {len(library)} | Egitim: {len(train_set)}")
    print(train_set["Etiket"].value_counts())


if __name__ == "__main__":
    main()
