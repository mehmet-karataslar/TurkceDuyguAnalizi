"""
Duygu ipucu (lexicon) kuralları — net anahtar kelimelerde modeli düzeltir / güçlendirir.
Eğitim verisindeki gürültü ve sınıf karışıklığına karşı çıkarım aşaması desteği.
"""

from __future__ import annotations

import re
from typing import Any

# (etiket, ipuçları, güç) — daha spesifik kalıplar önce eşleşmeli
EMOTION_CUES: list[tuple[str, list[str], float]] = [
    ("Umutsuz", ["umudumu kaybett", "umutsuzum", "umutsuz", "çaresizim", "çaresiz", "bitirdim her şeyi", "anlamsız geliyor"], 3.5),
    ("üzgün", ["çok üzdü", "beni üzdü", "üzgünüm", "üzgün", "üzdü", "hüzün", "ağlıyorum", "ağladım", "mutsuzum", "kederli", "kahroldum"], 3.2),
    ("korku", ["korktum", "korkuyorum", "çok kork", "dehşete", "ürperd", "panikledim", "korkunç"], 3.5),
    ("kızgın", ["kızgınım", "öfkeliyim", "öfkeli", "sinirlerim", "sinirlendim", "sinir oldum", "nefret ediyorum", "çok kızgın"], 3.5),
    ("Heyecanlı", ["heyecanlıyım", "heyecanlandım", "heyecandan", "yerimde duramıyorum", "heyecanla bekliyorum"], 4.0),
    ("Meraklı", ["merak ediyorum", "merak ettim", "meraklıyım", "merak ettim doğrusu", "merak içindeyim"], 4.0),
    ("Sorgulayıcı", ["sorguluyorum", "sorgulamak", "diye sorgula", "acaba gerçekten", "neden herkes", "neden böyle", "kim yaptı bu"], 4.0),
    ("Şaşırmış", ["şaşırmış durumdayım", "hayretler içinde", "afalladım", "donakaldım"], 3.5),
    ("surpriz", ["beklemiyordum", "hiç beklemiyordum", "inanamıyorum", "vay be", "şok oldum", "şaşırdım"], 3.0),
    ("mutlu", ["çok mutluyum", "mutluyum", "mutlu bir gün", "çok mutlu", "keyifliyim", "sevinçliyim", "harika geçti", "harika bir"], 2.8),
]


def tr_fold(text: str) -> str:
    """Türkçe için güvenli küçük harf."""
    if not text:
        return ""
    text = text.replace("I", "ı").replace("İ", "i")
    return text.lower()


def apply_emotion_cues(
    cleaned_text: str,
    probabilities: Any,
    id_to_label: dict[int, str],
    min_boost_conf: float = 0.12,
) -> tuple[Any, str | None, float]:
    """
    İpucu eşleşirse ilgili sınıf olasılığını yükseltir.
    Güçlü ipuçlarında sınıfı açıkça öne çıkarır.
    Returns: (yeni_proba, ipucu_etiketi_veya_None, boost_çarpanı)
    """
    import numpy as np

    proba = np.array(probabilities, dtype=float).copy()
    text = tr_fold(cleaned_text)
    label_to_id = {v: int(k) for k, v in id_to_label.items()}

    best_label = None
    best_strength = 0.0
    for label, cues, strength in EMOTION_CUES:
        for cue in cues:
            if tr_fold(cue) in text:
                if strength > best_strength:
                    best_label = label
                    best_strength = strength
                break

    if best_label is None or best_label not in label_to_id:
        return proba, None, 1.0

    idx = label_to_id[best_label]
    if idx < 0 or idx >= len(proba):
        return proba, None, 1.0

    # Zaten yeterince baskınsa dokunma
    if proba[idx] >= 0.55 and int(np.argmax(proba)) == idx:
        return proba, best_label, 1.0

    # Güçlü ipucu: taban düşük olsa bile sınıfı öne çıkar
    if best_strength >= 3.5:
        floor = max(0.42, float(np.max(proba)) + 0.08)
        proba[idx] = max(float(proba[idx]) * best_strength, floor)
    else:
        proba[idx] *= best_strength

    total = float(proba.sum())
    if total > 0:
        proba /= total

    if float(proba[idx]) < min_boost_conf:
        return np.array(probabilities, dtype=float), None, 1.0

    return proba, best_label, best_strength
