"""
DuyguAsistan — canlı sohbet arayüzü + Türkçe açıklamalı proje paneli
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from data_preprocessing import clean_text, load_preprocessing_artifacts, remove_stopwords
from emotion_rules import apply_emotion_cues
from fuzzy_sentiment import FuzzySentimentClassifier

st.set_page_config(
    page_title="DuyguAsistan",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

EMOTION_META = {
    "mutlu": {"emoji": "😊", "color": "#38bdf8", "hint": "Pozitif ve neşeli bir ton"},
    "üzgün": {"emoji": "😢", "color": "#7dd3fc", "hint": "Üzüntü veya hayal kırıklığı"},
    "korku": {"emoji": "😨", "color": "#60a5fa", "hint": "Endişe veya korku hissi"},
    "kızgın": {"emoji": "😠", "color": "#93c5fd", "hint": "Öfke veya rahatsızlık"},
    "surpriz": {"emoji": "😲", "color": "#22d3ee", "hint": "Şaşkınlık / beklenmedik durum"},
    "Şaşırmış": {"emoji": "😯", "color": "#67e8f9", "hint": "Hayret ve şaşırma"},
    "Heyecanlı": {"emoji": "🎉", "color": "#0ea5e9", "hint": "Enerji ve heyecan"},
    "Meraklı": {"emoji": "🔍", "color": "#0284c7", "hint": "Merak ve ilgi"},
    "Sorgulayıcı": {"emoji": "🤔", "color": "#3b82f6", "hint": "Soru sorma / sorgulama"},
    "Umutsuz": {"emoji": "😔", "color": "#94a3b8", "hint": "Umutsuzluk veya yılgınlık"},
}

MEMBERSHIP_TR = {
    "triangular": "Üçgen",
    "trapezoidal": "Yamuk",
    "sigmoid": "Sigmoid",
    "gaussian": "Gauss",
    "bell": "Bell (çan)",
}

DATASET_SOURCES = [
    "Orijinal Türkçe tweet veri seti (10 duygu sınıfı)",
    "Hugging Face: 8 sınıflı Türkçe duygu veri seti",
]

STARTERS = [
    "Bugün çok mutlu bir gün geçirdim!",
    "Bu haber beni çok üzdü.",
    "Aniden karşımda belirdi, çok korktum!",
    "Vay be, hiç beklemiyordum!",
    "Harika bir film izledim.",
    "İşten çıktım, yorgunum ama mutluyum.",
]

VISUALS = [
    ("Başarı metrikleri", "Gorseller/metrics_comparison.png"),
    ("Üyelik karşılaştırması", "Gorseller/membership_function_comparison.png"),
    ("Karışıklık matrisi", "Gorseller/confusion_matrix.png"),
    ("Sınıf dağılımı", "Gorseller/class_distribution.png"),
    ("Güven dağılımı", "Gorseller/confidence_distribution.png"),
]

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@500;600;700;800&family=Outfit:wght@600;700;800&display=swap');

:root {
  --ocean-deep: #071525;
  --ocean-mid: #0c2340;
  --ocean-panel: #0a1f3d;
  --sea: #0284c7;
  --sea-bright: #38bdf8;
  --foam: #e8f4fc;
  --white: #ffffff;
  --mist: #94a3b8;
}

html, body, [class*="css"] {
  font-family: 'Nunito', 'Segoe UI', sans-serif !important;
}

.stApp {
  background:
    radial-gradient(900px 480px at 12% -8%, rgba(14, 165, 233, 0.22) 0%, transparent 55%),
    radial-gradient(720px 420px at 92% 8%, rgba(56, 189, 248, 0.14) 0%, transparent 50%),
    radial-gradient(600px 360px at 55% 100%, rgba(2, 132, 199, 0.18) 0%, transparent 48%),
    linear-gradient(165deg, #050f1c 0%, #071525 42%, #0a1c33 100%) !important;
  color: #e8f4fc !important;
}

[data-testid="stHeader"] {
  background: transparent !important;
}
[data-testid="stToolbar"] { background: transparent !important; }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stStatusWidget"] { display: none !important; }

/* ===== Sol paneli açan düğme — büyük, etiketli, animasyonlu ===== */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
  display: flex !important;
  align-items: center !important;
  gap: 0.45rem !important;
  background: linear-gradient(135deg, #0369a1, #38bdf8) !important;
  color: #fff !important;
  border: 2px solid #7dd3fc !important;
  border-radius: 16px !important;
  padding: 0.55rem 0.85rem !important;
  min-width: 3.2rem !important;
  min-height: 3.2rem !important;
  box-shadow: 0 10px 28px rgba(2, 132, 199, 0.45), 0 0 0 3px rgba(56, 189, 248, 0.2) !important;
  z-index: 999999 !important;
  animation: pulse-panel 2.2s ease-in-out infinite !important;
}
[data-testid="stSidebarCollapsedControl"]::after,
[data-testid="collapsedControl"]::after {
  content: "Proje Paneli";
  font-family: 'Nunito', sans-serif !important;
  font-weight: 800 !important;
  font-size: 0.82rem !important;
  color: #fff !important;
  white-space: nowrap !important;
  letter-spacing: 0.02em !important;
}
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="collapsedControl"] svg {
  fill: #fff !important;
  color: #fff !important;
  width: 1.5rem !important;
  height: 1.5rem !important;
}
@keyframes pulse-panel {
  0%, 100% { transform: scale(1); box-shadow: 0 10px 28px rgba(2,132,199,.45), 0 0 0 3px rgba(56,189,248,.2); }
  50% { transform: scale(1.04); box-shadow: 0 12px 32px rgba(56,189,248,.4), 0 0 0 5px rgba(125,211,252,.28); }
}

button[kind="headerNoPadding"],
[data-testid="stExpandSidebarButton"],
[data-testid="stBaseButton-headerNoPadding"] {
  background: linear-gradient(135deg, #0369a1, #0ea5e9) !important;
  color: #fff !important;
  border: 2px solid #7dd3fc !important;
  border-radius: 12px !important;
  box-shadow: 0 6px 16px rgba(2, 132, 199, 0.35) !important;
}

/* Sidebar — koyu deniz mavisi */
section[data-testid="stSidebar"] {
  background: linear-gradient(185deg, #061628 0%, #0a2344 45%, #0c2f56 100%) !important;
  border-right: 3px solid #1d4ed8 !important;
}
section[data-testid="stSidebar"] > div { background: transparent !important; }
section[data-testid="stSidebar"] * { color: #e8f4fc !important; }
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] small {
  color: #bae6fd !important;
}
section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
  color: #ffffff !important;
  font-weight: 800 !important;
  font-size: 1.35rem !important;
}
section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
  color: #7dd3fc !important;
  font-weight: 800 !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] {
  background: rgba(14, 165, 233, 0.12) !important;
  border: 1px solid rgba(125, 211, 252, 0.28) !important;
  border-radius: 14px !important;
  margin-bottom: 0.45rem !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
  background: rgba(8, 30, 55, 0.9) !important;
  border-color: rgba(125, 211, 252, 0.35) !important;
}
section[data-testid="stSidebar"] img {
  border-radius: 10px;
  border: 2px solid rgba(56, 189, 248, 0.35);
}

.block-container {
  padding-top: 1.1rem !important;
  padding-bottom: 2.5rem !important;
  max-width: 840px !important;
}

.brand {
  display: flex;
  align-items: center;
  gap: 0.95rem;
  margin-bottom: 0.65rem;
}
.brand-mark {
  width: 56px; height: 56px;
  border-radius: 18px;
  background: linear-gradient(145deg, #0369a1 0%, #38bdf8 100%);
  color: #fff;
  display: grid; place-items: center;
  font-size: 1.45rem;
  box-shadow: 0 10px 24px rgba(14, 165, 233, 0.35);
  border: 2px solid #7dd3fc;
}
.brand h1 {
  font-family: 'Outfit', sans-serif;
  font-size: 2rem;
  font-weight: 800;
  margin: 0;
  background: linear-gradient(90deg, #e8f4fc, #38bdf8);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  letter-spacing: -0.03em;
}
.brand p {
  margin: 0.2rem 0 0;
  color: #7dd3fc;
  font-size: 1rem;
  font-weight: 700;
}

.welcome {
  background: linear-gradient(135deg, #0c2340 0%, #0a2a4a 100%);
  border: 2px solid #1d4ed8;
  border-radius: 20px;
  padding: 1.25rem 1.35rem;
  margin: 0.35rem 0 1rem;
  color: #e8f4fc;
  box-shadow: 0 12px 28px rgba(2, 132, 199, 0.18);
  font-size: 1.05rem;
  line-height: 1.55;
}
.welcome strong { color: #38bdf8; }

.section-label {
  font-size: 0.8rem;
  font-weight: 800;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: #38bdf8;
  margin: 0.35rem 0 0.55rem;
}

.emotion-card {
  border-radius: 20px;
  padding: 1.15rem 1.25rem;
  margin: 0.35rem 0 0.8rem;
  background: linear-gradient(160deg, #0c2340 0%, #102a4a 100%);
  border: 1px solid rgba(125, 211, 252, 0.22);
  box-shadow: 0 10px 26px rgba(2, 20, 40, 0.45);
  border-left: 6px solid #38bdf8;
}
.emotion-head { display: flex; align-items: center; gap: 0.85rem; }
.emotion-emoji {
  width: 56px; height: 56px;
  border-radius: 16px;
  display: grid; place-items: center;
  font-size: 1.6rem;
}
.emotion-label {
  font-family: 'Outfit', sans-serif;
  font-size: 1.45rem;
  font-weight: 800;
  color: #ffffff;
  margin: 0;
}
.emotion-sub {
  margin: 0.15rem 0 0;
  color: #bae6fd;
  font-size: 0.92rem;
  font-weight: 600;
}
.confidence-row {
  margin-top: 0.9rem;
  display: flex;
  align-items: center;
  gap: 0.7rem;
}
.confidence-bar {
  flex: 1; height: 12px;
  border-radius: 999px;
  background: #1e3a5f;
  overflow: hidden;
}
.confidence-fill { height: 100%; border-radius: 999px; }
.confidence-pct {
  font-weight: 800;
  font-size: 1.05rem;
  min-width: 3.4rem;
  text-align: right;
  color: #e8f4fc;
}
.alts {
  margin-top: 0.85rem;
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
}
.alt-chip {
  font-size: 0.8rem;
  font-weight: 700;
  padding: 0.32rem 0.75rem;
  border-radius: 999px;
  background: rgba(14, 165, 233, 0.18);
  color: #e0f2fe;
  border: 1px solid rgba(125, 211, 252, 0.4);
}

.side-title {
  font-family: 'Outfit', sans-serif;
  font-size: 1.5rem;
  font-weight: 800;
  margin: 0 0 0.3rem 0;
  color: #fff !important;
}
.side-chip {
  display: inline-block;
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  padding: 0.3rem 0.7rem;
  border-radius: 999px;
  background: #38bdf8;
  color: #071525 !important;
  border: none;
  margin-bottom: 0.75rem;
}
.info-box {
  background: rgba(14, 165, 233, 0.12);
  border: 1px solid rgba(125, 211, 252, 0.28);
  border-radius: 12px;
  padding: 0.75rem 0.85rem;
  margin: 0.45rem 0 0.7rem;
  font-size: 0.88rem;
  line-height: 1.5;
  color: #e8f4fc !important;
}
.pipeline-step {
  font-size: 0.86rem;
  line-height: 1.45;
  padding: 0.55rem 0.7rem;
  margin-bottom: 0.4rem;
  border-radius: 10px;
  background: rgba(14, 165, 233, 0.12);
  border-left: 4px solid #38bdf8;
}

[data-testid="stChatMessage"] { background: transparent !important; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong {
  color: #e8f4fc !important;
  font-size: 1.02rem !important;
  line-height: 1.55 !important;
}

[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"],
div[data-testid="stBottomBlockContainer"] {
  background: transparent !important;
  background-color: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
[data-testid="stBottomBlockContainer"]::before {
  display: none !important;
}

div[data-testid="stChatInput"],
[data-testid="stChatInput"] {
  background: transparent !important;
  pointer-events: auto !important;
  z-index: 100 !important;
}
div[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div {
  background: #0c2340 !important;
  border: 2px solid #38bdf8 !important;
  border-radius: 18px !important;
  box-shadow: 0 12px 32px rgba(2, 132, 199, 0.28), 0 0 0 3px rgba(56, 189, 248, 0.12) !important;
  pointer-events: auto !important;
}
div[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea {
  font-family: 'Nunito', sans-serif !important;
  font-size: 1.08rem !important;
  color: #e8f4fc !important;
  caret-color: #38bdf8 !important;
  background: #0c2340 !important;
  pointer-events: auto !important;
  opacity: 1 !important;
}
div[data-testid="stChatInput"] textarea::placeholder {
  color: #7dd3fc !important;
  opacity: 0.75 !important;
}
div[data-testid="stChatInput"] button {
  background: linear-gradient(135deg, #0369a1, #38bdf8) !important;
  color: #fff !important;
  border-radius: 12px !important;
}

.stButton > button {
  border-radius: 14px !important;
  border: 2px solid #0284c7 !important;
  background: #0c2340 !important;
  color: #e8f4fc !important;
  font-weight: 700 !important;
  box-shadow: 0 4px 12px rgba(2, 132, 199, 0.2) !important;
  transition: all 0.15s ease !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #0369a1, #0ea5e9) !important;
  border-color: #38bdf8 !important;
  color: #fff !important;
}

.composer-wrap {
  background: #0c2340;
  border: 2px solid #38bdf8;
  border-radius: 18px;
  padding: 0.55rem 0.7rem;
  box-shadow: 0 12px 28px rgba(2,132,199,0.22);
  margin-top: 0.5rem;
}

/* Form yazma kutusu — koyu deniz teması */
div[data-testid="stForm"] {
  background: #0c2340 !important;
  border: 2px solid #38bdf8 !important;
  border-radius: 18px !important;
  padding: 0.65rem 0.85rem 0.35rem !important;
  box-shadow: 0 12px 28px rgba(2, 132, 199, 0.25), 0 0 0 3px rgba(56, 189, 248, 0.12) !important;
}
div[data-testid="stForm"] [data-testid="stTextInput"] input,
div[data-testid="stForm"] input,
div[data-testid="stForm"] input[type="text"] {
  font-size: 1.08rem !important;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  caret-color: #38bdf8 !important;
  border: 2px solid #0284c7 !important;
  background: #071525 !important;
  background-color: #071525 !important;
  border-radius: 12px !important;
  min-height: 2.75rem !important;
}
div[data-testid="stForm"] [data-testid="stTextInput"] input::placeholder,
div[data-testid="stForm"] input::placeholder {
  color: #7dd3fc !important;
  -webkit-text-fill-color: #7dd3fc !important;
  opacity: 0.8 !important;
}
div[data-testid="stForm"] [data-testid="stTextInput"] > div,
div[data-testid="stForm"] [data-baseweb="input"],
div[data-testid="stForm"] [data-baseweb="base-input"],
div[data-testid="stForm"] [data-baseweb="input"] > div {
  background: #071525 !important;
  background-color: #071525 !important;
  border-color: #0284c7 !important;
  color: #ffffff !important;
}
div[data-testid="stForm"] [data-testid="stFormSubmitButton"] button {
  background: linear-gradient(135deg, #0369a1, #38bdf8) !important;
  color: #fff !important;
  border: 2px solid #7dd3fc !important;
  border-radius: 12px !important;
  font-weight: 800 !important;
  min-height: 2.75rem !important;
}
</style>
"""


def emotion_meta(label: str) -> dict[str, str]:
    if label in EMOTION_META:
        return EMOTION_META[label]
    for key, meta in EMOTION_META.items():
        if key.casefold() == label.casefold():
            return meta
    return {"emoji": "💬", "color": "#38bdf8", "hint": "Algılanan duygu"}


@st.cache_resource
def load_runtime_models():
    import joblib

    if not os.path.exists("preprocessing_artifacts.pkl"):
        return None, None, None, None, None, "Ön işleme dosyası bulunamadı."
    try:
        vectorizer, label_encoder = load_preprocessing_artifacts("preprocessing_artifacts.pkl")
    except Exception as exc:  # noqa: BLE001
        return None, None, None, None, None, str(exc)

    sk_bundle = None
    if os.path.exists("best_sklearn_model.pkl"):
        try:
            sk_bundle = joblib.load("best_sklearn_model.pkl")
        except Exception:
            sk_bundle = None

    fuzzy = None
    if os.path.exists("best_fuzzy_model.pkl"):
        try:
            fuzzy = FuzzySentimentClassifier.load("best_fuzzy_model.pkl")
        except Exception:
            fuzzy = None

    if sk_bundle is None and fuzzy is None:
        return None, None, None, None, None, "Model dosyası bulunamadı."

    return sk_bundle, fuzzy, vectorizer, label_encoder, ("sklearn" if sk_bundle else "fuzzy"), None


@st.cache_data
def load_corpus_stats():
    path = "data/turkish_emotion_corpus.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return {
        "total": len(df),
        "counts": df["Etiket"].value_counts(),
        "sources": df["Kaynak"].value_counts() if "Kaynak" in df.columns else None,
    }


@st.cache_data
def load_eval_report(path: str = "evaluation_report.csv") -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path).iloc[0].to_dict()


@st.cache_data
def load_model_comparison(path: str = "membership_function_comparison.csv") -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path).copy()
    if "membership_type" in df.columns and "model" not in df.columns:
        df["model"] = df["membership_type"].map(lambda x: MEMBERSHIP_TR.get(x, x))
    elif "model" in df.columns:
        df["model"] = df["model"].map(lambda x: MEMBERSHIP_TR.get(x, str(x).replace("_", " ")))
    return df


def analyze_text(text: str, sk_bundle, fuzzy_model, vectorizer, label_encoder, active: str) -> dict[str, Any]:
    raw_clean = clean_text(text)
    cleaned = remove_stopwords(raw_clean)
    if not cleaned.strip():
        return {"ok": False, "error": "Metin çok kısa kaldı. Biraz daha uzun bir cümle dener misin?"}

    vector = vectorizer.transform([cleaned])
    id_to_label = {int(k): v for k, v in label_encoder["id_to_label"].items()}

    if active == "sklearn" and sk_bundle is not None:
        model = sk_bundle["model"]
        dense = vector.toarray()
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(dense)[0]
        else:
            scores = model.decision_function(dense)[0]
            exp = np.exp(scores - np.max(scores))
            probabilities = exp / exp.sum()
        # classes_ sırasına göre etiketle (güvenli eşleme)
        classes = [int(c) for c in getattr(model, "classes_", range(len(probabilities)))]
        proba_by_id = np.zeros(len(id_to_label), dtype=float)
        for i, cid in enumerate(classes):
            if cid < len(proba_by_id):
                proba_by_id[cid] = float(probabilities[i])
        probabilities = proba_by_id
    else:
        dense = vector.toarray()
        _prediction, confidence = fuzzy_model.predict(dense)
        probabilities = fuzzy_model.predict_proba(dense)[0]
        if float(confidence[0]) < 1e-6:
            probabilities = probabilities

    # Net duygu ipuçlarıyla düzelt / güçlendir (stopword'süz metin)
    probabilities, cue_label, _boost = apply_emotion_cues(raw_clean, probabilities, id_to_label)
    pred = int(np.argmax(probabilities))
    label = id_to_label[pred]
    conf = float(probabilities[pred])

    ranked = sorted(
        ((id_to_label[i], float(probabilities[i])) for i in range(len(probabilities)) if i in id_to_label),
        key=lambda item: item[1],
        reverse=True,
    )
    return {
        "ok": True,
        "label": label,
        "confidence": conf,
        "ranked": ranked,
        "cleaned": cleaned,
        "cue": cue_label,
    }


def render_assistant_reply(result: dict[str, Any]) -> None:
    if not result.get("ok"):
        st.markdown(result.get("error", "Bir şeyler ters gitti."))
        return

    meta = emotion_meta(result["label"])
    conf_pct = max(0.0, min(1.0, result["confidence"])) * 100
    alts = [item for item in result["ranked"][1:4] if item[1] >= 0.05]
    alt_html = "".join(
        f'<span class="alt-chip">{emotion_meta(n)["emoji"]} {n} · {p:.0%}</span>'
        for n, p in alts
    ) or '<span class="alt-chip">Diğer duygular zayıf</span>'

    st.markdown(
        f"""
        <div class="emotion-card" style="border-left-color:{meta['color']};">
          <div class="emotion-head">
            <div class="emotion-emoji" style="background:{meta['color']}18;">{meta['emoji']}</div>
            <div>
              <p class="emotion-label">{result['label']}</p>
              <p class="emotion-sub">{meta['hint']}</p>
            </div>
          </div>
          <div class="confidence-row">
            <div class="confidence-bar">
              <div class="confidence-fill" style="width:{conf_pct:.1f}%;background:{meta['color']};"></div>
            </div>
            <div class="confidence-pct">%{conf_pct:.0f}</div>
          </div>
          <div class="alts">{alt_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    lead = f"Bu metinde baskın duygu **{result['label']}** (güven ≈ %{conf_pct:.0f})."
    if alts:
        lead += f" Yanında hafif {', '.join(n for n, _ in alts[:2])} izleri de var."
    st.markdown(lead)


def render_project_sidebar(sk_bundle, fuzzy_model, label_encoder, active: str) -> None:
    report = load_eval_report()
    comparison_df = load_model_comparison()

    st.markdown('<p class="side-title">Proje Paneli</p>', unsafe_allow_html=True)
    st.markdown('<span class="side-chip">Teknik bilgiler</span>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Bu panelde modelin <b>ne kadar doğru çalıştığı</b>, '
        "<b>hangi veriyle eğitildiği</b> ve sistemin <b>nasıl işlediği</b> anlatılır. "
        "Paneli kapatmak/açmak için sol üstteki <b>☰ ok</b> düğmesini kullan.</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Şu an kullanılan model")
    if active == "sklearn" and sk_bundle:
        tip = str(sk_bundle.get("name", "sklearn")).replace("_", " ")
        tip_tr = {
            "linear svc calibrated": "Doğrusal Destek Vektör Makinesi (kalibre)",
            "logistic regression": "Lojistik Regresyon",
        }.get(tip.lower(), tip)
        m1, m2 = st.columns(2)
        m1.metric("Model tipi", tip_tr.split("(")[0].strip()[:18])
        m2.metric("Duygu sınıfı", len(label_encoder["id_to_label"]) if label_encoder else 10)
        st.caption(
            f"**{tip_tr}** — metni sayısal özelliklere çevirip (TF-IDF) "
            "en olası duygu sınıfını seçer. Kalibrasyon, güven skorunu daha anlamlı yapar."
        )
    elif fuzzy_model:
        m1, m2 = st.columns(2)
        m1.metric("Üyelik", MEMBERSHIP_TR.get(fuzzy_model.membership_type, fuzzy_model.membership_type))
        m2.metric("Kural sayısı", f"{len(fuzzy_model.rules_ or []):,}")
        st.caption("Bulanık mantık: kurallarla yorumlanabilir karar üretir.")

    with st.expander("Başarı oranları (ne anlama gelir?)", expanded=True):
        st.markdown(
            '<div class="info-box">'
            "<b>Doğruluk (Accuracy):</b> Test örneklerinin yüzde kaçı doğru tahmin edildi.<br>"
            "<b>F1 Skoru:</b> Kesinlik ve duyarlılığın dengeli ortalaması (0–1; yüksek daha iyi).<br>"
            "<b>Kesinlik (Precision):</b> Model “X duygusu” dediğinde ne kadar haklı.<br>"
            "<b>Duyarlılık (Recall):</b> Gerçek X duygularının ne kadarını yakaladı.<br>"
            "<b>R²:</b> Modelin olasılık dağılımını ne kadar iyi açıkladığı.<br>"
            "<b>Ortalama güven:</b> Tahminlere verdiği ortalama emin olma oranı."
            "</div>",
            unsafe_allow_html=True,
        )
        if report:
            c1, c2 = st.columns(2)
            c1.metric("Doğruluk (ne kadar doğru?)", f"{float(report['accuracy']) * 100:.1f}%")
            c2.metric("F1 skoru (denge)", f"{float(report['f1_score']):.3f}")
            c3, c4 = st.columns(2)
            c3.metric("Kesinlik (dediğimde haklı mı?)", f"{float(report['precision']):.3f}")
            c4.metric("Duyarlılık (hepsini buldu mu?)", f"{float(report['recall']):.3f}")
            conf = report.get("mean_confidence", report.get("confidence_mean", 0))
            c5, c6 = st.columns(2)
            c5.metric("R² (açıklama gücü)", f"{float(report.get('r_squared', 0)):.3f}")
            c6.metric("Ort. güven (ne kadar emin?)", f"{float(conf) * 100:.1f}%")
            if "sklearn_accuracy" in report:
                st.caption(
                    f"Karşılaştırma — Makine öğrenmesi: %{float(report['sklearn_accuracy'])*100:.1f} · "
                    f"Bulanık mantık: %{float(report.get('fuzzy_accuracy', 0))*100:.1f}"
                )
        else:
            st.caption("Metrik dosyası bulunamadı.")

    with st.expander("Veri seti (eğitim kaynağı)", expanded=True):
        st.markdown(
            '<div class="info-box">'
            "Model, etiketli Türkçe metinlerle eğitilir. "
            "<b>Kütüphane</b> tüm toplanan örnekler; "
            "<b>eğitim seti</b> kalite filtresinden geçmiş örneklerdir."
            "</div>",
            unsafe_allow_html=True,
        )
        corpus = load_corpus_stats()
        if corpus:
            st.metric("Kütüphanedeki örnek", f"{corpus['total']:,}")
            for src in DATASET_SOURCES:
                st.markdown(f"- {src}")
            st.bar_chart(
                corpus["counts"].rename_axis("Duygu").reset_index(name="Adet").set_index("Duygu")
            )
        if report and "train_size" in report:
            st.caption(
                f"Bu koşuda eğitim: {int(report['train_size']):,} örnek · "
                f"test: {int(report['test_size']):,} örnek (%80/%20 ayrım)"
            )

    with st.expander("Sistem nasıl çalışır?"):
        for step in [
            "1. Metin temizleme — link, @kullanıcı, noktalama; Türkçe dolgu kelimeleri çıkarılır",
            "2. TF-IDF — kelime/kelime gruplarının önem ağırlıkları çıkarılır",
            "3a. Bulanık yol — üyelik kümeleri + kurallar (açıklanabilir)",
            "3b. Doğrusal SVM yolu — yüksek doğruluklu sınıflandırma (aktif)",
            "4. Sonuç — duygu etiketi + güven skoru sohbette gösterilir",
        ]:
            st.markdown(f'<div class="pipeline-step">{step}</div>', unsafe_allow_html=True)

    with st.expander("Model karşılaştırması"):
        st.caption("Aynı test verisinde modellerin başarı oranları.")
        if comparison_df is not None and not comparison_df.empty:
            show = comparison_df.copy()
            label_col = "model" if "model" in show.columns else "Fonksiyon"
            show = show.rename(columns={label_col: "Model"})
            show["Doğruluk"] = (show["accuracy"] * 100).map(lambda x: f"{x:.1f}%")
            show["F1 skoru"] = show["f1_score"].map(lambda x: f"{x:.3f}")
            st.dataframe(show[["Model", "Doğruluk", "F1 skoru"]], hide_index=True, width="stretch")
            chart = show.set_index("Model")[["accuracy", "f1_score"]]
            chart.columns = ["Doğruluk", "F1"]
            st.bar_chart(chart)
        else:
            st.caption("Karşılaştırma dosyası yok.")

    with st.expander("10 duygu sınıfı ne demek?"):
        st.caption("Her sınıf, metindeki baskın duyguyu temsil eder.")
        if label_encoder:
            for _, label in sorted(label_encoder["id_to_label"].items()):
                meta = emotion_meta(label)
                st.markdown(f"{meta['emoji']} **{label}** — {meta['hint']}")

    with st.expander("Görseller"):
        available = [(n, p) for n, p in VISUALS if os.path.exists(p)]
        if available:
            choice = st.selectbox("Grafik seçin", [n for n, _ in available])
            st.image(dict(available)[choice], width="stretch")
            st.caption("Grafikler modelin test sonuçlarını özetler.")
        else:
            st.caption("Görsel klasörü boş.")

    st.caption("Eğitim projesi · Bulanık mantık + TF-IDF + makine öğrenmesi")


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    sk_bundle, fuzzy_model, vectorizer, label_encoder, active, error = load_runtime_models()

    with st.sidebar:
        render_project_sidebar(sk_bundle, fuzzy_model, label_encoder, active or "fuzzy")

    top_l, top_r = st.columns([5, 1.15])
    with top_l:
        st.markdown(
            """
            <div class="brand">
              <div class="brand-mark">💬</div>
              <div>
                <h1>DuyguAsistan</h1>
                <p>Metnini yaz, duygusunu birlikte bulalım</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_r:
        if st.button("Yeni sohbet", width="stretch"):
            st.session_state.messages = []
            st.rerun()

    if error or not vectorizer:
        st.error("Model yüklenemedi. Önce `python train_model.py` çalıştır.")
        st.caption(error or "")
        return

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="welcome">
              Merhaba! Ben <strong>DuyguAsistan</strong>.
              Aşağıya Türkçe bir cümle yaz veya örneklerden birine tıkla;
              duygusunu hemen söyleyeyim.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<p class="section-label">Hızlı dene</p>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, starter in enumerate(STARTERS):
            if cols[i % 2].button(starter, key=f"starter_{i}", width="stretch"):
                st.session_state.pending_prompt = starter
                st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "💬"):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_assistant_reply(message["result"])

    # Güvenilir yazma alanı (chat_input bazen CSS ile kilitlenebiliyordu)
    prompt = None
    if "pending_prompt" in st.session_state:
        prompt = st.session_state.pop("pending_prompt")

    with st.form("composer", clear_on_submit=True):
        c_in, c_btn = st.columns([6.2, 1.3], vertical_alignment="bottom")
        with c_in:
            typed = st.text_input(
                "Metin",
                placeholder="Buraya cümleni yaz… örn. Bugün harika geçti",
                label_visibility="collapsed",
                key="composer_text",
            )
        with c_btn:
            sent = st.form_submit_button("Gönder", width="stretch")
        if sent and typed and typed.strip():
            prompt = typed.strip()

    if prompt and prompt.strip():
        user_text = prompt.strip()
        st.session_state.messages.append({"role": "user", "content": user_text})

        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_text)

        with st.chat_message("assistant", avatar="💬"):
            with st.spinner("Duyguyu okuyorum…"):
                result = analyze_text(
                    user_text, sk_bundle, fuzzy_model, vectorizer, label_encoder, active
                )
            render_assistant_reply(result)

        st.session_state.messages.append({"role": "assistant", "result": result})
        st.rerun()


if __name__ == "__main__":
    main()
