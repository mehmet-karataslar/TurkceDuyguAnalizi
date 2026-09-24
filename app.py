"""
DuyguAsistan — profesyonel sohbet arayüzü + proje paneli
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from data_preprocessing import clean_text, load_preprocessing_artifacts, remove_stopwords
from fuzzy_sentiment import FuzzySentimentClassifier

st.set_page_config(
    page_title="DuyguAsistan",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Duygu renkleri: mat, okunaklı (neon yok)
EMOTION_META = {
    "mutlu": {"emoji": "😊", "color": "#1f6b5a", "hint": "Pozitif ve neşeli bir ton"},
    "üzgün": {"emoji": "😢", "color": "#4a5568", "hint": "Üzüntü veya hayal kırıklığı"},
    "korku": {"emoji": "😨", "color": "#3d4f7c", "hint": "Endişe veya korku hissi"},
    "kızgın": {"emoji": "😠", "color": "#9b2c2c", "hint": "Öfke veya rahatsızlık"},
    "surpriz": {"emoji": "😲", "color": "#9a5b1f", "hint": "Şaşkınlık / beklenmedik durum"},
    "Şaşırmış": {"emoji": "😯", "color": "#8a5a1e", "hint": "Hayret ve şaşırma"},
    "Heyecanlı": {"emoji": "🎉", "color": "#8b3a5c", "hint": "Enerji ve heyecan"},
    "Meraklı": {"emoji": "🔍", "color": "#2b5f8a", "hint": "Merak ve ilgi"},
    "Sorgulayıcı": {"emoji": "🤔", "color": "#3f4e8c", "hint": "Soru sorma / sorgulama"},
    "Umutsuz": {"emoji": "😔", "color": "#5c534a", "hint": "Umutsuzluk veya yılgınlık"},
}

MEMBERSHIP_TR = {
    "triangular": "Üçgen",
    "trapezoidal": "Yamuk",
    "sigmoid": "Sigmoid",
    "gaussian": "Gauss",
    "bell": "Bell",
}

DATASET_SOURCES = [
    "Orijinal TurkishTweets (10 sınıf)",
    "HuggingFace: nihalenc/turkish-8class-emotion-dataset",
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
    ("Metrikler", "Gorseller/metrics_comparison.png"),
    ("Üyelik karşılaştırma", "Gorseller/membership_function_comparison.png"),
    ("Confusion matrix", "Gorseller/confusion_matrix.png"),
    ("Sınıf dağılımı", "Gorseller/class_distribution.png"),
    ("Güven dağılımı", "Gorseller/confidence_distribution.png"),
]

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Libre+Baskerville:wght@400;700&display=swap');

:root {
  --ink: #12151c;
  --ink-soft: #3a4150;
  --muted: #5c6575;
  --line: #d8dce3;
  --paper: #f4f2ee;
  --card: #ffffff;
  --accent: #1c4b5a;
  --accent-soft: #e6eef1;
  --sidebar: #161b22;
  --sidebar-text: #e8eaed;
  --sidebar-muted: #9aa3b2;
}

html, body, [class*="css"] {
  font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif !important;
  color: var(--ink);
}

.stApp {
  background:
    linear-gradient(180deg, #efece6 0%, var(--paper) 40%, #ebe8e2 100%);
}

[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* —— Sidebar: koyu, sakin, profesyonel —— */
section[data-testid="stSidebar"] {
  background: var(--sidebar) !important;
  border-right: 1px solid #2a313c;
}
section[data-testid="stSidebar"] > div {
  background: var(--sidebar) !important;
}
section[data-testid="stSidebar"] * {
  color: var(--sidebar-text) !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] small {
  color: var(--sidebar-muted) !important;
}
section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
  color: #ffffff !important;
  font-weight: 700 !important;
  font-size: 1.35rem !important;
}
section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
  color: #b8c0cc !important;
  font-weight: 600 !important;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  font-size: 0.7rem !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] {
  background: #1e2530 !important;
  border: 1px solid #2f3846 !important;
  border-radius: 8px !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary {
  font-weight: 600 !important;
  color: #f0f2f5 !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
  background: #1e2530 !important;
  border: 1px solid #3a4454 !important;
  color: #f0f2f5 !important;
}
section[data-testid="stSidebar"] img {
  border-radius: 6px;
  border: 1px solid #3a4454;
}
section[data-testid="stSidebar"] hr {
  border-color: #2f3846 !important;
}

.block-container {
  padding-top: 1.5rem !important;
  padding-bottom: 7rem !important;
  max-width: 820px !important;
}

/* Brand */
.brand {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding-bottom: 0.75rem;
  border-bottom: 2px solid var(--ink);
  margin-bottom: 1.25rem;
}
.brand-mark {
  width: 44px;
  height: 44px;
  border-radius: 8px;
  background: var(--accent);
  color: #fff;
  display: grid;
  place-items: center;
  font-size: 1.1rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  font-family: 'Libre Baskerville', Georgia, serif;
}
.brand h1 {
  font-family: 'Libre Baskerville', Georgia, serif;
  font-size: 1.7rem;
  font-weight: 700;
  margin: 0;
  color: var(--ink);
  letter-spacing: -0.02em;
  line-height: 1.15;
}
.brand p {
  margin: 0.2rem 0 0;
  color: var(--muted);
  font-size: 0.95rem;
  font-weight: 500;
}

.welcome {
  background: var(--card);
  border: 1px solid var(--line);
  border-left: 4px solid var(--accent);
  border-radius: 8px;
  padding: 1.35rem 1.4rem;
  margin: 0.5rem 0 1rem;
  color: var(--ink);
  font-size: 1.02rem;
  line-height: 1.55;
}
.welcome strong {
  color: var(--accent);
  font-weight: 700;
}

.section-label {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 0.5rem 0 0.65rem;
}

.emotion-card {
  border-radius: 8px;
  padding: 1.15rem 1.25rem;
  margin: 0.4rem 0 0.85rem;
  background: var(--card);
  border: 1px solid var(--line);
  border-left: 5px solid var(--accent);
}
.emotion-head {
  display: flex;
  align-items: center;
  gap: 0.9rem;
}
.emotion-emoji {
  width: 52px;
  height: 52px;
  border-radius: 8px;
  display: grid;
  place-items: center;
  font-size: 1.55rem;
  border: 1px solid var(--line);
  background: var(--accent-soft);
}
.emotion-label {
  font-family: 'Libre Baskerville', Georgia, serif;
  font-size: 1.45rem;
  font-weight: 700;
  color: var(--ink);
  margin: 0;
  text-transform: capitalize;
}
.emotion-sub {
  margin: 0.2rem 0 0;
  color: var(--ink-soft);
  font-size: 0.95rem;
  font-weight: 500;
}
.confidence-row {
  margin-top: 1rem;
  display: flex;
  align-items: center;
  gap: 0.75rem;
}
.confidence-bar {
  flex: 1;
  height: 8px;
  border-radius: 2px;
  background: #e4e7ec;
  overflow: hidden;
}
.confidence-fill {
  height: 100%;
  border-radius: 2px;
}
.confidence-pct {
  font-weight: 700;
  font-size: 1rem;
  min-width: 3.4rem;
  text-align: right;
  color: var(--ink);
  font-variant-numeric: tabular-nums;
}
.alts {
  margin-top: 0.95rem;
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
}
.alt-chip {
  font-size: 0.8rem;
  font-weight: 600;
  padding: 0.32rem 0.7rem;
  border-radius: 6px;
  background: #f0efe9;
  color: var(--ink-soft);
  border: 1px solid var(--line);
}

.side-title {
  font-family: 'Libre Baskerville', Georgia, serif;
  font-size: 1.35rem;
  font-weight: 700;
  margin: 0 0 0.35rem 0;
  color: #ffffff !important;
}
.side-chip {
  display: inline-block;
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 0.28rem 0.6rem;
  border-radius: 4px;
  background: #243040;
  border: 1px solid #3a4658;
  color: #c5ced9 !important;
  margin-bottom: 0.85rem;
}
.pipeline-step {
  font-size: 0.84rem;
  line-height: 1.45;
  padding: 0.55rem 0.65rem;
  margin-bottom: 0.4rem;
  border-radius: 6px;
  background: #1e2530;
  border-left: 3px solid #6b8f9c;
  color: #d5dae3 !important;
}

[data-testid="stChatMessage"] {
  background: transparent !important;
  padding: 0.35rem 0 !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong {
  color: var(--ink) !important;
  font-size: 1.02rem !important;
  line-height: 1.6 !important;
}

/* Chat input — net, okunaklı */
div[data-testid="stChatInput"] {
  background: transparent !important;
}
div[data-testid="stChatInput"] textarea {
  font-family: 'IBM Plex Sans', sans-serif !important;
  font-size: 1rem !important;
  color: var(--ink) !important;
}

/* Butonlar: keskin, profesyonel (pill/neon yok) */
.stButton > button {
  border-radius: 8px !important;
  border: 1.5px solid var(--ink) !important;
  background: var(--card) !important;
  color: var(--ink) !important;
  font-weight: 600 !important;
  font-size: 0.9rem !important;
  padding: 0.55rem 0.9rem !important;
  box-shadow: none !important;
  transition: background 0.15s ease, color 0.15s ease;
}
.stButton > button:hover {
  background: var(--accent) !important;
  color: #fff !important;
  border-color: var(--accent) !important;
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
  background: var(--accent) !important;
  color: #fff !important;
  border-color: var(--accent) !important;
}

[data-testid="stCaptionContainer"] {
  color: var(--muted) !important;
  font-weight: 500 !important;
}
</style>
"""


def emotion_meta(label: str) -> dict[str, str]:
    if label in EMOTION_META:
        return EMOTION_META[label]
    for key, meta in EMOTION_META.items():
        if key.casefold() == label.casefold():
            return meta
    return {"emoji": "◈", "color": "#1c4b5a", "hint": "Algılanan duygu"}


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

    active = "sklearn" if sk_bundle is not None else "fuzzy"
    return sk_bundle, fuzzy, vectorizer, label_encoder, active, None


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
        df["model"] = df["model"].map(
            lambda x: MEMBERSHIP_TR.get(x, str(x).replace("_", " "))
        )
    return df


def analyze_text(text: str, sk_bundle, fuzzy_model, vectorizer, label_encoder, active: str) -> dict[str, Any]:
    cleaned = remove_stopwords(clean_text(text))
    if not cleaned.strip():
        return {"ok": False, "error": "Metin çok kısa kaldı. Biraz daha uzun bir cümle dener misin?"}

    vector = vectorizer.transform([cleaned])
    id_to_label = label_encoder["id_to_label"]

    if active == "sklearn" and sk_bundle is not None:
        model = sk_bundle["model"]
        dense = vector.toarray()
        pred = int(model.predict(dense)[0])
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(dense)[0]
        else:
            scores = model.decision_function(dense)[0]
            exp = np.exp(scores - np.max(scores))
            probabilities = exp / exp.sum()
        label = id_to_label[pred]
        conf = float(probabilities[pred])
    else:
        dense = vector.toarray()
        prediction, confidence = fuzzy_model.predict(dense)
        probabilities = fuzzy_model.predict_proba(dense)[0]
        label = id_to_label[int(prediction[0])]
        conf = float(confidence[0])
        if conf < 1e-6:
            conf = float(np.max(probabilities))
            label = id_to_label[int(np.argmax(probabilities))]

    ranked = sorted(
        ((id_to_label[i], float(probabilities[i])) for i in range(len(probabilities))),
        key=lambda item: item[1],
        reverse=True,
    )
    return {
        "ok": True,
        "label": label,
        "confidence": conf,
        "ranked": ranked,
        "cleaned": cleaned,
    }


def render_assistant_reply(result: dict[str, Any]) -> None:
    if not result.get("ok"):
        st.markdown(result.get("error", "Bir şeyler ters gitti."))
        return

    meta = emotion_meta(result["label"])
    conf_pct = max(0.0, min(1.0, result["confidence"])) * 100
    alts = [item for item in result["ranked"][1:4] if item[1] >= 0.05]

    alt_html = "".join(
        f'<span class="alt-chip">{emotion_meta(name)["emoji"]} {name} · {prob:.0%}</span>'
        for name, prob in alts
    ) or '<span class="alt-chip">Diğer duygular zayıf</span>'

    st.markdown(
        f"""
        <div class="emotion-card" style="border-left-color:{meta['color']};">
          <div class="emotion-head">
            <div class="emotion-emoji" style="background:{meta['color']}14;border-color:{meta['color']}33;">{meta['emoji']}</div>
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

    lead = (
        f"Bu metinde baskın duygu **{result['label']}** "
        f"(güven ≈ %{conf_pct:.0f})."
    )
    if alts:
        alt_names = ", ".join(name for name, _ in alts[:2])
        lead += f" Yanında hafif {alt_names} izleri de var."
    st.markdown(lead)


def render_project_sidebar(sk_bundle, fuzzy_model, label_encoder, active: str) -> None:
    report = load_eval_report()
    comparison_df = load_model_comparison()

    st.markdown('<p class="side-title">Proje Paneli</p>', unsafe_allow_html=True)
    st.markdown('<span class="side-chip">Duygu Analizi · NLP</span>', unsafe_allow_html=True)
    st.caption("Veri seti, model performansı ve sistem mimarisi.")

    st.subheader("Aktif model")
    if active == "sklearn" and sk_bundle:
        name = str(sk_bundle.get("name", "sklearn")).replace("_", " ")
        m1, m2 = st.columns(2)
        m1.metric("Tip", name.title())
        m2.metric("Sınıf", len(label_encoder["id_to_label"]) if label_encoder else 10)
        st.caption("TF-IDF + dengeli LinearSVC — yüksek doğruluk.")
    elif fuzzy_model:
        m1, m2 = st.columns(2)
        m1.metric("Üyelik", MEMBERSHIP_TR.get(fuzzy_model.membership_type, fuzzy_model.membership_type))
        m2.metric("Kurallar", f"{len(fuzzy_model.rules_ or []):,}")
        st.caption("Bulanık mantık — yorumlanabilir kurallar.")

    with st.expander("Performans", expanded=True):
        if report:
            c1, c2 = st.columns(2)
            c1.metric("Accuracy", f"{float(report['accuracy']) * 100:.1f}%")
            c2.metric("F1", f"{float(report['f1_score']):.3f}")
            c3, c4 = st.columns(2)
            c3.metric("Precision", f"{float(report['precision']):.3f}")
            c4.metric("Recall", f"{float(report['recall']):.3f}")
            conf = report.get("mean_confidence", report.get("confidence_mean", 0))
            c5, c6 = st.columns(2)
            c5.metric("R²", f"{float(report.get('r_squared', 0)):.3f}")
            c6.metric("Güven", f"{float(conf) * 100:.1f}%")
            if "sklearn_accuracy" in report:
                st.caption(
                    f"Sklearn %{float(report['sklearn_accuracy'])*100:.1f} · "
                    f"Fuzzy %{float(report.get('fuzzy_accuracy', 0))*100:.1f}"
                )
        else:
            st.caption("Metrik dosyası yok.")

    with st.expander("Veri seti", expanded=True):
        corpus = load_corpus_stats()
        if corpus:
            st.metric("Kütüphane", f"{corpus['total']:,}")
            for src in DATASET_SOURCES:
                st.markdown(f"- {src}")
            st.bar_chart(corpus["counts"].rename_axis("Duygu").reset_index(name="Adet").set_index("Duygu"))
        else:
            st.caption("Korpus CSV bulunamadı.")
        if report and "train_size" in report:
            st.caption(f"Eğitim: {int(report['train_size']):,} / Test: {int(report['test_size']):,}")

    with st.expander("Mimari"):
        for step in [
            "1. Metin temizleme + Türkçe stop-words",
            "2. TF-IDF (n-gram 1–3)",
            "3a. Bulanık kurallar (yorumlanabilir)",
            "3b. LinearSVC (aktif tahmin)",
            "4. Sohbet arayüzüne bağlama",
        ]:
            st.markdown(f'<div class="pipeline-step">{step}</div>', unsafe_allow_html=True)

    with st.expander("Model karşılaştırma"):
        if comparison_df is not None and not comparison_df.empty:
            show = comparison_df.copy()
            label_col = "model" if "model" in show.columns else "Fonksiyon"
            show["Accuracy"] = (show["accuracy"] * 100).map(lambda x: f"{x:.1f}%")
            show["F1"] = show["f1_score"].map(lambda x: f"{x:.3f}")
            cols = [c for c in [label_col, "Accuracy", "F1"] if c in show.columns]
            st.dataframe(show[cols], hide_index=True, width="stretch")
            chart = show.set_index(label_col)[["accuracy", "f1_score"]]
            chart.columns = ["Accuracy", "F1"]
            st.bar_chart(chart)
        else:
            st.caption("Karşılaştırma yok.")

    with st.expander("Duygu sınıfları"):
        if label_encoder:
            for _, label in sorted(label_encoder["id_to_label"].items()):
                meta = emotion_meta(label)
                st.markdown(f"{meta['emoji']} **{label}** — {meta['hint']}")

    with st.expander("Görseller"):
        available = [(n, p) for n, p in VISUALS if os.path.exists(p)]
        if available:
            choice = st.selectbox("Grafik", [n for n, _ in available], label_visibility="collapsed")
            st.image(dict(available)[choice], width="stretch")
        else:
            st.caption("Görsel yok.")

    st.caption("Eğitim projesi · Fuzzy + TF-IDF + Sklearn")


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    sk_bundle, fuzzy_model, vectorizer, label_encoder, active, error = load_runtime_models()

    with st.sidebar:
        render_project_sidebar(sk_bundle, fuzzy_model, label_encoder, active or "fuzzy")

    top_l, top_r = st.columns([5, 1.2])
    with top_l:
        st.markdown(
            """
            <div class="brand">
              <div class="brand-mark">D</div>
              <div>
                <h1>DuyguAsistan</h1>
                <p>Türkçe metinlerde duygu analizi</p>
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
        st.error("Model yüklenemedi. `python train_model.py` çalıştırın.")
        st.caption(error or "")
        return

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="welcome">
              Merhaba — ben <strong>DuyguAsistan</strong>.
              Bir cümle yaz; duygusunu net şekilde çıkarayım.
              Solda proje verisi ve performans özeti var.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<p class="section-label">Örnekler</p>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, starter in enumerate(STARTERS):
            if cols[i % 2].button(starter, key=f"starter_{i}", width="stretch"):
                st.session_state.pending_prompt = starter
                st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="●" if message["role"] == "user" else "◈"):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_assistant_reply(message["result"])

    prompt = st.chat_input("Cümlenizi yazın…")
    if "pending_prompt" in st.session_state:
        prompt = st.session_state.pop("pending_prompt")

    if prompt and prompt.strip():
        user_text = prompt.strip()
        st.session_state.messages.append({"role": "user", "content": user_text})

        with st.chat_message("user", avatar="●"):
            st.markdown(user_text)

        with st.chat_message("assistant", avatar="◈"):
            with st.spinner("Analiz ediliyor…"):
                result = analyze_text(
                    user_text, sk_bundle, fuzzy_model, vectorizer, label_encoder, active
                )
            render_assistant_reply(result)

        st.session_state.messages.append({"role": "assistant", "result": result})


if __name__ == "__main__":
    main()
