"""
DuyguAsistan — sohbet arayüzü + proje sunum paneli
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import streamlit as st

from data_preprocessing import clean_text, load_preprocessing_artifacts, remove_stopwords
from fuzzy_sentiment import FuzzySentimentClassifier

st.set_page_config(
    page_title="DuyguAsistan",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sabitler ─────────────────────────────────────────────────────────────────
EMOTION_META = {
    "mutlu": {"emoji": "😊", "color": "#0d9488", "hint": "Pozitif ve neşeli bir ton"},
    "üzgün": {"emoji": "😢", "color": "#64748b", "hint": "Üzüntü veya hayal kırıklığı"},
    "korku": {"emoji": "😨", "color": "#7c3aed", "hint": "Endişe veya korku hissi"},
    "kızgın": {"emoji": "😠", "color": "#dc2626", "hint": "Öfke veya rahatsızlık"},
    "surpriz": {"emoji": "😲", "color": "#ea580c", "hint": "Şaşkınlık / beklenmedik durum"},
    "Şaşırmış": {"emoji": "😯", "color": "#d97706", "hint": "Hayret ve şaşırma"},
    "Heyecanlı": {"emoji": "🎉", "color": "#db2777", "hint": "Enerji ve heyecan"},
    "Meraklı": {"emoji": "🔍", "color": "#0284c7", "hint": "Merak ve ilgi"},
    "Sorgulayıcı": {"emoji": "🤔", "color": "#4f46e5", "hint": "Soru sorma / sorgulama"},
    "Umutsuz": {"emoji": "😔", "color": "#57534e", "hint": "Umutsuzluk veya yılgınlık"},
}

MEMBERSHIP_TR = {
    "triangular": "Üçgen",
    "trapezoidal": "Yamuk",
    "sigmoid": "Sigmoid",
    "gaussian": "Gauss",
    "bell": "Bell",
}

CLASS_F1 = [
    ("kızgın", 0.89),
    ("surpriz", 0.90),
    ("üzgün", 0.79),
    ("Sorgulayıcı", 0.84),
    ("Şaşırmış", 0.78),
    ("mutlu", 0.72),
    ("Heyecanlı", 0.61),
    ("korku", 0.60),
    ("Meraklı", 0.58),
    ("Umutsuz", 0.13),
]

DATASET_COUNTS = [
    ("kızgın", 800),
    ("korku", 800),
    ("mutlu", 800),
    ("surpriz", 800),
    ("üzgün", 800),
    ("Heyecanlı", 250),
    ("Umutsuz", 249),
    ("Sorgulayıcı", 244),
    ("Şaşırmış", 197),
    ("Meraklı", 173),
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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');

html, body, [class*="css"] {
  font-family: 'DM Sans', system-ui, sans-serif;
}

.stApp {
  background:
    radial-gradient(1200px 600px at 10% -10%, #cfeee8 0%, transparent 55%),
    radial-gradient(900px 500px at 100% 0%, #fde8d8 0%, transparent 50%),
    linear-gradient(180deg, #f7faf9 0%, #eef4f2 100%);
}

[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0f766e 0%, #134e4a 42%, #0f172a 100%);
  border-right: none;
}
section[data-testid="stSidebar"] * {
  color: #ecfdf5 !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] label {
  color: #d1fae5 !important;
}
section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
  color: #fff !important;
  font-weight: 700 !important;
}
section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
  color: #a7f3d0 !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
  background: rgba(255,255,255,0.12) !important;
  border-color: rgba(255,255,255,0.2) !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] {
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 12px;
}
section[data-testid="stSidebar"] img {
  border-radius: 10px;
  border: 1px solid rgba(255,255,255,0.15);
}

.block-container {
  padding-top: 1.1rem !important;
  padding-bottom: 6rem !important;
  max-width: 860px !important;
}

.brand {
  display: flex;
  align-items: center;
  gap: 0.85rem;
  margin-bottom: 0.35rem;
}
.brand-mark {
  width: 48px; height: 48px;
  border-radius: 16px;
  background: linear-gradient(145deg, #0f766e, #14b8a6);
  color: #fff;
  display: grid; place-items: center;
  font-size: 1.35rem;
  box-shadow: 0 10px 24px rgba(15, 118, 110, 0.28);
}
.brand h1 {
  font-family: 'Fraunces', Georgia, serif;
  font-size: 1.65rem;
  font-weight: 700;
  margin: 0;
  color: #0f172a;
  letter-spacing: -0.02em;
}
.brand p {
  margin: 0.1rem 0 0;
  color: #475569;
  font-size: 0.92rem;
}

.welcome {
  background: rgba(255,255,255,0.78);
  border: 1px solid rgba(15, 118, 110, 0.12);
  backdrop-filter: blur(10px);
  border-radius: 20px;
  padding: 1.25rem 1.35rem;
  margin: 1rem 0 0.75rem;
  box-shadow: 0 8px 28px rgba(15, 23, 42, 0.04);
  color: #0f172a;
}
.welcome strong { color: #0f766e; }

.emotion-card {
  border-radius: 18px;
  padding: 1rem 1.1rem;
  margin: 0.35rem 0 0.75rem;
  background: #fff;
  border: 1px solid rgba(15, 23, 42, 0.06);
  box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05);
}
.emotion-head {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}
.emotion-emoji {
  width: 52px; height: 52px;
  border-radius: 14px;
  display: grid; place-items: center;
  font-size: 1.6rem;
}
.emotion-label {
  font-family: 'Fraunces', Georgia, serif;
  font-size: 1.35rem;
  font-weight: 700;
  color: #0f172a;
  margin: 0;
}
.emotion-sub {
  margin: 0.15rem 0 0;
  color: #475569;
  font-size: 0.9rem;
}
.confidence-row {
  margin-top: 0.85rem;
  display: flex;
  align-items: center;
  gap: 0.65rem;
}
.confidence-bar {
  flex: 1;
  height: 10px;
  border-radius: 999px;
  background: #e2e8f0;
  overflow: hidden;
}
.confidence-fill {
  height: 100%;
  border-radius: 999px;
}
.confidence-pct {
  font-weight: 700;
  font-size: 0.95rem;
  min-width: 3.2rem;
  text-align: right;
}
.alts {
  margin-top: 0.85rem;
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
}
.alt-chip {
  font-size: 0.78rem;
  font-weight: 600;
  padding: 0.28rem 0.65rem;
  border-radius: 999px;
  background: #f1f5f9;
  color: #334155;
  border: 1px solid #e2e8f0;
}

.side-title {
  font-family: 'Fraunces', Georgia, serif;
  font-size: 1.25rem;
  font-weight: 700;
  margin: 0 0 0.25rem 0;
}
.side-chip {
  display: inline-block;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  padding: 0.2rem 0.55rem;
  border-radius: 999px;
  background: rgba(52, 211, 153, 0.2);
  border: 1px solid rgba(167, 243, 208, 0.35);
  margin-bottom: 0.75rem;
}
.pipeline-step {
  font-size: 0.82rem;
  line-height: 1.45;
  padding: 0.45rem 0.55rem;
  margin-bottom: 0.35rem;
  border-radius: 10px;
  background: rgba(255,255,255,0.08);
  border-left: 3px solid #34d399;
}

[data-testid="stChatMessage"] {
  background: transparent !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong {
  color: #0f172a !important;
  font-size: 0.98rem;
  line-height: 1.55;
}

div[data-testid="stChatInput"] textarea {
  font-family: 'DM Sans', system-ui, sans-serif !important;
}

.stButton > button {
  border-radius: 999px !important;
  border: 1px solid rgba(15, 118, 110, 0.18) !important;
  background: rgba(255,255,255,0.9) !important;
  color: #0f766e !important;
  font-weight: 600 !important;
  box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}
.stButton > button:hover {
  border-color: #0f766e !important;
  background: #ecfdf5 !important;
}
</style>
"""


def emotion_meta(label: str) -> dict[str, str]:
    if label in EMOTION_META:
        return EMOTION_META[label]
    for key, meta in EMOTION_META.items():
        if key.casefold() == label.casefold():
            return meta
    return {"emoji": "💬", "color": "#0f766e", "hint": "Algılanan duygu"}


@st.cache_resource
def load_model_and_preprocessing(model_path: str, preprocessing_path: str):
    if not os.path.exists(model_path):
        return None, None, None, "Model dosyası bulunamadı."
    if not os.path.exists(preprocessing_path):
        return None, None, None, "Ön işleme dosyası bulunamadı."
    try:
        model = FuzzySentimentClassifier.load(model_path)
        vectorizer, label_encoder = load_preprocessing_artifacts(preprocessing_path)
        return model, vectorizer, label_encoder, None
    except Exception as exc:  # noqa: BLE001
        return None, None, None, str(exc)


@st.cache_data
def load_eval_report(path: str = "evaluation_report.csv") -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    row = pd.read_csv(path).iloc[0].to_dict()
    return row


@st.cache_data
def load_membership_comparison(path: str = "membership_function_comparison.csv") -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path).copy()
    df["Fonksiyon"] = df["membership_type"].map(lambda x: MEMBERSHIP_TR.get(x, x))
    return df


def analyze_text(text: str, model, vectorizer, label_encoder) -> dict[str, Any]:
    cleaned = remove_stopwords(clean_text(text))
    if not cleaned.strip():
        return {"ok": False, "error": "Metin çok kısa kaldı. Biraz daha uzun bir cümle dener misin?"}

    vector = vectorizer.transform([cleaned]).toarray()
    prediction, confidence = model.predict(vector)
    probabilities = model.predict_proba(vector)[0]
    id_to_label = label_encoder["id_to_label"]

    ranked = sorted(
        ((id_to_label[i], float(probabilities[i])) for i in range(len(probabilities))),
        key=lambda item: item[1],
        reverse=True,
    )
    label = id_to_label[int(prediction[0])]
    conf = float(confidence[0])
    if conf < 1e-6 and ranked:
        conf = ranked[0][1]
        label = ranked[0][0]

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
        <div class="emotion-card">
          <div class="emotion-head">
            <div class="emotion-emoji" style="background:{meta['color']}18">{meta['emoji']}</div>
            <div>
              <p class="emotion-label">{result['label']}</p>
              <p class="emotion-sub">{meta['hint']}</p>
            </div>
          </div>
          <div class="confidence-row">
            <div class="confidence-bar">
              <div class="confidence-fill" style="width:{conf_pct:.1f}%;background:{meta['color']};"></div>
            </div>
            <div class="confidence-pct" style="color:{meta['color']};">%{conf_pct:.0f}</div>
          </div>
          <div class="alts">{alt_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    lead = (
        f"Bu metinde baskın duygu **{result['label']}** gibi duruyor "
        f"(güven ≈ %{conf_pct:.0f})."
    )
    if alts:
        alt_names = ", ".join(name for name, _ in alts[:2])
        lead += f" Yanında hafif {alt_names} izleri de var."
    st.markdown(lead)


def render_project_sidebar(model, label_encoder) -> None:
    report = load_eval_report()
    membership_df = load_membership_comparison()

    st.markdown('<p class="side-title">Proje Paneli</p>', unsafe_allow_html=True)
    st.markdown('<span class="side-chip">Bulanık Mantık · NLP</span>', unsafe_allow_html=True)
    st.caption(
        "Türkçe tweet duygu analizi — sohbet arayüzünün arkasındaki "
        "eğitilmiş bulanık mantık modeli."
    )

    # Canlı model özeti
    st.subheader("Model")
    if model:
        mem_tr = MEMBERSHIP_TR.get(model.membership_type, model.membership_type)
        n_rules = len(model.rules_) if model.rules_ else 0
        m1, m2 = st.columns(2)
        m1.metric("Üyelik", mem_tr)
        m2.metric("Kurallar", f"{n_rules:,}")
        m3, m4 = st.columns(2)
        m3.metric("Özellik", model.n_features)
        m4.metric("Sınıf", model.n_classes)
        st.caption(
            "En iyi üyelik: **Üçgen**. "
            "Varyans + F-score ile 150 özellik; kural başına en fazla 5 özellik."
        )

    # Performans
    with st.expander("Performans metrikleri", expanded=True):
        if report:
            c1, c2 = st.columns(2)
            c1.metric("Accuracy", f"{float(report['accuracy']) * 100:.1f}%")
            c2.metric("F1-Score", f"{float(report['f1_score']):.3f}")
            c3, c4 = st.columns(2)
            c3.metric("Precision", f"{float(report['precision']):.3f}")
            c4.metric("Recall", f"{float(report['recall']):.3f}")
            c5, c6 = st.columns(2)
            c5.metric("R²", f"{float(report['r_squared']):.3f}")
            c6.metric("Ort. güven", f"{float(report['mean_confidence']) * 100:.1f}%")
            st.caption("İlk model ~%27 accuracy → iyileştirmelerle ~%73.")
        else:
            st.caption("evaluation_report.csv bulunamadı.")

        st.markdown("**Sınıf bazlı F1**")
        class_df = pd.DataFrame(CLASS_F1, columns=["Duygu", "F1"]).sort_values("F1", ascending=False)
        st.dataframe(class_df, hide_index=True, width="stretch")

    # Veri seti
    with st.expander("Veri seti"):
        st.markdown(
            """
            - **Kaynak:** `TurkishTweets.xlsx`
            - **Toplam:** 5.113 tweet
            - **Train / Test:** %80 / %20 (stratified)
            - **Dil:** Türkçe
            """
        )
        data_df = pd.DataFrame(DATASET_COUNTS, columns=["Duygu", "Adet"])
        st.bar_chart(data_df.set_index("Duygu"))
        st.caption("Ana 5 duygu dengeli; diğerleri daha az örnek içeriyor.")

    # Mimari
    with st.expander("Sistem mimarisi"):
        steps = [
            "1. Temizleme — URL, @mention, noktalama; Türkçe stop-words",
            "2. TF-IDF — 1000 özellik, n-gram (1–3), sublinear TF",
            "3. Özellik seçimi — varyans × F-score → top 150",
            "4. Bulanıklaştırma — düşük / orta / yüksek kümeler",
            "5. Kural çıkarımı — örneklerden birleştirilmiş kurallar",
            "6. Çıkarım — kural skoru × güven × support → duygu",
        ]
        for step in steps:
            st.markdown(f'<div class="pipeline-step">{step}</div>', unsafe_allow_html=True)
        st.caption(
            "Min. üyelik eşiği 0.15 · min. kural güveni %30 · support faktörü aktif."
        )

    # Üyelik karşılaştırması
    with st.expander("Üyelik fonksiyonları"):
        if membership_df is not None:
            show = membership_df.copy()
            show["Accuracy"] = (show["accuracy"] * 100).map(lambda x: f"{x:.1f}%")
            show["F1"] = show["f1_score"].map(lambda x: f"{x:.3f}")
            show["R²"] = show["r_squared"].map(lambda x: f"{x:.3f}")
            st.dataframe(
                show[["Fonksiyon", "Accuracy", "F1", "R²"]],
                hide_index=True,
                width="stretch",
            )
            chart_src = show.set_index("Fonksiyon")[["accuracy", "f1_score"]]
            chart_src.columns = ["Accuracy", "F1"]
            st.bar_chart(chart_src)
            st.caption("Bu veri setinde Üçgen net şekilde önde.")
        else:
            st.caption("membership_function_comparison.csv bulunamadı.")

    # Duygu sınıfları
    with st.expander("10 duygu sınıfı"):
        if label_encoder:
            for idx, label in sorted(label_encoder["id_to_label"].items()):
                meta = emotion_meta(label)
                st.markdown(f"{meta['emoji']} **{label}** — {meta['hint']}")

    # Grafikler
    with st.expander("Görseller", expanded=True):
        available = [(name, path) for name, path in VISUALS if os.path.exists(path)]
        if not available:
            st.caption("Gorseller/ klasöründe görsel yok.")
        else:
            names = [name for name, _ in available]
            choice = st.selectbox("Grafik seç", names, label_visibility="collapsed")
            path = dict(available)[choice]
            st.image(path, width="stretch")
            captions = {
                "Metrikler": "Accuracy, Precision, Recall, F1 ve R² özeti.",
                "Üyelik karşılaştırma": "5 üyelik fonksiyonunun yan yana performansı.",
                "Confusion matrix": "Hangi sınıfların karıştığını gösterir.",
                "Sınıf dağılımı": "Gerçek vs tahmin edilen sınıf adetleri.",
                "Güven dağılımı": "Test tahminlerindeki güven skorları.",
            }
            st.caption(captions.get(choice, ""))

    st.markdown("---")
    st.caption("Eğitim amaçlı proje · Fuzzy Logic + TF-IDF + Streamlit")


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    model, vectorizer, label_encoder, error = load_model_and_preprocessing(
        "best_fuzzy_model.pkl",
        "preprocessing_artifacts.pkl",
    )

    with st.sidebar:
        render_project_sidebar(model, label_encoder)

    # Ana sohbet alanı
    top_l, top_r = st.columns([5, 1.15])
    with top_l:
        st.markdown(
            """
            <div class="brand">
              <div class="brand-mark">💬</div>
              <div>
                <h1>DuyguAsistan</h1>
                <p>Metnini yaz, duygusunu anında konuşalım.</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_r:
        if st.button("Yeni sohbet", width="stretch", help="Sohbeti temizle"):
            st.session_state.messages = []
            st.rerun()

    if error or not model:
        st.error("Model yüklenemedi. Önce `python train_model.py` çalıştır.")
        st.caption(error or "")
        return

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="welcome">
              Merhaba! Ben <strong>DuyguAsistan</strong>.
              Türkçe bir cümle yaz; soldaki panelde modelin teknik arka planını,
              burada ise sohbet gibi duygu analizini görürsün.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Hızlı dene")
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

    prompt = st.chat_input("Bir cümle yaz… örn. Bugün harika geçti")
    if "pending_prompt" in st.session_state:
        prompt = st.session_state.pop("pending_prompt")

    if prompt and prompt.strip():
        user_text = prompt.strip()
        st.session_state.messages.append({"role": "user", "content": user_text})

        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_text)

        with st.chat_message("assistant", avatar="💬"):
            with st.spinner("Duyguyu okuyorum…"):
                result = analyze_text(user_text, model, vectorizer, label_encoder)
            render_assistant_reply(result)

        st.session_state.messages.append({"role": "assistant", "result": result})


if __name__ == "__main__":
    main()
