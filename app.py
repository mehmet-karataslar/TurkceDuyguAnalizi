"""
Streamlit Web Arayüzü
Kullanıcı Türkçe cümle girerek duygu analizi yapabilir
"""

import streamlit as st
import numpy as np
import pandas as pd
from fuzzy_sentiment import FuzzySentimentClassifier
from data_preprocessing import load_preprocessing_artifacts, clean_text, remove_stopwords
import os
import pickle


# Sayfa yapılandırması
st.set_page_config(
    page_title="Bulanık Mantık Duygu Analizi",
    page_icon="😊",
    layout="wide"
)

# Başlık
st.title("😊 Bulanık Mantık ile Türkçe Duygu Analizi")
st.markdown("---")

# Sidebar - Model yükleme
st.sidebar.header("Model Ayarları")

model_path = st.sidebar.text_input("Model Dosyası", value="best_fuzzy_model.pkl")
preprocessing_path = st.sidebar.text_input("Ön İşleme Dosyası", value="preprocessing_artifacts.pkl")

@st.cache_resource
def load_model_and_preprocessing(model_path, preprocessing_path):
    """
    Modeli ve ön işleme sonuçlarını yükle (cache ile)
    """
    try:
        if not os.path.exists(model_path):
            return None, None, None, "Model dosyası bulunamadı!"
        
        if not os.path.exists(preprocessing_path):
            return None, None, None, "Ön işleme dosyası bulunamadı!"
        
        model = FuzzySentimentClassifier.load(model_path)
        vectorizer, label_encoder = load_preprocessing_artifacts(preprocessing_path)
        
        return model, vectorizer, label_encoder, None
    except Exception as e:
        return None, None, None, f"Hata: {str(e)}"


# Modeli yükle
model, vectorizer, label_encoder, error = load_model_and_preprocessing(model_path, preprocessing_path)

if error:
    st.sidebar.error(error)
    st.warning("⚠️ Model yüklenemedi. Lütfen model dosyalarının doğru yolda olduğundan emin olun.")
    st.info("💡 İpucu: Önce `train_model.py` dosyasını çalıştırarak modeli eğitin.")
else:
    st.sidebar.success("✅ Model başarıyla yüklendi!")
    
    # Model bilgileri
    if model:
        st.sidebar.markdown("### Model Bilgileri")
        st.sidebar.write(f"**Üyelik Fonksiyonu:** {model.membership_type}")
        st.sidebar.write(f"**Özellik Sayısı:** {model.n_features}")
        st.sidebar.write(f"**Sınıf Sayısı:** {model.n_classes}")
        
        # Sınıf isimleri
        if label_encoder:
            id_to_label = label_encoder['id_to_label']
            st.sidebar.markdown("### Sınıflar")
            for idx, label in sorted(id_to_label.items()):
                st.sidebar.write(f"{idx}: {label}")

# Ana içerik
if model and vectorizer and label_encoder:
    # Kullanıcı girişi
    st.header("📝 Duygu Analizi Yap")
    
    # Metin girişi
    user_input = st.text_area(
        "Türkçe bir cümle veya tweet girin:",
        height=150,
        placeholder="Örnek: Bugün çok mutlu bir gün geçirdim! 🎉"
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        analyze_button = st.button("🔍 Analiz Et", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🗑️ Temizle", use_container_width=True):
            st.rerun()
    
    # Analiz sonuçları
    if analyze_button and user_input.strip():
        with st.spinner("Analiz yapılıyor..."):
            try:
                # Metni temizle
                cleaned_text = clean_text(user_input)
                cleaned_text = remove_stopwords(cleaned_text)
                
                if not cleaned_text.strip():
                    st.warning("⚠️ Metin temizlendikten sonra boş kaldı. Lütfen daha uzun bir metin girin.")
                else:
                    # TF-IDF vektörizasyonu
                    text_vector = vectorizer.transform([cleaned_text]).toarray()
                    
                    # Tahmin yap
                    prediction, confidence = model.predict(text_vector)
                    probabilities = model.predict_proba(text_vector)[0]
                    
                    # Sonuçları göster
                    st.markdown("---")
                    st.header("📊 Analiz Sonuçları")
                    
                    # Ana sonuç
                    id_to_label = label_encoder['id_to_label']
                    predicted_label = id_to_label[prediction[0]]
                    confidence_score = confidence[0]
                    
                    # Sonuç kartı
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Tahmin Edilen Duygu", predicted_label)
                    
                    with col2:
                        st.metric("Güven Skoru", f"{confidence_score:.2%}")
                    
                    with col3:
                        # En yüksek olasılık
                        max_prob_idx = np.argmax(probabilities)
                        max_prob_label = id_to_label[max_prob_idx]
                        st.metric("En Yüksek Olasılık", max_prob_label)
                    
                    # Olasılık dağılımı
                    st.subheader("📈 Sınıf Olasılıkları")
                    
                    # DataFrame oluştur
                    prob_df = pd.DataFrame({
                        'Duygu': [id_to_label[i] for i in range(len(probabilities))],
                        'Olasılık': probabilities,
                        'Yüzde': [f"{p:.2%}" for p in probabilities]
                    }).sort_values('Olasılık', ascending=False)
                    
                    # Bar chart
                    st.bar_chart(prob_df.set_index('Duygu')['Olasılık'])
                    
                    # Tablo
                    st.dataframe(
                        prob_df[['Duygu', 'Yüzde']].style.format({'Yüzde': '{:.2%}'}),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Görsel gösterim
                    st.subheader("🎯 Görsel Gösterim")
                    
                    # Progress bar'lar
                    for idx, label in sorted(id_to_label.items()):
                        prob = probabilities[idx]
                        color = "🟢" if prob > 0.3 else "🟡" if prob > 0.15 else "🔴"
                        st.write(f"{color} **{label}**")
                        st.progress(prob, text=f"{prob:.2%}")
                    
                    # Temizlenmiş metin (opsiyonel)
                    with st.expander("🔍 Temizlenmiş Metin"):
                        st.write(cleaned_text)
                    
            except Exception as e:
                st.error(f"❌ Hata oluştu: {str(e)}")
                st.exception(e)
    
    elif analyze_button:
        st.warning("⚠️ Lütfen bir metin girin!")
    
    # Örnekler
    st.markdown("---")
    st.header("💡 Örnek Metinler")
    
    example_texts = [
        "Bugün çok mutlu bir gün geçirdim!",
        "Bu haber beni çok üzdü.",
        "Aniden karşımda belirdi, çok korktum!",
        "Vay be, hiç beklemiyordum bu kadar güzel olacağını!",
        "Harika bir film izledim, çok eğlendim.",
        "İşten çıktım, yorgunum ama mutluyum."
    ]
    
    cols = st.columns(3)
    for i, example in enumerate(example_texts):
        with cols[i % 3]:
            if st.button(f"📌 {example[:30]}...", key=f"example_{i}", use_container_width=True):
                st.session_state.example_text = example
                st.rerun()
    
    # Eğer örnek seçildiyse
    if 'example_text' in st.session_state:
        user_input = st.text_area(
            "Türkçe bir cümle veya tweet girin:",
            value=st.session_state.example_text,
            height=150
        )
        del st.session_state.example_text

else:
    # Model yüklenemedi durumu
    st.info("""
    ### 🚀 Başlamak İçin
    
    1. **Model Eğitimi**: Terminal'de şu komutu çalıştırın:
       ```bash
       python train_model.py
       ```
    
    2. **Model Değerlendirme** (Opsiyonel):
       ```bash
       python evaluate_model.py
       ```
    
    3. Model eğitildikten sonra bu sayfayı yenileyin.
    """)
    
    st.markdown("---")
    st.subheader("📚 Proje Hakkında")
    st.markdown("""
    Bu proje, **Bulanık Mantık (Fuzzy Logic)** kullanarak Türkçe tweet'lerde duygu analizi yapar.
    
    **Özellikler:**
    - 🎯 4 sınıflı duygu analizi (mutlu, üzgün, korku, sürpriz)
    - 🔬 Farklı üyelik fonksiyonları karşılaştırması (Üçgen, Yamuk, Sigmoid, Gauss, Bell)
    - 📊 Detaylı performans metrikleri (Accuracy, F1-Score, R²)
    - 🎨 Kullanıcı dostu web arayüzü
    
    **Kullanılan Teknolojiler:**
    - Python, scikit-learn, scikit-fuzzy
    - Streamlit (Web Arayüzü)
    - TF-IDF vektörizasyonu
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Bulanık Mantık Duygu Analizi Sistemi | "
    "Türkçe Tweet Analizi"
    "</div>",
    unsafe_allow_html=True
)

