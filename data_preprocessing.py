"""
Veri Ön İşleme Modülü
Excel'den veri okuma, Türkçe metin temizleme ve TF-IDF çıkarımı
"""


import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
import nltk


# NLTK verilerini indir (ilk çalıştırmada)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Türkçe stop words — duygu/soru taşıyan kelimeler BİLEREK tutulur
# (çok, hiç, neden, nasıl, beni, daha, en … stop listesinde YOK)
TURKISH_STOP_WORDS = [
    've', 'ile', 'bir', 'bu', 'şu', 'o', 'de', 'da', 'ki', 'mi', 'mu', 'mü',
    'için', 'gibi', 'ise', 'ama', 'ancak', 'fakat', 'lakin', 'çünkü', 'zira',
    'dolayı', 'göre', 'rağmen', 'beri', 'sonra', 'önce', 'şimdi',
    'yine', 'tekrar', 'gene', 'bile', 'dahi', 'sadece', 'yalnız', 'sanki',
    'güya', 'meğer', 'mı', 'ya', 'yani', 'hem', 'veya', 'ya da', 'ister',
    'her', 'herhangi', 'birkaç', 'biri', 'diğer', 'başka', 'bazı', 'tüm',
    'hep', 'bazıları', 'şey', 'şeyler', 'şöyle', 'böyle', 'öyle', 'aslında',
    'zaten', 'hatta', 'örneğin', 'örnek', 'özellikle', 'genelde', 'yaklaşık',
    'ayrıca', 'hemen', 'çoğu', 'arada', 'bazen', 'nadiren',
    'kendi', 'kendisi', 'kendim', 'kendin', 'kendileri',
    'burada', 'orada', 'buraya', 'buradan', 'oraya', 'oradan',
    'biz', 'siz', 'onlar', 'bize', 'size', 'onlara', 'bizim', 'sizin', 'onların',
]


def clean_text(text):
    """
    Türkçe metni temizle: noktalama, küçük harf, fazla boşluklar
    """
    if pd.isna(text) or text == '':
        return ''
    
    # String'e çevir
    text = str(text)

    # Türkçe güvenli küçük harf (İ/I karışıklığını önle)
    text = text.replace("I", "ı").replace("İ", "i").lower()
    
    # URL'leri kaldır
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Kullanıcı adlarını kaldır (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Hashtag işaretini kaldır (#hashtag -> hashtag)
    text = re.sub(r'#', '', text)
    
    # Noktalama işaretlerini kaldır (Türkçe karakterleri koru)
    # Türkçe karakterler: ı, ğ, ü, ş, ö, ç
    text = re.sub(r'[^\w\sığüşöçİĞÜŞÖÇ]', ' ', text)
    
    # Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text)
    
    # Başta ve sonda boşlukları kaldır
    text = text.strip()
    
    return text


def remove_stopwords(text):
    """
    Türkçe stop words'leri kaldır
    """
    if not text:
        return ''
    
    words = text.split()
    filtered_words = [word for word in words if word not in TURKISH_STOP_WORDS]
    return ' '.join(filtered_words)


def preprocess_data(excel_path, text_column=None, label_column=None):
    """
    Excel dosyasından veriyi oku ve ön işle
    
    Parameters:
    -----------
    excel_path : str
        Excel dosyasının yolu
    text_column : str, optional
        Tweet metni sütununun adı (None ise ilk sütun)
    label_column : str, optional
        Etiket sütununun adı (None ise son sütun)
    
    Returns:
    --------
    X_train, X_test, y_train, y_test, vectorizer, label_encoder
    """
    # Excel'i oku
    print("Excel dosyası okunuyor...")
    df = pd.read_excel(excel_path)
    
    print(f"Veri seti boyutu: {df.shape}")
    print(f"Sütunlar: {df.columns.tolist()}")
    
    # Sütun isimlerini belirle
    if text_column is None:
        text_column = df.columns[0]
    if label_column is None:
        label_column = df.columns[-1]
    
    print(f"Metin sütunu: {text_column}")
    print(f"Etiket sütunu: {label_column}")
    
    # Veriyi al
    texts = df[text_column].astype(str)
    labels = df[label_column].astype(str)
    
    # Etiket dağılımını göster
    print("\nEtiket dağılımı:")
    print(labels.value_counts())
    
    # Metinleri temizle
    print("\nMetinler temizleniyor...")
    cleaned_texts = texts.apply(clean_text)
    cleaned_texts = cleaned_texts.apply(remove_stopwords)
    
    # Boş metinleri kaldır
    mask = cleaned_texts.str.len() > 0
    cleaned_texts = cleaned_texts[mask]
    labels = labels[mask]
    
    print(f"Temizleme sonrası veri sayısı: {len(cleaned_texts)}")
    
    # Etiketleri kodla
    unique_labels = sorted(labels.unique())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    
    print(f"\nEtiket eşleştirmesi:")
    for label, idx in label_to_id.items():
        print(f"  {label} -> {idx}")
    
    encoded_labels = labels.map(label_to_id)
    
    # TF-IDF vektörizasyonu (genişletilmiş korpus için ölçeklendirildi)
    print("\nTF-IDF vektörizasyonu yapılıyor...")
    vectorizer = TfidfVectorizer(
        max_features=2500,
        min_df=2,
        max_df=0.92,
        ngram_range=(1, 3),
        sublinear_tf=True,
    )

    X = vectorizer.fit_transform(cleaned_texts)
    X = X.astype(np.float32).toarray()

    print(f"Özellik vektörü boyutu: {X.shape}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        encoded_labels.values,
        test_size=0.2,
        random_state=42,
        stratify=encoded_labels.values,
    )
    
    print(f"\nTrain seti: {X_train.shape[0]} örnek")
    print(f"Test seti: {X_test.shape[0]} örnek")
    
    # Label encoder'ı kaydet
    label_encoder = {
        'label_to_id': label_to_id,
        'id_to_label': id_to_label,
        'unique_labels': unique_labels
    }
    
    return X_train, X_test, y_train, y_test, vectorizer, label_encoder


def save_preprocessing_artifacts(vectorizer, label_encoder, filepath='preprocessing_artifacts.pkl'):
    """
    Ön işleme sonuçlarını kaydet
    """
    artifacts = {
        'vectorizer': vectorizer,
        'label_encoder': label_encoder
    }
    with open(filepath, 'wb') as f:
        pickle.dump(artifacts, f)
    print(f"Ön işleme sonuçları kaydedildi: {filepath}")


def load_preprocessing_artifacts(filepath='preprocessing_artifacts.pkl'):
    """
    Ön işleme sonuçlarını yükle
    """
    with open(filepath, 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts['vectorizer'], artifacts['label_encoder']


if __name__ == "__main__":
    # Test için
    X_train, X_test, y_train, y_test, vectorizer, label_encoder = preprocess_data(
        'TurkishTweets.xlsx'
    )
    
    # Kaydet
    save_preprocessing_artifacts(vectorizer, label_encoder)
    
    print("\nÖn işleme tamamlandı!")

