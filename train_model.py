"""
Model Eğitimi ve Üyelik Fonksiyonu Karşılaştırması
Tüm üyelik fonksiyonlarını test edip en iyisini seçer
"""

import numpy as np
import pandas as pd
from data_preprocessing import preprocess_data, save_preprocessing_artifacts
from fuzzy_sentiment import FuzzySentimentClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os


def calculate_r_squared(y_true, y_pred_proba):
    """
    R² skorunu hesapla (sınıflandırma için)
    Olasılık skorları üzerinden hesaplanır
    """
    # Gerçek etiketleri one-hot encode et
    n_classes = y_pred_proba.shape[1]
    y_true_onehot = np.zeros((len(y_true), n_classes))
    y_true_onehot[np.arange(len(y_true)), y_true] = 1
    
    # R² = 1 - (SS_res / SS_tot)
    # SS_res: Residual sum of squares
    # SS_tot: Total sum of squares
    
    ss_res = np.sum((y_true_onehot - y_pred_proba) ** 2)
    y_mean = np.mean(y_true_onehot, axis=0)
    ss_tot = np.sum((y_true_onehot - y_mean) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    r_squared = 1 - (ss_res / ss_tot)
    return np.mean(r_squared)  # Tüm sınıflar için ortalama


def evaluate_model(model, X_test, y_test):
    """
    Modeli değerlendir ve metrikleri döndür
    """
    y_pred, confidences = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    r_squared = calculate_r_squared(y_test, y_pred_proba)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'r_squared': r_squared,
        'confidence_mean': np.mean(confidences)
    }


def compare_membership_functions(X_train, X_test, y_train, y_test, n_features=50):
    """
    Tüm üyelik fonksiyonlarını karşılaştır
    """
    membership_types = ['triangular', 'trapezoidal', 'sigmoid', 'gaussian', 'bell']
    
    results = []
    models = {}
    
    print("=" * 80)
    print("ÜYELİK FONKSİYONU KARŞILAŞTIRMASI")
    print("=" * 80)
    
    for mem_type in membership_types:
        print(f"\n{mem_type.upper()} üyelik fonksiyonu test ediliyor...")
        
        try:
            # Model oluştur ve eğit
            model = FuzzySentimentClassifier(
                membership_type=mem_type,
                n_features=n_features,
                n_classes=len(np.unique(y_train))
            )
            
            # Özellik isimlerini oluştur
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
            
            model.fit(X_train, y_train, feature_names=feature_names)
            
            # Değerlendir
            metrics = evaluate_model(model, X_test, y_test)
            
            results.append({
                'membership_type': mem_type,
                **metrics
            })
            
            models[mem_type] = model
            
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
            print(f"  R²: {metrics['r_squared']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            
        except Exception as e:
            print(f"  HATA: {str(e)}")
            continue
    
    # Sonuçları DataFrame'e çevir
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        print("\n" + "=" * 80)
        print("KARŞILAŞTIRMA SONUÇLARI")
        print("=" * 80)
        print(results_df.to_string(index=False))
        
        # En iyi modeli seç (F1-score'a göre, eşitse R²'ye göre)
        results_df['combined_score'] = results_df['f1_score'] * 0.7 + results_df['r_squared'] * 0.3
        best_idx = results_df['combined_score'].idxmax()
        best_type = results_df.loc[best_idx, 'membership_type']
        
        print(f"\n{'='*80}")
        print(f"EN İYİ ÜYELİK FONKSİYONU: {best_type.upper()}")
        print(f"{'='*80}")
        print(f"F1-Score: {results_df.loc[best_idx, 'f1_score']:.4f}")
        print(f"R²: {results_df.loc[best_idx, 'r_squared']:.4f}")
        print(f"Accuracy: {results_df.loc[best_idx, 'accuracy']:.4f}")
        
        return models[best_type], best_type, results_df
    else:
        print("\nHATA: Hiçbir model başarıyla eğitilemedi!")
        return None, None, None


def train_best_model(excel_path='TurkishTweets.xlsx', n_features=50, save_path='best_fuzzy_model.pkl'):
    """
    Veri setini yükle, tüm üyelik fonksiyonlarını test et, en iyisini kaydet
    """
    print("Veri seti yükleniyor ve ön işleniyor...")
    X_train, X_test, y_train, y_test, vectorizer, label_encoder = preprocess_data(excel_path)
    
    # Ön işleme sonuçlarını kaydet
    save_preprocessing_artifacts(vectorizer, label_encoder)
    
    print(f"\nEğitim seti: {X_train.shape[0]} örnek, {X_train.shape[1]} özellik")
    print(f"Test seti: {X_test.shape[0]} örnek")
    print(f"Sınıf sayısı: {len(np.unique(y_train))}")
    
    # Üyelik fonksiyonlarını karşılaştır
    best_model, best_type, results_df = compare_membership_functions(
        X_train, X_test, y_train, y_test, n_features=n_features
    )
    
    if best_model is not None:
        # Label encoder'ı modele ekle
        best_model.label_encoder_ = label_encoder
        
        # Modeli kaydet
        best_model.save(save_path)
        print(f"\nEn iyi model kaydedildi: {save_path}")
        
        # Sonuçları CSV olarak kaydet
        results_df.to_csv('membership_function_comparison.csv', index=False)
        print("Karşılaştırma sonuçları kaydedildi: membership_function_comparison.csv")
        
        # Son test
        print("\n" + "=" * 80)
        print("FİNAL TEST")
        print("=" * 80)
        final_metrics = evaluate_model(best_model, X_test, y_test)
        print(f"Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"F1-Score: {final_metrics['f1_score']:.4f}")
        print(f"R²: {final_metrics['r_squared']:.4f}")
        print(f"Precision: {final_metrics['precision']:.4f}")
        print(f"Recall: {final_metrics['recall']:.4f}")
        
        return best_model, vectorizer, label_encoder, results_df
    else:
        print("\nModel eğitilemedi!")
        return None, None, None, None


if __name__ == "__main__":
    # Modeli eğit
    model, vectorizer, label_encoder, results_df = train_best_model(
        excel_path='TurkishTweets.xlsx',
        n_features=50
    )
    
    if model is not None:
        print("\n✓ Model eğitimi başarıyla tamamlandı!")
    else:
        print("\n✗ Model eğitimi başarısız oldu!")

