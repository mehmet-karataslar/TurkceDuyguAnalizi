"""
Model Değerlendirme Modülü
Detaylı metrikler, confusion matrix ve görselleştirmeler
"""


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from fuzzy_sentiment import FuzzySentimentClassifier
from data_preprocessing import load_preprocessing_artifacts
import os



def calculate_r_squared(y_true, y_pred_proba):
    """
    R² skorunu hesapla
    """
    n_classes = y_pred_proba.shape[1]
    y_true_onehot = np.zeros((len(y_true), n_classes))
    y_true_onehot[np.arange(len(y_true)), y_true] = 1
    
    ss_res = np.sum((y_true_onehot - y_pred_proba) ** 2)
    y_mean = np.mean(y_true_onehot, axis=0)
    ss_tot = np.sum((y_true_onehot - y_mean) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    r_squared = 1 - (ss_res / ss_tot)
    return np.mean(r_squared)



def evaluate_model_detailed(model, X_test, y_test, label_encoder):
    """
    Modeli detaylı olarak değerlendir
    """
    print("=" * 80)
    print("MODEL DEĞERLENDİRME RAPORU")
    print("=" * 80)
    
    # Tahminler
    y_pred, confidences = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    
    # Temel metrikler
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    r_squared = calculate_r_squared(y_test, y_pred_proba)
    
    print(f"\nGENEL METRİKLER:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  R²:        {r_squared:.4f}")
    print(f"  Ortalama Güven: {np.mean(confidences):.4f}")
    
    # Sınıf bazlı metrikler
    print(f"\nSINIF BAZLI METRİKLER:")
    id_to_label = label_encoder['id_to_label']
    class_names = [id_to_label[i] for i in sorted(id_to_label.keys())]
    
    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    print(f"\n{'Sınıf':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 50)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} {precision_per_class[i]:<12.4f} {recall_per_class[i]:<12.4f} {f1_per_class[i]:<12.4f}")
    
    # Classification report
    print(f"\nDETAYLI SINIFLANDIRMA RAPORU:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'r_squared': r_squared,
        'confusion_matrix': cm,
        'y_true': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confidences': confidences,
        'class_names': class_names
    }


def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """
    Confusion matrix görselleştir
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Örnek Sayısı'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('Gerçek Etiket', fontsize=12)
    plt.xlabel('Tahmin Edilen Etiket', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix kaydedildi: {save_path}")
    plt.close()


def plot_class_distribution(y_true, y_pred, class_names, save_path='class_distribution.png'):
    """
    Sınıf dağılımını görselleştir
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gerçek dağılım
    true_counts = pd.Series(y_true).value_counts().sort_index()
    ax1.bar([class_names[i] for i in true_counts.index], true_counts.values, color='steelblue')
    ax1.set_title('Gerçek Sınıf Dağılımı', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Örnek Sayısı')
    ax1.tick_params(axis='x', rotation=45)
    
    # Tahmin dağılımı
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    ax2.bar([class_names[i] for i in pred_counts.index], pred_counts.values, color='coral')
    ax2.set_title('Tahmin Edilen Sınıf Dağılımı', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Örnek Sayısı')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Sınıf dağılımı grafiği kaydedildi: {save_path}")
    plt.close()


def plot_confidence_distribution(confidences, save_path='confidence_distribution.png'):
    """
    Güven skorları dağılımını görselleştir
    """
    plt.figure(figsize=(10, 6))
    plt.hist(confidences, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    plt.axvline(np.mean(confidences), color='red', linestyle='--', 
                linewidth=2, label=f'Ortalama: {np.mean(confidences):.3f}')
    plt.xlabel('Güven Skoru', fontsize=12)
    plt.ylabel('Frekans', fontsize=12)
    plt.title('Güven Skorları Dağılımı', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Güven dağılımı grafiği kaydedildi: {save_path}")
    plt.close()


def plot_metrics_comparison(metrics_dict, save_path='metrics_comparison.png'):
    """
    Metrikleri karşılaştırmalı olarak göster
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'r_squared']
    values = [metrics_dict[m] for m in metrics]
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'R²']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6'])
    plt.ylim(0, 1)
    plt.ylabel('Skor', fontsize=12)
    plt.title('Model Performans Metrikleri', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Değerleri çubukların üzerine yaz
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Metrik karşılaştırması kaydedildi: {save_path}")
    plt.close()


def plot_membership_function_comparison(csv_path='membership_function_comparison.csv', 
                                       save_path='Gorseller/membership_function_comparison.png'):
    """
    Üyelik fonksiyonlarının performansını karşılaştırmalı olarak göster
    """
    if not os.path.exists(csv_path):
        print(f"HATA: CSV dosyası bulunamadı: {csv_path}")
        return None
    
    # CSV'yi oku
    df = pd.read_csv(csv_path)
    
    # Türkçe isimler
    membership_names = {
        'triangular': 'Üçgen',
        'trapezoidal': 'Yamuk',
        'sigmoid': 'Sigmoid',
        'gaussian': 'Gauss',
        'bell': 'Bell'
    }
    
    # Metrikler
    metrics = ['accuracy', 'f1_score', 'r_squared', 'precision', 'recall']
    metric_labels = ['Accuracy', 'F1-Score', 'R²', 'Precision', 'Recall']
    
    # Figür oluştur - daha geniş yaparak legend için yer aç
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # X pozisyonları
    x = np.arange(len(df))
    width = 0.15  # Çubuk genişliği
    
    # Renkler
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    
    # Her metrik için çubuklar
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = df[metric].values
        # Negatif değerleri 0'a çek (görselleştirme için)
        values = np.maximum(values, 0)
        bars = ax.bar(x + i * width, values, width, label=label, color=colors[i], alpha=0.8)
        
        # Değerleri çubukların üzerine yaz
        for j, (bar, value) in enumerate(zip(bars, df[metric].values)):
            height = bar.get_height()
            if height > 0.01:  # Sadece görünür çubuklar için
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}' if value >= 0 else f'{value:.2f}',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # X ekseni etiketleri
    membership_labels = [membership_names.get(mem_type, mem_type.title()) for mem_type in df['membership_type']]
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(membership_labels, fontsize=12, fontweight='bold')
    
    # Y ekseni
    ax.set_ylabel('Skor', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.5, 1.0)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Başlık
    ax.set_title('Üyelik Fonksiyonları Performans Karşılaştırması', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Legend - grafiğin dışına, sağ tarafa yerleştir
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, 
             framealpha=0.9, ncol=1)
    
    # En iyi performansı vurgula
    best_idx = df['f1_score'].idxmax()
    best_type = df.loc[best_idx, 'membership_type']
    best_label = membership_names.get(best_type, best_type.title())
    ax.text(0.02, 0.98, f'⭐ En İyi: {best_label} (F1: {df.loc[best_idx, "f1_score"]:.3f})',
           transform=ax.transAxes, fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
           verticalalignment='top')
    
    # Tight layout ile legend'ın taşmamasını sağla
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    
    # Klasör yoksa oluştur
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Üyelik fonksiyonu karşılaştırması kaydedildi: {save_path}")
    plt.close()


def generate_evaluation_report(model_path='best_fuzzy_model.pkl', 
                              preprocessing_path='preprocessing_artifacts.pkl',
                              X_test=None, y_test=None):
    """
    Tam değerlendirme raporu oluştur
    """
    # Modeli yükle
    if not os.path.exists(model_path):
        print(f"HATA: Model dosyası bulunamadı: {model_path}")
        print("Önce train_model.py'yi çalıştırın!")
        return None
    
    print("Model yükleniyor...")
    model = FuzzySentimentClassifier.load(model_path)
    
    # Ön işleme sonuçlarını yükle
    if not os.path.exists(preprocessing_path):
        print(f"HATA: Ön işleme dosyası bulunamadı: {preprocessing_path}")
        return None
    
    vectorizer, label_encoder = load_preprocessing_artifacts(preprocessing_path)
    
    # Eğer test verisi verilmemişse, veri setini tekrar yükle
    if X_test is None or y_test is None:
        from data_preprocessing import preprocess_data
        print("Test verisi yükleniyor...")
        _, X_test, _, y_test, _, _ = preprocess_data('TurkishTweets.xlsx')
    
    # Değerlendir
    results = evaluate_model_detailed(model, X_test, y_test, label_encoder)
    
    # Görselleştirmeler
    print("\nGörselleştirmeler oluşturuluyor...")
    plot_confusion_matrix(results['confusion_matrix'], results['class_names'])
    plot_class_distribution(results['y_true'], results['y_pred'], results['class_names'])
    plot_confidence_distribution(results['confidences'])
    plot_metrics_comparison({
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1_score': results['f1_score'],
        'r_squared': results['r_squared']
    })
    
    # Raporu kaydet
    report = {
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1_score': results['f1_score'],
        'r_squared': results['r_squared'],
        'mean_confidence': np.mean(results['confidences']),
        'membership_type': model.membership_type,
        'n_features': model.n_features,
        'n_classes': model.n_classes
    }
    
    report_df = pd.DataFrame([report])
    report_df.to_csv('evaluation_report.csv', index=False)
    print("\nDeğerlendirme raporu kaydedildi: evaluation_report.csv")
    
    return results


if __name__ == "__main__":
    # Değerlendirme raporu oluştur
    results = generate_evaluation_report()
    
    # Üyelik fonksiyonu karşılaştırması görselini oluştur
    print("\n" + "=" * 80)
    print("ÜYELİK FONKSİYONU KARŞILAŞTIRMASI GÖRSELİ OLUŞTURULUYOR...")
    print("=" * 80)
    plot_membership_function_comparison()
    
    if results:
        print("\n✓ Değerlendirme tamamlandı!")
    else:
        print("\n✗ Değerlendirme başarısız oldu!")

