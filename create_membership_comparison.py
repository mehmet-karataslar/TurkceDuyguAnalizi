"""
Üyelik Fonksiyonu Karşılaştırması Görseli Oluşturma Scripti
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

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
    print(f"✓ Üyelik fonksiyonu karşılaştırması kaydedildi: {save_path}")
    plt.close()

if __name__ == "__main__":
    plot_membership_function_comparison()

