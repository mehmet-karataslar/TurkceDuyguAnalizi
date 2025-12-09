"""
Bulanık Mantık Duygu Analizi Modeli
Farklı üyelik fonksiyonları ile bulanık çıkarım sistemi
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import pickle


class MembershipFunction:
    """
    Farklı üyelik fonksiyonları için wrapper sınıfı
    """
    
    @staticmethod
    def triangular(x, a, b, c):
        """
        Üçgen üyelik fonksiyonu
        a: sol köşe, b: tepe noktası, c: sağ köşe
        """
        return np.maximum(0, np.minimum((x - a) / (b - a) if b != a else 0,
                                        (c - x) / (c - b) if c != b else 0))
    
    @staticmethod
    def trapezoidal(x, a, b, c, d):
        """
        Yamuk üyelik fonksiyonu
        a: sol alt, b: sol üst, c: sağ üst, d: sağ alt
        """
        result = np.zeros_like(x)
        result[(x >= a) & (x < b)] = (x[(x >= a) & (x < b)] - a) / (b - a) if b != a else 1
        result[(x >= b) & (x <= c)] = 1
        result[(x > c) & (x <= d)] = (d - x[(x > c) & (x <= d)]) / (d - c) if d != c else 1
        return result
    
    @staticmethod
    def sigmoid(x, a, c):
        """
        Sigmoid üyelik fonksiyonu
        a: eğim, c: merkez noktası
        """
        return 1 / (1 + np.exp(-a * (x - c)))
    
    @staticmethod
    def gaussian(x, mean, sigma):
        """
        Gauss üyelik fonksiyonu
        mean: ortalama, sigma: standart sapma
        """
        return np.exp(-0.5 * ((x - mean) / sigma) ** 2)
    
    @staticmethod
    def bell(x, a, b, c):
        """
        Bell (çan) üyelik fonksiyonu
        a: genişlik, b: eğim, c: merkez
        """
        return 1 / (1 + np.abs((x - c) / a) ** (2 * b))


class FuzzySentimentClassifier(BaseEstimator, ClassifierMixin):
    """
    Bulanık mantık tabanlı duygu analizi sınıflandırıcısı
    """
    
    def __init__(self, membership_type='trapezoidal', n_features=10, n_classes=4, 
                 min_membership_threshold=0.1, max_features_per_rule=5):
        """
        Parameters:
        -----------
        membership_type : str
            Üyelik fonksiyonu tipi: 'triangular', 'trapezoidal', 'sigmoid', 'gaussian', 'bell'
        n_features : int
            Kullanılacak özellik sayısı (en önemli n özellik)
        n_classes : int
            Sınıf sayısı
        min_membership_threshold : float
            Minimum üyelik derecesi eşiği (düşük üyelik dereceli kurallar filtrelenir)
        max_features_per_rule : int
            Her kuralda kullanılacak maksimum özellik sayısı
        """
        self.membership_type = membership_type
        self.n_features = n_features
        self.n_classes = n_classes
        self.min_membership_threshold = min_membership_threshold
        self.max_features_per_rule = max_features_per_rule
        self.feature_importance_ = None
        self.membership_params_ = None
        self.rules_ = None
        self.label_encoder_ = None
        
    def _create_membership_functions(self, feature_values, n_sets=3):
        """
        Her özellik için bulanık kümeler oluştur
        n_sets: Her özellik için kaç bulanık küme (düşük, orta, yüksek)
        """
        membership_funcs = {}
        params = {}
        
        for i, (feature_name, values) in enumerate(feature_values.items()):
            min_val = np.min(values)
            max_val = np.max(values)
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if self.membership_type == 'triangular':
                # Üçgen: düşük, orta, yüksek
                if n_sets == 3:
                    low = (min_val, min_val, mean_val)
                    medium = (min_val, mean_val, max_val)
                    high = (mean_val, max_val, max_val)
                else:
                    # 5 küme: çok düşük, düşük, orta, yüksek, çok yüksek
                    step = (max_val - min_val) / 4
                    very_low = (min_val, min_val, min_val + step)
                    low = (min_val, min_val + step, min_val + 2*step)
                    medium = (min_val + step, min_val + 2*step, min_val + 3*step)
                    high = (min_val + 2*step, min_val + 3*step, max_val)
                    very_high = (min_val + 3*step, max_val, max_val)
                    
                params[feature_name] = {
                    'very_low': very_low, 'low': low, 'medium': medium,
                    'high': high, 'very_high': very_high
                } if n_sets == 5 else {'low': low, 'medium': medium, 'high': high}
                
            elif self.membership_type == 'trapezoidal':
                # Yamuk
                if n_sets == 3:
                    step = (max_val - min_val) / 3
                    low = (min_val, min_val, min_val + step, min_val + 2*step)
                    medium = (min_val + step, min_val + 2*step, min_val + 2*step, max_val)
                    high = (min_val + 2*step, max_val, max_val, max_val)
                else:
                    step = (max_val - min_val) / 5
                    very_low = (min_val, min_val, min_val + step, min_val + 2*step)
                    low = (min_val + step, min_val + 2*step, min_val + 3*step, min_val + 4*step)
                    medium = (min_val + 2*step, min_val + 3*step, min_val + 3*step, min_val + 4*step)
                    high = (min_val + 3*step, min_val + 4*step, min_val + 4*step, max_val)
                    very_high = (min_val + 4*step, max_val, max_val, max_val)
                    
                params[feature_name] = {
                    'very_low': very_low, 'low': low, 'medium': medium,
                    'high': high, 'very_high': very_high
                } if n_sets == 5 else {'low': low, 'medium': medium, 'high': high}
                
            elif self.membership_type == 'sigmoid':
                # Sigmoid: düşük (azalan), yüksek (artan)
                if n_sets == 3:
                    low = (10/(max_val-min_val+1), min_val + (max_val-min_val)/3)
                    medium = (10/(max_val-min_val+1), mean_val)
                    high = (10/(max_val-min_val+1), max_val - (max_val-min_val)/3)
                else:
                    step = (max_val - min_val) / 5
                    very_low = (10/(max_val-min_val+1), min_val + step)
                    low = (10/(max_val-min_val+1), min_val + 2*step)
                    medium = (10/(max_val-min_val+1), mean_val)
                    high = (10/(max_val-min_val+1), max_val - 2*step)
                    very_high = (10/(max_val-min_val+1), max_val - step)
                    
                params[feature_name] = {
                    'very_low': very_low, 'low': low, 'medium': medium,
                    'high': high, 'very_high': very_high
                } if n_sets == 5 else {'low': low, 'medium': medium, 'high': high}
                
            elif self.membership_type == 'gaussian':
                # Gauss
                if n_sets == 3:
                    low = (min_val + (max_val-min_val)/3, std_val/2)
                    medium = (mean_val, std_val)
                    high = (max_val - (max_val-min_val)/3, std_val/2)
                else:
                    step = (max_val - min_val) / 5
                    very_low = (min_val + step, std_val/3)
                    low = (min_val + 2*step, std_val/2)
                    medium = (mean_val, std_val)
                    high = (max_val - 2*step, std_val/2)
                    very_high = (max_val - step, std_val/3)
                    
                if n_sets == 3:
                    params[feature_name] = {'low': low, 'medium': medium, 'high': high}
                else:
                    params[feature_name] = {
                        'very_low': very_low, 'low': low, 'medium': medium,
                        'high': high, 'very_high': very_high
                    }
                
            elif self.membership_type == 'bell':
                # Bell
                if n_sets == 3:
                    a = (max_val - min_val) / 6
                    low = (a, 2, min_val + (max_val-min_val)/3)
                    medium = (a, 2, mean_val)
                    high = (a, 2, max_val - (max_val-min_val)/3)
                else:
                    a = (max_val - min_val) / 10
                    step = (max_val - min_val) / 5
                    very_low = (a, 2, min_val + step)
                    low = (a, 2, min_val + 2*step)
                    medium = (a, 2, mean_val)
                    high = (a, 2, max_val - 2*step)
                    very_high = (a, 2, max_val - step)
                    
                params[feature_name] = {
                    'very_low': very_low, 'low': low, 'medium': medium,
                    'high': high, 'very_high': very_high
                } if n_sets == 5 else {'low': low, 'medium': medium, 'high': high}
        
        self.membership_params_ = params
        return params
    
    def _compute_membership(self, value, params, set_name):
        """
        Belirli bir değer için üyelik derecesini hesapla
        """
        if self.membership_type == 'triangular':
            a, b, c = params[set_name]
            return MembershipFunction.triangular(value, a, b, c)
        elif self.membership_type == 'trapezoidal':
            a, b, c, d = params[set_name]
            return MembershipFunction.trapezoidal(value, a, b, c, d)
        elif self.membership_type == 'sigmoid':
            a, c = params[set_name]
            return MembershipFunction.sigmoid(value, a, c)
        elif self.membership_type == 'gaussian':
            mean, sigma = params[set_name]
            return MembershipFunction.gaussian(value, mean, sigma)
        elif self.membership_type == 'bell':
            a, b, c = params[set_name]
            return MembershipFunction.bell(value, a, b, c)
        else:
            raise ValueError(f"Bilinmeyen üyelik fonksiyonu: {self.membership_type}")
    
    def _extract_rules_from_data(self, X, y, feature_names):
        """
        Eğitim verisinden bulanık kurallar çıkar
        Basit yaklaşım: Her örnek için en yüksek üyelik derecelerine sahip özellikleri kullan
        """
        rules = []
        
        for i in range(len(X)):
            sample = X[i]
            label = y[i]
            
            # Her özellik için üyelik derecelerini hesapla
            feature_memberships = {}
            for j, feature_name in enumerate(feature_names):
                value = sample[j]
                params = self.membership_params_[feature_name]
                
                # Her küme için üyelik derecesi
                memberships = {}
                for set_name in params.keys():
                    memberships[set_name] = self._compute_membership(value, params, set_name)
                
                # En yüksek üyelik derecesine sahip küme
                max_set = max(memberships, key=memberships.get)
                feature_memberships[feature_name] = (max_set, memberships[max_set])
            
            # En yüksek üyelik derecelerine sahip özellikleri seç
            sorted_features = sorted(feature_memberships.items(), 
                                   key=lambda x: x[1][1], reverse=True)
            # Minimum eşiği geçen özellikleri filtrele
            filtered_features = [(feat, mem) for feat, mem in sorted_features 
                               if mem[1] >= self.min_membership_threshold]
            
            if len(filtered_features) == 0:
                # Eşik geçen özellik yoksa en yüksek olanı al
                filtered_features = sorted_features[:1]
            
            top_features = filtered_features[:min(self.max_features_per_rule, len(filtered_features))]
            
            # Ortalama üyelik derecesi
            avg_membership = np.mean([mem[1] for _, mem in top_features])
            
            # Minimum eşiği geçen kuralları ekle
            if avg_membership >= self.min_membership_threshold:
                rule = {
                    'conditions': [(feat, mem[0]) for feat, mem in top_features],
                    'conclusion': label,
                    'weight': avg_membership,
                    'support': 1  # Bu kuralın kaç örnekte görüldüğü
                }
                rules.append(rule)
        
        # Benzer kuralları birleştir ve ağırlıklandır
        rule_dict = {}
        for rule in rules:
            key = tuple(sorted(rule['conditions']))
            if key not in rule_dict:
                rule_dict[key] = {'conclusions': [], 'weights': [], 'supports': []}
            rule_dict[key]['conclusions'].append(rule['conclusion'])
            rule_dict[key]['weights'].append(rule['weight'])
            rule_dict[key]['supports'].append(rule.get('support', 1))
        
        # Her kural için en sık görülen sonucu seç
        final_rules = []
        for key, data in rule_dict.items():
            conclusions = np.array(data['conclusions'])
            weights = np.array(data['weights'])
            supports = np.array(data['supports'])
            
            # Ağırlıklı mod (support ile çarpılmış)
            unique_labels = np.unique(conclusions)
            weighted_counts = {}
            total_weight = 0
            for label in unique_labels:
                mask = conclusions == label
                # Ağırlık * support (daha fazla örnekte görülen kurallar daha önemli)
                weighted_counts[label] = np.sum(weights[mask] * supports[mask])
                total_weight += weighted_counts[label]
            
            if total_weight > 0:
                final_label = max(weighted_counts, key=weighted_counts.get)
                confidence = weighted_counts[final_label] / total_weight
                
                # Minimum güven eşiğini geçen kuralları ekle
                if confidence >= 0.3:  # En az %30 güven
                    final_rules.append({
                        'conditions': list(key),
                        'conclusion': final_label,
                        'confidence': confidence,
                        'support': int(np.sum(supports[conclusions == final_label]))
                    })
        
        self.rules_ = final_rules
        return final_rules
    
    def fit(self, X, y, feature_names=None):
        """
        Modeli eğit
        """
        # Özellik isimlerini oluştur
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # En önemli özellikleri seç (varyans + sınıf ayrımına göre)
        from sklearn.feature_selection import f_classif
        
        # Varyans skorları
        feature_vars = np.var(X, axis=0)
        
        # F-score (sınıf ayrımı)
        try:
            f_scores, _ = f_classif(X, y)
            f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
        except:
            f_scores = np.ones(X.shape[1])
        
        # Kombine skor: varyans * F-score
        combined_scores = feature_vars * (1 + f_scores)
        top_indices = np.argsort(combined_scores)[-self.n_features:][::-1]
        
        self.feature_importance_ = top_indices
        X_selected = X[:, top_indices]
        feature_names_selected = [feature_names[i] for i in top_indices]
        
        # Her özellik için değer aralıklarını hesapla
        feature_values = {}
        for i, name in enumerate(feature_names_selected):
            feature_values[name] = X_selected[:, i]
        
        # Üyelik fonksiyonlarını oluştur
        self._create_membership_functions(feature_values, n_sets=3)
        
        # Kuralları çıkar
        self._extract_rules_from_data(X_selected, y, feature_names_selected)
        
        print(f"Model eğitildi: {len(self.rules_)} kural oluşturuldu")
        return self
    
    def predict(self, X):
        """
        Tahmin yap
        """
        # Özellikleri seç
        X_selected = X[:, self.feature_importance_]
        
        predictions = []
        confidences = []
        
        for sample in X_selected:
            # Her kural için uygunluk derecesini hesapla
            rule_scores = []
            
            for rule in self.rules_:
                # Kural koşullarını kontrol et
                min_membership = 1.0
                for feature_name, set_name in rule['conditions']:
                    # Özellik indeksini bul
                    feature_idx = None
                    for i, name in enumerate(self.membership_params_.keys()):
                        if name == feature_name:
                            feature_idx = i
                            break
                    
                    if feature_idx is not None:
                        value = sample[feature_idx]
                        params = self.membership_params_[feature_name]
                        membership = self._compute_membership(value, params, set_name)
                        min_membership = min(min_membership, membership)
                
                # Kural güveni ve support ile çarp (daha fazla örnekte görülen kurallar daha önemli)
                support_factor = 1 + 0.1 * rule.get('support', 1)  # Support'a göre bonus
                rule_score = min_membership * rule['confidence'] * support_factor
                rule_scores.append((rule['conclusion'], rule_score))
            
            # En yüksek skorlu kuralın sonucunu seç
            if rule_scores:
                # Aynı sınıf için skorları topla
                class_scores = {}
                for label, score in rule_scores:
                    if label not in class_scores:
                        class_scores[label] = 0
                    class_scores[label] += score
                
                # En yüksek skorlu sınıf
                predicted_class = max(class_scores, key=class_scores.get)
                confidence = class_scores[predicted_class] / sum(class_scores.values()) if sum(class_scores.values()) > 0 else 0
            else:
                # Kural yoksa rastgele tahmin
                predicted_class = 0
                confidence = 0.0
            
            predictions.append(predicted_class)
            confidences.append(confidence)
        
        return np.array(predictions), np.array(confidences)
    
    def predict_proba(self, X):
        """
        Sınıf olasılıklarını döndür
        """
        X_selected = X[:, self.feature_importance_]
        
        probabilities = []
        
        for sample in X_selected:
            class_scores = {i: 0.0 for i in range(self.n_classes)}
            
            for rule in self.rules_:
                min_membership = 1.0
                for feature_name, set_name in rule['conditions']:
                    feature_idx = None
                    for i, name in enumerate(self.membership_params_.keys()):
                        if name == feature_name:
                            feature_idx = i
                            break
                    
                    if feature_idx is not None:
                        value = sample[feature_idx]
                        params = self.membership_params_[feature_name]
                        membership = self._compute_membership(value, params, set_name)
                        min_membership = min(min_membership, membership)
                
                # Kural güveni ve support ile çarp
                support_factor = 1 + 0.1 * rule.get('support', 1)
                rule_score = min_membership * rule['confidence'] * support_factor
                class_scores[rule['conclusion']] += rule_score
            
            # Normalize et
            total = sum(class_scores.values())
            if total > 0:
                probs = [class_scores[i] / total for i in range(self.n_classes)]
            else:
                probs = [1.0 / self.n_classes] * self.n_classes
            
            probabilities.append(probs)
        
        return np.array(probabilities)
    
    def save(self, filepath):
        """
        Modeli kaydet
        """
        model_data = {
            'membership_type': self.membership_type,
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'feature_importance_': self.feature_importance_,
            'membership_params_': self.membership_params_,
            'rules_': self.rules_,
            'label_encoder_': self.label_encoder_
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath):
        """
        Modeli yükle
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(
            membership_type=model_data['membership_type'],
            n_features=model_data['n_features'],
            n_classes=model_data['n_classes']
        )
        model.feature_importance_ = model_data['feature_importance_']
        model.membership_params_ = model_data['membership_params_']
        model.rules_ = model_data['rules_']
        model.label_encoder_ = model_data.get('label_encoder_')
        
        return model

