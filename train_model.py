"""
Model Eğitimi ve Üyelik Fonksiyonu Karşılaştırması
Genişletilmiş korpus için ölçekli eğitim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_preprocessing import preprocess_data, save_preprocessing_artifacts
from fuzzy_sentiment import FuzzySentimentClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_r_squared(y_true, y_pred_proba):
    n_classes = y_pred_proba.shape[1]
    y_true_onehot = np.zeros((len(y_true), n_classes))
    y_true_onehot[np.arange(len(y_true)), y_true] = 1

    ss_res = np.sum((y_true_onehot - y_pred_proba) ** 2)
    y_mean = np.mean(y_true_onehot, axis=0)
    ss_tot = np.sum((y_true_onehot - y_mean) ** 2)

    if ss_tot == 0:
        return 0.0

    r_squared = 1 - (ss_res / ss_tot)
    return float(np.mean(r_squared))


def evaluate_model(model, X_test, y_test):
    y_pred, confidences = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "r_squared": calculate_r_squared(y_test, y_pred_proba),
        "confidence_mean": float(np.mean(confidences)),
    }


def stratified_take(X, y, n, seed=42):
    """Büyük settten stratified alt örnek al."""
    if len(y) <= n:
        return X, y
    # train_test_split ile oran koru
    frac = n / len(y)
    X_s, _, y_s, _ = train_test_split(
        X, y, train_size=frac, random_state=seed, stratify=y
    )
    return X_s, y_s


def compare_membership_functions(
    X_train,
    X_test,
    y_train,
    y_test,
    n_features=200,
    max_train_for_fuzzy=12000,
    max_test_for_compare=3000,
):
    membership_types = ["triangular", "trapezoidal", "sigmoid", "gaussian", "bell"]
    results = []
    models = {}

    X_fit, y_fit = stratified_take(X_train, y_train, max_train_for_fuzzy)
    X_cmp, y_cmp = stratified_take(X_test, y_test, max_test_for_compare)

    print("=" * 80)
    print("UYELIK FONKSIYONU KARSILASTIRMASI")
    print(f"Fuzzy egitim ornekleri: {len(y_fit)} | Karsilastirma test: {len(y_cmp)}")
    print("=" * 80)

    for mem_type in membership_types:
        print(f"\n{mem_type.upper()} uyelik fonksiyonu test ediliyor...")
        try:
            model = FuzzySentimentClassifier(
                membership_type=mem_type,
                n_features=n_features,
                n_classes=len(np.unique(y_train)),
                min_membership_threshold=0.12,
                max_features_per_rule=5,
            )
            feature_names = [f"feature_{i}" for i in range(X_fit.shape[1])]
            model.fit(X_fit, y_fit, feature_names=feature_names)
            metrics = evaluate_model(model, X_cmp, y_cmp)
            results.append({"membership_type": mem_type, **metrics})
            models[mem_type] = model
            print(
                f"  Acc={metrics['accuracy']:.4f} F1={metrics['f1_score']:.4f} "
                f"R2={metrics['r_squared']:.4f} rules={len(model.rules_)}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"  HATA: {e}")
            continue

    results_df = pd.DataFrame(results)
    if len(results_df) == 0:
        print("\nHATA: Hicbir model egitilemedi!")
        return None, None, None

    print("\n" + "=" * 80)
    print("KARSILASTIRMA SONUCLARI")
    print("=" * 80)
    print(results_df.to_string(index=False))

    results_df["combined_score"] = results_df["f1_score"] * 0.7 + results_df["r_squared"] * 0.3
    best_idx = results_df["combined_score"].idxmax()
    best_type = results_df.loc[best_idx, "membership_type"]
    print(f"\nEN IYI UYELIK: {best_type.upper()}")
    return models[best_type], best_type, results_df


def train_best_model(
    excel_path="TurkishTweets.xlsx",
    n_features=200,
    save_path="best_fuzzy_model.pkl",
    final_eval_size=5000,
):
    print("Veri seti yukleniyor ve on isleniyor...")
    X_train, X_test, y_train, y_test, vectorizer, label_encoder = preprocess_data(excel_path)
    save_preprocessing_artifacts(vectorizer, label_encoder)

    print(f"\nEgitim: {X_train.shape[0]} ornek, {X_train.shape[1]} ozellik")
    print(f"Test: {X_test.shape[0]} ornek | Sinif: {len(np.unique(y_train))}")

    best_model, best_type, results_df = compare_membership_functions(
        X_train, X_test, y_train, y_test, n_features=n_features
    )

    if best_model is None:
        print("\nModel egitilemedi!")
        return None, None, None, None

    # En iyi tipi daha genis ornekle yeniden egit
    print(f"\nEn iyi tip ({best_type}) genis ornekle yeniden egitiliyor...")
    X_refit, y_refit = stratified_take(X_train, y_train, 15000)
    best_model = FuzzySentimentClassifier(
        membership_type=best_type,
        n_features=n_features,
        n_classes=len(np.unique(y_train)),
        min_membership_threshold=0.12,
        max_features_per_rule=5,
    )
    best_model.fit(
        X_refit,
        y_refit,
        feature_names=[f"feature_{i}" for i in range(X_refit.shape[1])],
    )
    best_model.label_encoder_ = label_encoder
    best_model.save(save_path)
    print(f"Model kaydedildi: {save_path} | kurallar: {len(best_model.rules_)}")

    results_df.to_csv("membership_function_comparison.csv", index=False)

    X_final, y_final = stratified_take(X_test, y_test, final_eval_size)
    print("\n" + "=" * 80)
    print(f"FINAL TEST ({len(y_final)} ornek)")
    print("=" * 80)
    final_metrics = evaluate_model(best_model, X_final, y_final)
    for k, v in final_metrics.items():
        print(f"{k}: {v:.4f}")

    # evaluation_report.csv
    report = {
        **final_metrics,
        "mean_confidence": final_metrics["confidence_mean"],
        "membership_type": best_type,
        "n_features": n_features,
        "n_classes": len(np.unique(y_train)),
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "corpus_note": "expanded_multi_source",
    }
    pd.DataFrame([report]).to_csv("evaluation_report.csv", index=False)
    print("evaluation_report.csv guncellendi")

    return best_model, vectorizer, label_encoder, results_df


if __name__ == "__main__":
    model, vectorizer, label_encoder, results_df = train_best_model(
        excel_path="TurkishTweets.xlsx",
        n_features=200,
    )
    if model is not None:
        print("\nModel egitimi basariyla tamamlandi!")
    else:
        print("\nModel egitimi basarisiz!")
