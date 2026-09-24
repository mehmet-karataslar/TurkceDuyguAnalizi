"""
Egitim: (1) Bulanik mantik (yorumlanabilir)  (2) LinearSVC (yuksek performans)
Buyuk korpusta en iyi dogruluk icin SVC kullanilir; fuzzy arastirma karsilastirmasi saklanir.
"""

from __future__ import annotations

import pickle
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from data_preprocessing import preprocess_data, save_preprocessing_artifacts
from fuzzy_sentiment import FuzzySentimentClassifier

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def calculate_r_squared(y_true, y_pred_proba):
    n_classes = y_pred_proba.shape[1]
    y_true_onehot = np.zeros((len(y_true), n_classes))
    y_true_onehot[np.arange(len(y_true)), y_true] = 1
    ss_res = np.sum((y_true_onehot - y_pred_proba) ** 2)
    y_mean = np.mean(y_true_onehot, axis=0)
    ss_tot = np.sum((y_true_onehot - y_mean) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(np.mean(1 - (ss_res / ss_tot)))


def metrics_from_pred(y_true, y_pred, y_proba=None, confidences=None):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_proba is not None:
        out["r_squared"] = calculate_r_squared(y_true, y_proba)
        out["confidence_mean"] = float(np.mean(np.max(y_proba, axis=1)))
    elif confidences is not None:
        out["r_squared"] = 0.0
        out["confidence_mean"] = float(np.mean(confidences))
    else:
        out["r_squared"] = 0.0
        out["confidence_mean"] = 0.0
    return out


def balanced_take(X, y, per_class=900, seed=42):
    rng = np.random.RandomState(seed)
    indices = []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        take = min(per_class, len(idx))
        indices.extend(rng.choice(idx, size=take, replace=False).tolist())
    indices = np.array(indices)
    rng.shuffle(indices)
    return X[indices], y[indices]


def train_fuzzy(X_train, X_test, y_train, y_test, n_features=150):
    print("\n=== BULANIK MODEL ===")
    X_fit, y_fit = balanced_take(X_train, y_train, per_class=850)
    model = FuzzySentimentClassifier(
        membership_type="triangular",
        n_features=n_features,
        n_classes=len(np.unique(y_train)),
        min_membership_threshold=0.14,
        max_features_per_rule=5,
    )
    model.fit(X_fit, y_fit, feature_names=[f"feature_{i}" for i in range(X_fit.shape[1])])
    y_pred, conf = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    m = metrics_from_pred(y_test, y_pred, y_proba=y_proba)
    print(f"Fuzzy Acc={m['accuracy']:.4f} F1={m['f1_score']:.4f} rules={len(model.rules_)}")
    return model, m


def train_sklearn(X_train, X_test, y_train, y_test):
    print("\n=== LINEARSVC + CALIBRATION ===")
    base = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, dual="auto")
    clf = CalibratedClassifierCV(base, cv=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    m = metrics_from_pred(y_test, y_pred, y_proba=y_proba)
    print(f"SVC Acc={m['accuracy']:.4f} F1={m['f1_score']:.4f}")

    print("\n=== LOGISTIC REGRESSION ===")
    lr = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        C=2.0,
    )
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_proba_lr = lr.predict_proba(X_test)
    m_lr = metrics_from_pred(y_test, y_pred_lr, y_proba=y_proba_lr)
    print(f"LR  Acc={m_lr['accuracy']:.4f} F1={m_lr['f1_score']:.4f}")

    if m_lr["f1_score"] > m["f1_score"]:
        return lr, m_lr, "logistic_regression"
    return clf, m, "linear_svc_calibrated"


def main():
    print("Veri yukleniyor...")
    X_train, X_test, y_train, y_test, vectorizer, label_encoder = preprocess_data("TurkishTweets.xlsx")
    save_preprocessing_artifacts(vectorizer, label_encoder)
    print(f"Train={X_train.shape} Test={X_test.shape[0]}")

    fuzzy_model, fuzzy_metrics = train_fuzzy(X_train, X_test, y_train, y_test)
    fuzzy_model.label_encoder_ = label_encoder
    fuzzy_model.save("best_fuzzy_model.pkl")

    sk_model, sk_metrics, sk_name = train_sklearn(X_train, X_test, y_train, y_test)
    joblib.dump(
        {"model": sk_model, "name": sk_name, "label_encoder": label_encoder},
        "best_sklearn_model.pkl",
    )

    comparison = pd.DataFrame(
        [
            {"model": "fuzzy_triangular", **fuzzy_metrics},
            {"model": sk_name, **sk_metrics},
        ]
    )
    comparison.to_csv("membership_function_comparison.csv", index=False)

    # Uygulama icin en iyi model metriklerini yaz
    best = sk_metrics if sk_metrics["f1_score"] >= fuzzy_metrics["f1_score"] else fuzzy_metrics
    best_name = sk_name if sk_metrics["f1_score"] >= fuzzy_metrics["f1_score"] else "fuzzy_triangular"
    pd.DataFrame(
        [
            {
                **best,
                "mean_confidence": best["confidence_mean"],
                "membership_type": "triangular" if "fuzzy" in best_name else best_name,
                "n_features": 150 if "fuzzy" in best_name else X_train.shape[1],
                "n_classes": len(np.unique(y_train)),
                "train_size": int(X_train.shape[0]),
                "test_size": int(X_test.shape[0]),
                "active_model": best_name,
                "corpus_note": "curated_expanded_emotion",
                "fuzzy_accuracy": fuzzy_metrics["accuracy"],
                "sklearn_accuracy": sk_metrics["accuracy"],
            }
        ]
    ).to_csv("evaluation_report.csv", index=False)

    print("\n" + "=" * 60)
    print(f"AKTIF MODEL: {best_name}")
    print(f"Accuracy={best['accuracy']:.4f}  F1={best['f1_score']:.4f}")
    print("=" * 60)
    print("OK")


if __name__ == "__main__":
    main()
