import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, roc_auc_score,
    confusion_matrix
)

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluates a trained classifier and prints all key metrics.
    Returns a dict of scores for comparison later.
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    f1        = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    auc_pr    = average_precision_score(y_test, y_proba)
    auc_roc   = roc_auc_score(y_test, y_proba)

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  AUC-PR (main metric):  {auc_pr:.4f}")
    print(f"  AUC-ROC:               {auc_roc:.4f}")
    print(f"  F1-Score:              {f1:.4f}")
    print(f"  Precision:             {precision:.4f}")
    print(f"  Recall:                {recall:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm,
        index=['Actual Legit', 'Actual Fraud'],
        columns=['Predicted Legit', 'Predicted Fraud'])
    print(f"\nConfusion Matrix:")
    print(cm_df)

    tn, fp, fn, tp = cm.ravel()
    print(f"\n  True Positives  (fraud caught):        {tp}")
    print(f"  False Negatives (fraud missed):        {fn}  ← costly!")
    print(f"  False Positives (legit flagged wrong): {fp}  ← frustrating")
    print(f"  True Negatives  (legit correct):       {tn}")

    return {
        'model': model_name,
        'AUC-PR': round(auc_pr, 4),
        'AUC-ROC': round(auc_roc, 4),
        'F1': round(f1, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'Fraud Caught (TP)': tp,
        'Fraud Missed (FN)': fn
    }