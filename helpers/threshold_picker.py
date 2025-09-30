from sklearn.metrics import fbeta_score

y_true = df_labeled["true_match"].values
y_prob = df_labeled["prob_match"].values

# Precision-recall curve
prec, rec, thresh_pr = precision_recall_curve(y_true, y_prob)

# ROC curve
fpr, tpr, thresh_roc = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

# Compute metrics at multiple thresholds
thresholds = np.linspace(0.1, 0.95, 18)  # from 0.1 to 0.95 step 0.05
results = []
for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    results.append((t, p, r, f1, f05, f2))

import pandas as pd
metrics_df = pd.DataFrame(results, columns=["threshold","precision","recall","F1","F0.5","F2"])

import matplotlib.pyplot as plt

# Plot Precision-Recall curve
plt.figure(figsize=(6,5))
plt.plot(rec, prec, label="PR curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

# Plot ROC curve
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC={roc_auc:.3f})")
plt.plot([0,1],[0,1],"--", color="grey")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

import caas_jupyter_tools
caas_jupyter_tools.display_dataframe_to_user("Threshold Metrics", metrics_df)
