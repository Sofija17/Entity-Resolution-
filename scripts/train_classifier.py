# Example usage
import pandas as pd
from src.classifier.pairwise_classifier import train_pairwise_matcher

# Load the features CSV (must include your label column)
df = pd.read_csv("../data/er_blocking_candidates_k20_features_labeled.csv")

feature_cols = [
    "edit_ratio", "jaro_winkler", "lcs_ratio",
    "token_jaccard", "token_cosine",
    "tfidf_word_cosine", "tfidf_char_cosine",
    "dmetaphone_match",
]

tm = train_pairwise_matcher(
    df=df,
    feature_cols=feature_cols,
    label_col="label",      # 1 for MATCH, 0 for NON-MATCH
    model_name="xgb",    # or "rf", "xgb"
    n_folds=5
)

print("OOF ROC-AUC:", tm.metrics["oof_roc_auc"])
print("Best threshold (OOF):", tm.best_threshold)

# Get probabilities / predictions for the same file
probs = tm.predict_proba(df)            # shape (n_rows,)
preds = tm.predict(df)                  # 0/1 using best threshold

# If you also have IDs in df:
out = df[["src_id","cand_id"]].copy()
out["prob_match"] = probs
out["pred_match"] = preds
out.to_csv("../data/classifier_predictions_xgb.csv", index=False)
