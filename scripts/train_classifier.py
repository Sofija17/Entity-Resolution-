import pandas as pd
from models.pairwise_classifier import train_pairwise_matcher

df = pd.read_csv("../data/feature_extraction/er_blocking_candidates_k40_features_labeled.csv")

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
    model_name="xgb",
    n_folds=5
)

print("OOF ROC-AUC:", tm.metrics["oof_roc_auc"])
print("Best threshold (OOF):", tm.best_threshold)

probs = tm.predict_proba(df)
preds = tm.predict(df)

out = df[["src_id","cand_id"]].copy()
out["prob_match"] = probs
out["pred_match"] = preds
out.to_csv("../data/classifier_predictions_xgb_k40.csv", index=False)
