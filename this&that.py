import warnings
import numpy as np
import pandas as pd
import random
import os

from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
from lightgbm import LGBMClassifier
import xgboost as xgb

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
SEED = 42
TRAIN_FILE_PATH = "DataSetbearing-failure.csv"
TEST_FILE_PATH = "Star_test.csv"
SUBMISSION_PATH = "submission16.csv"

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(SEED)

def load_dataset(path):
    try:
        return pd.read_csv(path, encoding='utf-16', sep='\t')
    except Exception:
        return pd.read_csv(path)

print("=" * 70)
print("Bearing Fault Detection - NO SPLIT (Hybrid KNN + Regularization)")
print("=" * 70)

# ============================================================
# 1. DATA LOADING (NO SPLIT!)
# ============================================================
train_df = load_dataset(TRAIN_FILE_PATH).dropna(subset=['Label']).reset_index(drop=True)
val_df = load_dataset(TEST_FILE_PATH)

print(f"✓ Train: {train_df.shape} | Test: {val_df.shape}")

if len(val_df) != 600:
    print(f"⚠️ WARNING: Test has {len(val_df)} rows (expected 600)")

oc = [
    'Vel, Rms (RMS)', 'Acc, Rms (RMS)', 'Crest (RMS)',
    'Kurt (RMS)', 'Vel, Peak (RMS)', 'Vel, Peak to peak (RMS)'
]

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
eps = 1e-9

def extract_features(df):
    df_out = df.copy()

    # Ratios
    df_out["peak_to_rms"] = df_out["Vel, Peak (RMS)"] / (df_out["Vel, Rms (RMS)"] + eps)
    df_out["pp_to_rms"] = df_out["Vel, Peak to peak (RMS)"] / (df_out["Vel, Rms (RMS)"] + eps)
    df_out["acc_vel_ratio"] = df_out["Acc, Rms (RMS)"] / (df_out["Vel, Rms (RMS)"] + eps)

    df_out["kurtosis_index"] = df_out["Kurt (RMS)"] / 3.0
    df_out["severity_index"] = df_out["Vel, Rms (RMS)"] * df_out["Acc, Rms (RMS)"] * df_out["Crest (RMS)"]
    df_out["early_fault_index"] = (df_out["Kurt (RMS)"] * df_out["Crest (RMS)"]) / (df_out["Acc, Rms (RMS)"] + eps)

    # Log transforms
    for col in oc:
        df_out[f"{col}_log"] = np.log1p(np.maximum(0, df_out[col]))
        df_out[f"{col}_sq"] = df_out[col] ** 2

    df_out["kurt_crest"] = df_out["Kurt (RMS)"] * df_out["Crest (RMS)"]
    df_out["vel_acc"] = df_out["Vel, Rms (RMS)"] * df_out["Acc, Rms (RMS)"]

    return df_out

X_train_feat = extract_features(train_df)
X_val_feat = extract_features(val_df)

# Encode categorical
le_comp = LabelEncoder().fit(pd.concat([train_df['COMP_NAME'], val_df['COMP_NAME']]).astype(str))
le_loc = LabelEncoder().fit(pd.concat([train_df['MP_LOC'], val_df['MP_LOC']]).astype(str))

X_train_feat["COMP_NAME_encoded"] = le_comp.transform(train_df["COMP_NAME"].astype(str))
X_val_feat["COMP_NAME_encoded"]   = le_comp.transform(val_df["COMP_NAME"].astype(str))
X_train_feat["MP_LOC_encoded"] = le_loc.transform(train_df["MP_LOC"].astype(str))
X_val_feat["MP_LOC_encoded"]   = le_loc.transform(val_df["MP_LOC"].astype(str))

# Group Z-Score
for col in oc:
    grp_means = train_df.groupby(["COMP_NAME", "MP_LOC"])[col].mean()
    grp_stds = train_df.groupby(["COMP_NAME", "MP_LOC"])[col].std() + eps
    overall_mean = train_df[col].mean()
    overall_std = train_df[col].std() + eps

    def apply_zscore(source_df, target_df):
        keys = list(zip(source_df['COMP_NAME'], source_df['MP_LOC']))
        m = np.array([grp_means.get(k, overall_mean) for k in keys])
        s = np.array([grp_stds.get(k, overall_std) for k in keys])
        target_df[f"{col}_grp_zscore"] = (source_df[col] - m) / (s + eps)

    apply_zscore(train_df, X_train_feat)
    apply_zscore(val_df, X_val_feat)

drop_cols = ["Label", "COMP_NAME", "MP_LOC"]
y_train = train_df["Label"].values.astype(int)

X_train = X_train_feat.drop(columns=[c for c in drop_cols if c in X_train_feat.columns])
X_val   = X_val_feat.drop(columns=[c for c in drop_cols if c in X_val_feat.columns])

# ============================================================
# 3. FEATURE SELECTION & SCALING
# ============================================================
n_features = min(30, X_train.shape[1])
rf_selector = RandomForestClassifier(n_estimators=100, random_state=SEED)
rfe = RFE(estimator=rf_selector, n_features_to_select=n_features)

X_train_sel = rfe.fit_transform(X_train, y_train)
X_val_sel   = rfe.transform(X_val)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_sel)
X_val_scaled   = scaler.transform(X_val_sel)

print(f"✅ Features: {X_train_sel.shape[1]} selected")

# ============================================================
# 4. CROSS-VALIDATION (بدون Split! فقط برای ارزیابی)
# ============================================================
print("\n" + "=" * 60)
print("Cross-Validation on Training Data (No Split)")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# مدل با Regularization قوی
cv_model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=6,
    min_child_samples=20,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=0.1,
    class_weight="balanced",
    random_state=SEED,
    verbose=-1
)

cv_scores = cross_val_score(cv_model, X_train_scaled, y_train, cv=cv, scoring='f1_macro')
print(f"✅ CV F1-Macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================
# 5. TRAIN ENSEMBLE (با Regularization قوی)
# ============================================================
print("\nTraining Ensemble (Strong Regularization)...")

# XGBoost با Regularization
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=SEED,
    eval_metric="logloss"
)
xgb_model.fit(X_train_scaled, y_train)

# LightGBM با Regularization
lgbm_model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=6,
    min_child_samples=20,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=0.1,
    class_weight="balanced",
    random_state=SEED,
    verbose=-1
)
lgbm_model.fit(X_train_scaled, y_train)

# RandomForest با Regularization
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=SEED
)
rf_model.fit(X_train_scaled, y_train)

# Ensemble Probabilities
p_ens = (
    xgb_model.predict_proba(X_val_scaled) * 0.35 +
    lgbm_model.predict_proba(X_val_scaled) * 0.35 +
    rf_model.predict_proba(X_val_scaled) * 0.30
)

# ============================================================
# 6. HYBRID LOCAL ROUTING (KNN + Ensemble)
# ============================================================
print("Applying Hybrid Local-Global Routing...")

nn_scaler = StandardScaler()
X_tr_raw_scaled = nn_scaler.fit_transform(train_df[oc])
X_val_raw_scaled = nn_scaler.transform(val_df[oc])

final_probs = np.zeros((len(val_df), 3))

for i in range(len(val_df)):
    c = val_df.iloc[i]['COMP_NAME']
    loc = val_df.iloc[i]['MP_LOC']

    tr_idx = np.where((train_df['COMP_NAME'] == c) & (train_df['MP_LOC'] == loc))[0]

    if len(tr_idx) == 0:
        tr_idx = np.where(train_df['COMP_NAME'] == c)[0]
    if len(tr_idx) == 0:
        tr_idx = np.arange(len(train_df))

    X_sub = X_tr_raw_scaled[tr_idx]
    y_sub = train_df.iloc[tr_idx]['Label'].values

    distances = np.linalg.norm(X_sub - X_val_raw_scaled[[i]], axis=1)

    k = min(5, len(tr_idx))
    nearest_k_idx = np.argsort(distances)[:k]
    nearest_dists = distances[nearest_k_idx]
    nearest_labels = y_sub[nearest_k_idx]

    knn_prob = np.zeros(3)
    weights = 1.0 / (nearest_dists + 1e-5)
    weights /= np.sum(weights)

    for lbl, w in zip(nearest_labels, weights):
        knn_prob[lbl] += w

    min_dist = nearest_dists[0]
    if min_dist < 0.5:
        alpha = 0.90
    elif min_dist < 1.5:
        alpha = 0.70
    elif min_dist < 3.0:
        alpha = 0.40
    else:
        alpha = 0.15

    final_probs[i] = alpha * knn_prob + (1 - alpha) * p_ens[i]

y_pred = np.argmax(final_probs, axis=1)

# ============================================================
# 7. CHECK OVERFITTING
# ============================================================
print("\n" + "=" * 60)
print("Overfitting Check")
print("=" * 60)

y_pred_train = xgb_model.predict(X_train_scaled)
train_acc = accuracy_score(y_train, y_pred_train)
train_f1 = f1_score(y_train, y_pred_train, average='macro')

print(f"Train Accuracy: {train_acc*100:.2f}%")
print(f"Train F1-Macro: {train_f1*100:.2f}%")

if train_acc > 92:
    print("⚠️ WARNING: Still overfitting! Increase regularization.")
elif train_acc < 75:
    print("⚠️ WARNING: Underfitting! Decrease regularization.")
else:
    print("✅ Balanced! Good generalization.")

# ============================================================
# 8. SAVE SUBMISSION
# ============================================================
submission = val_df.copy()
submission['Label'] = y_pred.astype(int)
submission.to_csv(SUBMISSION_PATH, index=False)

print(f"\n{'=' * 60}")
print("✅ SUBMISSION GENERATED!")
print(f"  File: {SUBMISSION_PATH}")
print(f"  Rows: {len(submission)}")
print(f"  Distribution:")
print(f"    Class 0: {sum(y_pred == 0)}")
print(f"    Class 1: {sum(y_pred == 1)}")
print(f"    Class 2: {sum(y_pred == 2)}")

if len(submission) == 600:
    print("\n✅ PERFECT! 600 rows ready.")

print(f"{'=' * 60}")