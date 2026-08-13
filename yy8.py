import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from lightgbm import LGBMClassifier
import xgboost as xgb

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
TRAIN_FILE_PATH = "DataSetbearing-failure.csv"
TEST_FILE_PATH = "Bearing_Validation.csv"
SUBMISSION_PATH = "submission8.csv"


# ============================================================
# 1. DATA LOADING FUNCTION
# ============================================================
def load_dataset(path):
    try:

        return pd.read_csv(path, encoding='utf-16', sep='\t')
    except Exception:

        return pd.read_csv(path)


print("=" * 70)
print("Bearing Fault Detection - Automated Pipeline")
print("=" * 70)

train_df = load_dataset(TRAIN_FILE_PATH)
val_df = load_dataset(TEST_FILE_PATH)


train_df = train_df.dropna(subset=['Label']).reset_index(drop=True)

print(f"✓ Train Loaded: {train_df.shape}")
print(f"✓ Test/Validation Loaded: {val_df.shape}")

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


    df_out["peak_to_rms"] = df_out["Vel, Peak (RMS)"] / (df_out["Vel, Rms (RMS)"] + eps)
    df_out["pp_to_rms"] = df_out["Vel, Peak to peak (RMS)"] / (df_out["Vel, Rms (RMS)"] + eps)
    df_out["acc_vel_ratio"] = df_out["Acc, Rms (RMS)"] / (df_out["Vel, Rms (RMS)"] + eps)

    df_out["crest_factor"] = df_out["Vel, Peak (RMS)"] / (df_out["Vel, Rms (RMS)"] + eps)
    df_out["kurtosis_index"] = df_out["Kurt (RMS)"] / 3.0
    df_out["severity_index"] = df_out["Vel, Rms (RMS)"] * df_out["Acc, Rms (RMS)"] * df_out["Crest (RMS)"]
    df_out["early_fault_index"] = (df_out["Kurt (RMS)"] * df_out["Crest (RMS)"]) / (df_out["Acc, Rms (RMS)"] + eps)


    for col in oc:
        df_out[f"{col}_log"] = np.log1p(np.maximum(0, df_out[col]))
        df_out[f"{col}_sq"] = df_out[col] ** 2


    df_out["kurt_crest"] = df_out["Kurt (RMS)"] * df_out["Crest (RMS)"]
    df_out["vel_acc"] = df_out["Vel, Rms (RMS)"] * df_out["Acc, Rms (RMS)"]

    return df_out


X_train_feat = extract_features(train_df)
X_val_feat = extract_features(val_df)

le_comp = LabelEncoder()
le_comp.fit(pd.concat([train_df['COMP_NAME'], val_df['COMP_NAME']]).astype(str))
X_train_feat["COMP_NAME_encoded"] = le_comp.transform(train_df["COMP_NAME"].astype(str))
X_val_feat["COMP_NAME_encoded"]   = le_comp.transform(val_df["COMP_NAME"].astype(str))

le_loc = LabelEncoder()
le_loc.fit(pd.concat([train_df['MP_LOC'], val_df['MP_LOC']]).astype(str))
X_train_feat["MP_LOC_encoded"] = le_loc.transform(train_df["MP_LOC"].astype(str))
X_val_feat["MP_LOC_encoded"] = le_loc.transform(val_df["MP_LOC"].astype(str))

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
X_val = X_val_feat.drop(columns=[c for c in drop_cols if c in X_val_feat.columns])

# ============================================================
# 3. FEATURE SELECTION & SCALING
# ============================================================
n_features = min(35, X_train.shape[1])
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=rf_selector, n_features_to_select=n_features)

X_train_sel = rfe.fit_transform(X_train, y_train)
X_val_sel = rfe.transform(X_val)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_sel)
X_val_scaled = scaler.transform(X_val_sel)

# ============================================================
# 4. ENSEMBLE TRAINING
# ============================================================
print("\nTraining Ensemble Models...")

xgb_model = xgb.XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss"
)
xgb_model.fit(X_train_scaled, y_train)

lgbm_model = LGBMClassifier(
    n_estimators=500, max_depth=8, learning_rate=0.03,
    num_leaves=63, class_weight="balanced", random_state=42, verbose=-1
)
lgbm_model.fit(X_train_scaled, y_train)

rf_model = RandomForestClassifier(
    n_estimators=300, max_depth=10, class_weight="balanced", random_state=42
)
rf_model.fit(X_train_scaled, y_train)

p_ens = (
        xgb_model.predict_proba(X_val_scaled) * 0.4 +
        lgbm_model.predict_proba(X_val_scaled) * 0.4 +
        rf_model.predict_proba(X_val_scaled) * 0.2
)

# ============================================================
# 5. STRATEGY ROUTING (1-NN vs Ensemble)
# ============================================================
print("Applying Cluster Strategy Routing...")

nn_scaler = StandardScaler()
X_tr_raw_scaled = nn_scaler.fit_transform(train_df[oc])
X_val_raw_scaled = nn_scaler.transform(val_df[oc])

y_pred = np.zeros(len(val_df), dtype=int)

for i in range(len(val_df)):
    c = val_df.iloc[i]['COMP_NAME']
    loc = val_df.iloc[i]['MP_LOC']


    tr_idx = np.where((train_df['COMP_NAME'] == c) & (train_df['MP_LOC'] == loc))[0]

    if len(tr_idx) == 0:
        tr_idx = np.where(train_df['COMP_NAME'] == c)[0]
    if len(tr_idx) == 0:
        tr_idx = np.arange(len(train_df))

    if len(tr_idx) <= 10:
        X_sub = X_tr_raw_scaled[tr_idx]
        y_sub = train_df.iloc[tr_idx]['Label'].values
        distances = np.linalg.norm(X_sub - X_val_raw_scaled[[i]], axis=1)
        y_pred[i] = y_sub[np.argmin(distances)]
    else:
        y_pred[i] = np.argmax(p_ens[i])

# ============================================================
# 6. SAVE SUBMISSION
# ============================================================
submission = val_df.copy()
submission['Label'] = y_pred.astype(int)
submission.to_csv(SUBMISSION_PATH, index=False)

print(f"\n{'=' * 60}")
print("✓ Submission file generated successfully!")
print(f"  Total Output Rows:     {len(submission)}")
print(f"  Class 0 (Healthy):     {sum(y_pred == 0)}")
print(f"  Class 1 (Severe):      {sum(y_pred == 1)}")
print(f"  Class 2 (Mild):        {sum(y_pred == 2)}")
print(f"{'=' * 60}")