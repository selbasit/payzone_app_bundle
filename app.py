
import streamlit as st, json, joblib
import pandas as pd, numpy as np
import plotly.express as px, plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score

from utils import parse_las_text, build_features, compute_baseline_sw_vsh

st.set_page_config(page_title="Pay Zone, Sw & Vsh — Resilient App", layout="wide")
st.title("Pay Zone, Water Saturation (Sw), and Shale Volume (Vsh) — Resilient Predictor")

with st.sidebar:
    st.header("Options")
    st.caption("This build supports **bundled models** or **train-on-upload** fallback.")

    # Formation-based cutoff presets (optional)
    FORMATION_PRESETS = {
        "Nayil": {"vsh": 0.35, "phie": 0.08, "sw": 0.60},
        "Amal":  {"vsh": 0.30, "phie": 0.10, "sw": 0.55},
    }
    formation = st.selectbox("Formation preset", ["Custom", "Nayil", "Amal"], index=0)

    # Defaults (can still be adjusted)
    if formation != "Custom":
        _vsh0  = FORMATION_PRESETS[formation]["vsh"]
        _phie0 = FORMATION_PRESETS[formation]["phie"]
        _sw0   = FORMATION_PRESETS[formation]["sw"]
    else:
        _vsh0, _phie0, _sw0 = 0.35, 0.08, 0.60

    prob_thr = st.slider("PayZone probability cutoff", 0.0, 1.0, 0.5, 0.01)
    vsh_cut = st.slider("Vsh cutoff", 0.0, 1.0, float(_vsh0), 0.01)
    phi_cut = st.slider("Phi_eff cutoff", 0.0, 0.3, float(_phie0), 0.005)
    sw_cut  = st.slider("Sw cutoff", 0.0, 1.0, float(_sw0), 0.01)

@st.cache_resource
def load_artifacts():
    """
    Try to load pickled models. If it fails (e.g., sklearn version mismatch),
    return ok=False to trigger train-on-upload fallback.
    """
    try:
        with open("features.json","r") as f:
            feats = json.load(f)
        vsh = joblib.load("model_vsh.pkl")
        sw  = joblib.load("model_sw.pkl")
        pay = joblib.load("model_payzone.pkl")
        return feats, vsh, sw, pay, True
    except Exception as e:
        return None, None, None, None, False

feature_names, model_vsh, model_sw, model_pay, ok = load_artifacts()

with st.sidebar:
    mode = st.radio("Model source", ["Bundled models", "Train on uploaded data"],
                    index=(0 if ok else 1))
    if not ok and mode == "Bundled models":
        st.warning("Bundled models couldn't be loaded in this runtime. Using **Train on uploaded data** instead.")
        mode = "Train on uploaded data"

uploaded = st.file_uploader("Upload LAS (.las text) or CSV", type=["las","txt","csv"])

if uploaded is None:
    st.info("Upload a file to proceed.")
    st.stop()

# Parse file
text = uploaded.read().decode("utf-8", errors="ignore")
try:
    df = parse_las_text(text)
except Exception as e:
    st.error(f"Failed to parse file: {e}")
    st.stop()

st.success(f"Parsed file with shape: {df.shape}")
st.dataframe(df.head())

# --- Column auto-mapping + basic validation (prevents depth/index issues) ---
EXPECTED_COLUMNS = {
    "DEPT": ["DEPT", "DEPTH", "MD", "DEPTH_MD", "DEPT_m", "DEPTH_m", "Depth", "depth"],
    "GR":   ["GR", "GAMMA", "GAMMA_RAY", "CGR"],
    "ZDEN": ["ZDEN", "RHOB", "RHOZ", "DEN", "DENS"],
    "CNC":  ["CNC", "NPHI", "NEUT", "TNPH"],
    "RD":   ["RD", "RT", "RDEEP", "ILD", "LLD", "AT90", "RESDEEP"],
    "RS":   ["RS", "RMED", "ILM", "LLS", "AT60", "RESMED"],
    "RMSL": ["RMSL", "RSHAL", "MSFL", "AT10"],
}

def _auto_map_columns(df: pd.DataFrame):
    mapping = {}
    cols_upper = {c.upper(): c for c in df.columns}
    for canonical, variants in EXPECTED_COLUMNS.items():
        for v in variants:
            if v.upper() in cols_upper:
                mapping[canonical] = cols_upper[v.upper()]
                break
    return mapping

mapping = _auto_map_columns(df)

# Normalize depth column -> DEPT (critical for Net/Gross)
if "DEPT" not in mapping:
    st.warning("⚠ No depth column found (DEPT/DEPTH/MD). Using row index as depth; Net/Gross may be unreliable.")
    df["DEPT"] = np.arange(len(df), dtype=float)
else:
    if mapping["DEPT"] != "DEPT":
        df = df.rename(columns={mapping["DEPT"]: "DEPT"})
    df["DEPT"] = pd.to_numeric(df["DEPT"], errors="coerce")
    df = df.dropna(subset=["DEPT"]).sort_values("DEPT").reset_index(drop=True)

# Map other curves to expected names when possible (keeps utils.py features working)
rename_map = {}
for canon in ["GR", "ZDEN", "CNC", "RD", "RS", "RMSL"]:
    if canon in mapping and mapping[canon] != canon:
        rename_map[mapping[canon]] = canon
if rename_map:
    df = df.rename(columns=rename_map)

# Warn about missing key curves (app can still run, but quality may degrade)
recommended = ["GR", "ZDEN", "CNC", "RD"]
missing_rec = [c for c in recommended if c not in df.columns or df[c].isna().all()]
if missing_rec:
    st.warning("⚠ Missing or empty recommended curves: " + ", ".join(missing_rec) +
               ". Predictions/Net-Gross may be degraded depending on available data.")


# Build features and align to schema (if bundled models are used)
X = build_features(df)

# baseline physics (used for pay labeling in train-on-upload and optional comparisons)
Vsh_base, Sw_base, Phi_eff_base = compute_baseline_sw_vsh(df, rw=0.12)

# Combined deep resistivity for labeling convenience
RDeep = df.get("RD", np.nan).fillna(df.get("RS")).fillna(df.get("RMSL"))

def shade_pay(fig, flag, depth):
    intervals, start = [], None
    for i, f in enumerate(flag):
        if f and start is None:
            start = depth[i]
        if (not f or i==len(flag)-1) and start is not None:
            end = depth[i] if not f else depth[i]
            if end < start: start, end = end, start
            if (end - start) >= 0.5:
                intervals.append((float(start), float(end)))
            start = None
    for (s,e) in intervals:
        for col in [1,2,3,4]:
            fig.add_hrect(y0=s, y1=e, row=1, col=col, line_width=0, fillcolor="orange", opacity=0.15)
    return intervals

if mode == "Bundled models":
    # Align feature order
    for f in feature_names:
        if f not in X.columns: X[f] = np.nan
    X = X[feature_names]

    # Predict
    vsh_pred = model_vsh.predict(X)
    sw_pred  = model_sw.predict(X)
    pay_prob = model_pay.predict_proba(X)[:,1] if hasattr(model_pay, "predict_proba") else model_pay.predict(X)
    pay_pred = (pay_prob >= prob_thr).astype(int)

else:
    # Train light models on uploaded data (fallback)
    X["RDeep"] = RDeep
    X["logRDeep"] = np.log10(np.clip(X["RDeep"], 1e-3, np.inf))

    # Label pay using baseline physics + cutoffs
    pay_label = ((Vsh_base < vsh_cut) & (Phi_eff_base > phi_cut) & (Sw_base < sw_cut) & (RDeep > 10)).astype(int)

    # Mask rows with essential data
    mask = X["RDeep"].notna() & np.isfinite(Vsh_base) & np.isfinite(Sw_base)
    if mask.sum() < 50:
        st.error("Not enough valid rows to train fallback models. Provide more data or ensure GR/ZDEN/CNC/RD are present.")
        st.stop()

    Xtrain = X[mask]
    y_vsh  = Vsh_base[mask]
    y_sw   = Sw_base[mask]
    y_pay  = pay_label[mask]

    # Train
    reg_vsh = Pipeline([("imp", SimpleImputer(strategy="median")),
                        ("rf", RandomForestRegressor(n_estimators=80, max_depth=12, random_state=42))]).fit(Xtrain, y_vsh)
    reg_sw  = Pipeline([("imp", SimpleImputer(strategy="median")),
                        ("rf", RandomForestRegressor(n_estimators=80, max_depth=12, random_state=42))]).fit(Xtrain, y_sw)
    clf_pay = Pipeline([("imp", SimpleImputer(strategy="median")),
                        ("rf", RandomForestClassifier(n_estimators=120, max_depth=12, class_weight="balanced", random_state=42))]).fit(Xtrain, y_pay)

    # Predict on all rows
    vsh_pred = reg_vsh.predict(X)
    sw_pred  = reg_sw.predict(X)
    pay_prob = clf_pay.predict_proba(X)[:,1]
    pay_pred = (pay_prob >= prob_thr).astype(int)

# Build results table
out = pd.DataFrame({
    "DEPTH_m": df["DEPT"].values,
    "GR_API": df.get("GR", np.nan),
    "ZDEN_gcc": df.get("ZDEN", np.nan),
    "RD_ohmm": df.get("RD", np.nan),
    "RS_ohmm": df.get("RS", np.nan),
    "RMSL_ohmm": df.get("RMSL", np.nan),
    "Vsh_pred": vsh_pred,
    "Sw_pred": sw_pred,
    "PayZone_prob": pay_prob,
    "PayZone_pred": pay_pred
})

st.download_button("⬇️ Download Predictions (CSV)",
                   out.to_csv(index=False).encode("utf-8"),
                   file_name="predictions.csv",
                   mime="text/csv")

# Plotly composite (4 tracks) with shaded pay
fig = make_subplots(rows=1, cols=4, shared_yaxes=True, horizontal_spacing=0.05,
                    subplot_titles=("GR","Resistivity (log)","Vsh & Sw (pred)","Pay Probability"))
if "GR_API" in out:
    fig.add_traces(px.line(out, x="GR_API", y="DEPTH_m").data, rows=[1], cols=[1])
for c in ["RD_ohmm","RS_ohmm","RMSL_ohmm"]:
    if c in out and out[c].notna().any():
        tr = px.line(out, x=c, y="DEPTH_m").data[0]; tr.name = c.replace("_ohmm","")
        fig.add_trace(tr, row=1, col=2)
fig.update_xaxes(type="log", row=1, col=2)
fig.add_traces(px.line(out, x="Vsh_pred", y="DEPTH_m").data, rows=[1], cols=[3])
fig.add_traces(px.line(out, x="Sw_pred", y="DEPTH_m").data, rows=[1], cols=[3])
fig.add_traces(px.line(out, x="PayZone_prob", y="DEPTH_m").data, rows=[1], cols=[4])
fig.update_yaxes(autorange="reversed")
_intervals_shade = shade_pay(fig, out["PayZone_pred"].values.astype(bool), out["DEPTH_m"].values)
fig.update_layout(height=900, width=1500, title=f"Predictions ({mode})", showlegend=True)
st.plotly_chart(fig, use_container_width=True)

# ---------------- Net/Gross thickness (fixed) ----------------
def thickness_from_mask(depth: np.ndarray, mask: np.ndarray) -> float:
    """Robust thickness integration using depth steps (handles irregular sampling)."""
    depth = np.asarray(depth, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    if len(depth) < 2 or mask.sum() == 0:
        return 0.0

    # Ensure increasing depth
    if depth[0] > depth[-1]:
        depth = depth[::-1]
        mask = mask[::-1]

    dz = np.diff(depth)
    dz[~np.isfinite(dz)] = 0.0
    dz[dz < 0] = 0.0

    dz_last = np.nanmedian(dz[dz > 0]) if np.any(dz > 0) else 0.0
    dz_full = np.r_[dz, dz_last]

    return float(np.sum(dz_full[mask]))

def intervals_from_mask(depth: np.ndarray, mask: np.ndarray, min_thk=0.5):
    """Intervals for shading/visuals only (not used for thickness math)."""
    depth = np.asarray(depth, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if len(depth) == 0:
        return []

    if depth[0] > depth[-1]:
        depth = depth[::-1]
        mask = mask[::-1]

    idx = np.where(mask)[0]
    if idx.size == 0:
        return []

    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]

    dz = np.diff(depth)
    dz_last = np.nanmedian(dz[dz > 0]) if np.any(dz > 0) else 0.0
    dz_full = np.r_[dz, dz_last]

    intervals = []
    for s, e in zip(starts, ends):
        y0 = depth[s]
        y1 = depth[e] + dz_full[e]  # include last sample thickness
        if (y1 - y0) >= min_thk:
            intervals.append((float(y0), float(y1)))
    return intervals

# Rebuild shading intervals with the fixed interval builder
depth = out["DEPTH_m"].values.astype(float)
intervals = intervals_from_mask(depth, out["PayZone_pred"].values.astype(bool), min_thk=0.5)

# NET = predicted pay
net_mask = out["PayZone_pred"].values.astype(bool)

# GROSS = simple reservoir-quality flag (same intent as before, but thickness computed correctly)
gross_mask = ((df.get("ZDEN", np.nan).notna()) & (RDeep > 10) & (Phi_eff_base > phi_cut)).fillna(False).values.astype(bool)

net_thk = thickness_from_mask(depth, net_mask)
gross_thk = thickness_from_mask(depth, gross_mask)
ng_ratio = (net_thk / gross_thk * 100.0) if gross_thk > 0 else 0.0

st.subheader("Net/Gross Summary (Fixed)")
st.write({
    "Gross thickness (approx)": f"{gross_thk:.1f}",
    "Net pay (predicted)": f"{net_thk:.1f}",
    "N/G (%)": f"{ng_ratio:.1f}"
})
