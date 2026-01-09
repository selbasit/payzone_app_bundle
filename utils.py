
import numpy as np
import pandas as pd

def parse_las_text(text: str) -> pd.DataFrame:
    """
    Robust LAS-text (LAS 2.0-like) parser with CSV fallback.
    Returns a DataFrame sorted by DEPT and replaces -999.25 with NaN.
    """
    lines = text.splitlines()
    curves, data_start, in_curve = [], None, False
    for i, ln in enumerate(lines):
        t = ln.strip()
        if t.lower().startswith("~curve"):
            in_curve = True
            continue
        if in_curve and t.startswith("~"):
            in_curve = False
        if in_curve and t and not t.startswith("#"):
            curves.append(t.split()[0])
        if t.startswith("~A"):
            data_start = i + 1
            break

    if data_start is None:
        # CSV fallback
        from io import StringIO
        df = pd.read_csv(StringIO(text))
        if "DEPT" in df.columns:
            df = df.sort_values("DEPT").reset_index(drop=True)
        df = df.replace(-999.25, np.nan)
        return df

    # Parse numeric ~A rows
    data = []
    for t in lines[data_start:]:
        t = t.strip()
        if not t:
            continue
        try:
            row = [float(x) for x in t.split()]
            data.append(row)
        except Exception:
            # skip malformed
            pass

    df = pd.DataFrame(np.array(data), columns=curves)
    df = df.replace(-999.25, np.nan).sort_values("DEPT").reset_index(drop=True)
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature builder (must match training): raw logs + combined deep resistivity + log transform + simple SMAs.
    """
    for c in ["GR","ZDEN","CNC","RD","RS","RMSL","SPBR","CAL","PE","DT"]:
        if c not in df.columns:
            df[c] = np.nan

    X = pd.DataFrame(index=df.index)
    X["GR"] = df["GR"]
    X["ZDEN"] = df["ZDEN"]
    X["CNC_frac"] = df["CNC"] / 100.0
    X["RD"] = df["RD"]; X["RS"] = df["RS"]; X["RMSL"] = df["RMSL"]
    X["RDeep"] = df["RD"].fillna(df["RS"]).fillna(df["RMSL"])
    X["logRDeep"] = np.log10(np.clip(X["RDeep"], 1e-3, np.inf))
    X["PE"] = df["PE"]; X["CAL"] = df["CAL"]; X["SPBR"] = df["SPBR"]; X["DT"] = df["DT"]

    for col in ["GR","ZDEN","CNC_frac","RDeep","logRDeep"]:
        X[f"{col}_ma5"] = X[col].rolling(window=5, min_periods=1, center=True).mean()
    return X

def compute_baseline_sw_vsh(df: pd.DataFrame, rw=0.12, rhom=2.65, rhof=1.00, a=1.0, m=2.0, n=2.0):
    """
    Physics-based baseline: Vsh (Larionov), Phi_eff (density+neutron with shale correction), Sw (Archie).
    """
    GR = df.get("GR", np.nan).astype(float)
    if np.all(~np.isfinite(GR)):
        IGR = np.full(len(df), np.nan)
    else:
        gr_min, gr_max = np.nanpercentile(GR, 5), np.nanpercentile(GR, 95)
        IGR = np.clip((GR - gr_min)/max(1e-6, (gr_max - gr_min)), 0, 1)

    Vsh = np.clip(0.083*(np.power(2, 3.7*IGR)-1), 0, 1)

    RHOB = df.get("ZDEN", np.nan).astype(float)
    phi_d = np.clip((rhom - RHOB)/max(1e-6, (rhom - rhof)), 0, 0.6)
    phi_n = np.clip(df.get("CNC", np.nan).astype(float)/100.0, 0, 0.6)
    phi_total = np.nanmean(np.vstack([phi_d, phi_n]), axis=0)
    phi_eff = np.clip(phi_total*(1 - Vsh), 0, 0.6)

    RDeep = df.get("RD", np.nan).fillna(df.get("RS")).fillna(df.get("RMSL")).astype(float)
    Rt = RDeep.copy()
    Rt[(~np.isfinite(Rt)) | (Rt <= 0)] = np.nan
    phi_used = phi_eff.copy()
    phi_used[~np.isfinite(phi_used)] = np.nan
    Sw = np.clip(((a*rw)/(np.power(phi_used, m)*Rt))**(1.0/n), 0, 1)

    return Vsh, Sw, phi_eff
