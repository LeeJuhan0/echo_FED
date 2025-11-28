# ============================================================
# Forecasting CPI, ANFCI, and NFCI with FOMC Sentiment Index (Python)
# LINEAR-ONLY VERSION
# - Linear regression (with/without sentiment)
# - Rolling window evaluation: train 30 → predict next (event-based)
# - Granger causality (lag 1): Linear only
# - Metrics: RMSE, MAE, MAPE only
# - Data restricted to 2017-01 through 2022-12
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests


# -----------------------------
# Utilities
# -----------------------------
def clean_month_string(s: str) -> str:
    import numpy as np, re
    if s is None:
        return np.nan
    s = str(s).strip()
    if s == "":
        return np.nan
    m = re.search(r"(\d{4})[^\d]?(\d{1,2})", s)
    if not m:
        return np.nan
    yyyy = int(m.group(1)); mm = int(m.group(2))
    if mm < 1 or mm > 12:
        return np.nan
    return f"{yyyy}-{mm:02d}"


# -----------------------------
# New readers: event-based prev/next + event date → month
# -----------------------------
def _parse_event_date(s):
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

def read_event_nfci_anfci(path="data/anfci_nfci.csv"):
    df = pd.read_csv(path, sep=None, engine="python")
    cols = {c.lower().strip(): c for c in df.columns}
    req = ["prev_nfci", "prev_anfci", "next_nfci", "next_anfci", "statement_date"]
    missing = [r for r in req if r not in cols]
    if missing:
        raise KeyError(f"Missing required columns in {path}: {missing}. Found: {list(df.columns)}")
    out = pd.DataFrame({
        "event_date_nfci": _parse_event_date(df[cols["statement_date"]]),
        "anfci_prev": pd.to_numeric(df[cols["prev_anfci"]], errors="coerce"),
        "anfci_next": pd.to_numeric(df[cols["next_anfci"]], errors="coerce"),
        "nfci_prev":  pd.to_numeric(df[cols["prev_nfci"]],  errors="coerce"),
        "nfci_next":  pd.to_numeric(df[cols["next_nfci"]],  errors="coerce"),
    }).dropna(subset=["event_date_nfci"])
    out["month"] = out["event_date_nfci"].dt.to_period("M").dt.to_timestamp()
    out = out.dropna(subset=["month"]).sort_values(["month", "event_date_nfci"]).reset_index(drop=True)
    return out

def read_event_cpi(path="data/cpi.csv"):
    df = pd.read_csv(path, sep=None, engine="python")
    cols = {c.lower().strip(): c for c in df.columns}
    req = ["prev_cpi_value", "next_cpi_value", "fomc_date"]
    missing = [r for r in req if r not in cols]
    if missing:
        raise KeyError(f"Missing required columns in {path}: {missing}. Found: {list(df.columns)}")
    out = pd.DataFrame({
        "event_date_cpi": _parse_event_date(df[cols["fomc_date"]]),
        "cpi_prev": pd.to_numeric(df[cols["prev_cpi_value"]], errors="coerce"),
        "cpi_next": pd.to_numeric(df[cols["next_cpi_value"]], errors="coerce"),
    }).dropna(subset=["event_date_cpi"])
    out["month"] = out["event_date_cpi"].dt.to_period("M").dt.to_timestamp()
    out = out.dropna(subset=["month"]).sort_values(["month", "event_date_cpi"]).reset_index(drop=True)
    return out

def read_sentiment(path="data/sentiment_13+19.csv"):
    df = pd.read_csv(path, sep=None, engine="python", header=0)
    if df.shape[1] == 1:
        s = df.iloc[:, 0].astype(str).str.strip()
        if s.iloc[0].lower().startswith("date"):
            cols = re.split(r"\s+|\t+|,", s.iloc[0])
            body = s.iloc[1:].str.split(r"\s+|\t+|,", expand=True)
            body.columns = [c.strip() for c in cols]
            df = body
        else:
            body = s.str.split(r"\s+|\t+|,", expand=True)
            if body.shape[1] < 4:
                raise ValueError("Sentiment must have 4 cols: date, statement, Theory, Policy")
            body = body.iloc[:, :4]; body.columns = ["date", "statement", "Theory", "Policy"]
            df = body
    df.columns = [str(c).strip() for c in df.columns]
    col_map, date_col = {}, None
    for c in df.columns:
        cl = c.lower().strip()
        if cl == "date": date_col = c
        elif cl in ("statement", "statment"): col_map[c] = "statement"
        elif cl == "theory": col_map[c] = "Theory"
        elif cl == "policy": col_map[c] = "Policy"
    if date_col is None:
        candidate = None
        for c in df.columns:
            vals = df[c].astype(str).head(5).tolist()
            if any(re.search(r"\d{4}[-/]\d{1,2}", v) for v in vals):
                candidate = c; break
        if candidate is None:
            raise KeyError("Sentiment file must have a 'date' column.")
        date_col = candidate
    df = df.rename(columns=col_map)
    required = ["statement", "Theory", "Policy"]
    if any(r not in df.columns for r in required):
        raise KeyError(f"Sentiment missing columns: {required}. Got: {list(df.columns)}")
    df["month"] = df[date_col].astype(str).map(clean_month_string)
    df = df.dropna(subset=["month"])
    df["month"] = pd.to_datetime(df["month"].astype(str) + "-01", errors="coerce")
    out = (
        df[["month", "statement", "Theory", "Policy"]]
        .groupby("month", as_index=False)
        .mean(numeric_only=True)
        .sort_values("month")
        .reset_index(drop=True)
    )
    return out


# -----------------------------
# Metrics: RMSE, MAE, MAPE only
# -----------------------------
def rmse(y, yhat):
    y = np.asarray(y, dtype=float); yhat = np.asarray(yhat, dtype=float)
    if y.shape != yhat.shape: raise ValueError("rmse: y and yhat must have the same shape.")
    if np.isnan(y).any() or np.isnan(yhat).any(): raise ValueError("rmse: NaN in inputs.")
    n = y.size
    return float(np.sqrt(np.sum((y - yhat) ** 2) / n))

def mae(y, yhat):
    y = np.asarray(y, dtype=float); yhat = np.asarray(yhat, dtype=float)
    if y.shape != yhat.shape: raise ValueError("mae: y and yhat must have the same shape.")
    if np.isnan(y).any() or np.isnan(yhat).any(): raise ValueError("mae: NaN in inputs.")
    n = y.size
    return float(np.sum(np.abs(y - yhat)) / n)

def mape(y, yhat):
    y = np.asarray(y, dtype=float); yhat = np.asarray(yhat, dtype=float)
    if y.shape != yhat.shape: raise ValueError("mape: y and yhat must have the same shape.")
    if np.isnan(y).any() or np.isnan(yhat).any(): raise ValueError("mape: NaN in inputs.")
    if np.any(y == 0): raise ZeroDivisionError("mape: y contains zero; undefined.")
    n = y.size
    return float(100.0 * np.sum(np.abs((y - yhat) / y)) / n)


# -----------------------------
# Rolling window evaluation (Linear only)
# -----------------------------
def evaluate_target(dat: pd.DataFrame, target: str, sentiment_col: str):
    assert target in ("cpi", "anfci", "nfci")
    assert sentiment_col in dat.columns, f"{sentiment_col} not in data"
    y_col = {"cpi": "cpi_next", "anfci": "anfci_next", "nfci": "nfci_next"}[target]
    prev_col = {"cpi": "cpi_prev", "anfci": "anfci_prev", "nfci": "nfci_prev"}[target]
    if y_col not in dat.columns or prev_col not in dat.columns:
        raise KeyError(f"Required columns '{y_col}' and '{prev_col}' not found in dat.")
    n = len(dat)
    if n < 31:
        raise ValueError(f"Need at least 31 event rows for rolling eval; got {n}.")
    train_ends = list(range(30, n))
    rows = []
    for te in train_ends:
        tr = dat.iloc[:te].copy()
        ts = dat.iloc[te:te+1].copy()
        y_tr = tr[y_col].values
        x_tr_sent = tr[[sentiment_col]].values
        x_ts_sent = ts[[sentiment_col]].values
        prev_ts = float(ts[prev_col].values[0])

        # Linear with sentiment (ARX: y_next ~ const + sentiment + prev)
        x_tr_arx = np.column_stack([x_tr_sent, tr[[prev_col]].values])
        x_ts_arx = np.array([[x_ts_sent[0, 0], prev_ts]], dtype=float)
        X_tr_w = sm.add_constant(x_tr_arx, has_constant="add")
        ols_w = sm.OLS(y_tr, X_tr_w).fit()
        X_ts_w = sm.add_constant(x_ts_arx, has_constant="add")
        pred_lm_w = float(ols_w.predict(X_ts_w)[0])

        # Linear no sentiment (naive prev)
        pred_lm_wo = prev_ts

        rows.append({
            "month": ts["month"].values[0],
            "y": ts[y_col].values[0],
            "lm_w": pred_lm_w,
            "lm_wo": pred_lm_wo,
        })

    results = pd.DataFrame(rows)

    def _metric_row(pred_col):
        _rmse = rmse(results["y"], results[pred_col])
        _mae = mae(results["y"], results[pred_col])
        _mape = mape(results["y"], results[pred_col])
        return _rmse, _mae, _mape

    labels = [
        ("Linear regression (sentiment)", "lm_w"),
        ("Linear regression (no sentiment)", "lm_wo"),
    ]
    metrics = pd.DataFrame([
        {"Model": name, "RMSE": _metric_row(col)[0], "MAE": _metric_row(col)[1], "MAPE": _metric_row(col)[2]}
        for name, col in labels
    ])
    return {"oos": results, "metrics": metrics}


# -----------------------------
# Granger causality tables (lag 1) — Linear only
# -----------------------------
def _pvalue_linear_granger(y, x, maxlag=1):
    df = pd.DataFrame({"y": y, "x": x}).dropna()
    if len(df) < maxlag + 5:
        return np.nan
    try:
        res = grangercausalitytests(df[["y", "x"]].values, maxlag=maxlag, verbose=False)
        return float(res[maxlag][0]["ssr_ftest"][1])
    except Exception:
        return np.nan

def granger_table(dat: pd.DataFrame, sentiment_col: str):
    required = [sentiment_col, "cpi_prev", "cpi_next", "anfci_prev", "anfci_next", "nfci_prev", "nfci_next"]
    for r in required:
        if r not in dat.columns:
            raise KeyError(f"Required column '{r}' not found for Granger table.")
    df = dat[["month", sentiment_col, "cpi_prev", "cpi_next", "anfci_prev", "anfci_next", "nfci_prev", "nfci_next"]].copy()
    df = df.dropna().sort_values("month").reset_index(drop=True)
    S = df[sentiment_col]

    rows = []
    p_lin = _pvalue_linear_granger(S, df["cpi_prev"], maxlag=1)
    rows.append({"Model": "Previous Month CPI -> Sentiment Index", "Linear Granger (p-value)": p_lin})
    p_lin = _pvalue_linear_granger(df["cpi_next"], S, maxlag=1)
    rows.append({"Model": "Sentiment Index -> Next Month CPI", "Linear Granger (p-value)": p_lin})
    p_lin = _pvalue_linear_granger(S, df["anfci_prev"], maxlag=1)
    rows.append({"Model": "Previous Month ANFCI -> Sentiment Index", "Linear Granger (p-value)": p_lin})
    p_lin = _pvalue_linear_granger(df["anfci_next"], S, maxlag=1)
    rows.append({"Model": "Sentiment Index -> Next Month ANFCI", "Linear Granger (p-value)": p_lin})
    p_lin = _pvalue_linear_granger(S, df["nfci_prev"], maxlag=1)
    rows.append({"Model": "Previous Month NFCI -> Sentiment Index", "Linear Granger (p-value)": p_lin})
    p_lin = _pvalue_linear_granger(df["nfci_next"], S, maxlag=1)
    rows.append({"Model": "Sentiment Index -> Next Month NFCI", "Linear Granger (p-value)": p_lin})

    out = pd.DataFrame(rows)
    out["Linear Granger (p-value)"] = out["Linear Granger (p-value)"].round(4)
    return out


# -----------------------------
# Data loading (2017-2022 restriction) and Runner
# -----------------------------
def load_and_prepare():
    nfci_anfci_ev = read_event_nfci_anfci("../data/anfci_nfci.csv")
    cpi_ev = read_event_cpi("../data/cpi.csv")
    sent = read_sentiment("../data/sentiment_13+19_llm.csv")
    dat = (
        nfci_anfci_ev.merge(cpi_ev, on="month", how="inner")
        .merge(sent, on="month", how="inner")
        .sort_values("month")
        .reset_index(drop=True if hasattr(pd.DataFrame, "reset_index") else False)
    )
    # Fallback for older pandas without reset_index(drop_by=...)
    if "month" not in dat.columns or "reset_index" in dir(pd.DataFrame):
        dat = dat.reset_index(drop=True)

    start = pd.Timestamp("2017-01-01")
    end = pd.Timestamp("2022-12-31")  # Restrict to 2017-01 through 2022-12
    dat = dat[(dat["month"] >= start) & (dat["month"] <= end)].reset_index(drop=True)

    dat["cpi"] = dat["cpi_next"]
    dat["anfci"] = dat["anfci_next"]
    dat["nfci"] = dat["nfci_next"]
    if len(dat) < 31:
        raise ValueError(f"Need at least 31 event rows after merge (2017-2022); got {len(dat)}.")
    return dat

def print_table_12(dat: pd.DataFrame, sentiment_col: str):
    res_cpi = evaluate_target(dat, target="cpi", sentiment_col=sentiment_col)["metrics"]
    res_anfci = evaluate_target(dat, target="anfci", sentiment_col=sentiment_col)["metrics"]
    res_nfci = evaluate_target(dat, target="nfci", sentiment_col=sentiment_col)["metrics"]

    cpi_tbl = res_cpi.rename(columns={"RMSE": "CPI_RMSE", "MAE": "CPI_MAE", "MAPE": "CPI_MAPE"})
    anfci_tbl = res_anfci.rename(columns={"RMSE": "ANFCI_RMSE", "MAE": "ANFCI_MAE", "MAPE": "ANFCI_MAPE"})
    nfci_tbl = res_nfci.rename(columns={"RMSE": "NFCI_RMSE", "MAE": "NFCI_MAE", "MAPE": "NFCI_MAPE"})

    tbl12 = cpi_tbl.merge(anfci_tbl, on="Model", how="inner").merge(nfci_tbl, on="Model", how="inner")
    metric_cols = [c for c in tbl12.columns if c != "Model"]
    tbl12[metric_cols] = tbl12[metric_cols].astype(float).round(3)
    return tbl12

def main():
    dat = load_and_prepare()
    sentiments = ["statement", "Theory", "Policy"]
    for s_col in sentiments:
        if s_col not in dat.columns:
            continue
        print(f"\n=== Table 11: Granger causality (lag 1, Linear only) using sentiment: {s_col} (2017-2022) ===")
        tbl11 = granger_table(dat, sentiment_col=s_col)
        print(tbl11.to_string(index=False))

        print(f"\n=== Table 12: Mean forecasting accuracy (Linear only) using sentiment: {s_col} (2017-2022) ===")
        tbl12 = print_table_12(dat, sentiment_col=s_col)
        cols = [
            "Model",
            "CPI_RMSE", "CPI_MAE", "CPI_MAPE",
            "ANFCI_RMSE", "ANFCI_MAE", "ANFCI_MAPE",
            "NFCI_RMSE", "NFCI_MAE", "NFCI_MAPE",
        ]
        print(tbl12[cols].to_string(index=False))

if __name__ == "__main__":
    main()