# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 21:35:04 2025

@author: HUFS_MATH
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from statsmodels.tsa.ar_model import AutoReg

import re
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests, acf, pacf, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import norm
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit


# ============================================================
# Utilities
# ============================================================
def _first_pred(predicted_mean):
    return float(np.asarray(predicted_mean).ravel()[0])

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

def _parse_event_date(s):
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

# ============================================================
# Readers
# ============================================================
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

# ============================================================
# Metrics
# ============================================================
def rmse(y, yhat):
    y = np.asarray(y, dtype=float); yhat = np.asarray(yhat, dtype=float)
    if y.shape != yhat.shape: raise ValueError("rmse: shape mismatch.")
    if np.isnan(y).any() or np.isnan(yhat).any(): raise ValueError("rmse: NaN found.")
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def mae(y, yhat):
    y = np.asarray(y, dtype=float); yhat = np.asarray(yhat, dtype=float)
    if y.shape != yhat.shape: raise ValueError("mae: shape mismatch.")
    if np.isnan(y).any() or np.isnan(yhat).any(): raise ValueError("mae: NaN found.")
    return float(np.mean(np.abs(y - yhat)))

def mape(y, yhat):
    y = np.asarray(y, dtype=float); yhat = np.asarray(yhat, dtype=float)
    if y.shape != yhat.shape: raise ValueError("mape: shape mismatch.")
    if np.isnan(y).any() or np.isnan(yhat).any(): raise ValueError("mape: NaN found.")
    if np.any(y == 0): raise ZeroDivisionError("mape: zero in y.")
    return float(100.0 * np.mean(np.abs((y - yhat) / y)))

def mape_eps(y, yhat, eps=0.1):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    denom = np.maximum(np.abs(y), eps)
    return float(100.0 * np.mean(np.abs(y - yhat) / denom))

# ============================================================
# Helper: ADF Test
# ============================================================
def adf_test_on_residuals(
        resid,
        var="unknown",
        name="best_x residuals",
        alpha=0.05,
        regression="c",
        autolag="AIC",
        maxlag=None,
        verbose=True
):
    s = pd.Series(resid).iloc[1:].dropna().astype(float).values

    if s.size < 10:
        if verbose: print(f"[ADF] 표본이 너무 작습니다 (n={s.size}).")
        return None

    stat, pval, usedlag, nobs, crit, icbest = adfuller(
        s, regression=regression, autolag=autolag, maxlag=maxlag
    )
    decision = "통과(정상)" if pval < alpha else "불통과(단위근 기각 실패)"

    if verbose:
        print(f"\n=== ADF 단위근 검정: {name} ===")
        print(f"옵션: regression='{regression}', autolag='{autolag}', maxlag={maxlag}")
        print(f"표본 크기 n={s.size}, 사용 시차 used_lag={usedlag}, 유효 관측 nobs={nobs}")
        print(f"ADF 통계량: {stat:.4f}, p-value: {pval:.6f}, ICbest: {icbest:.4f}")
        print("임계값:", ", ".join([f"{k}: {float(v):.4f}" for k, v in crit.items()]))
        print(f"판정 @ alpha={alpha:.2f}: {decision} varible : {var}")

    return {
        "stat": float(stat),
        "pvalue": float(pval),
        "usedlag": int(usedlag),
        "nobs": int(nobs),
        "critical_values": {k: float(v) for k, v in crit.items()},
        "icbest": float(icbest),
        "decision": decision
    }

# ============================================================
# ARIMA selection (p,d,q)
# ============================================================
def fit_arima(y, order, exog=None, enforce_stationarity=False):
    try:
        model = SARIMAX(endog=y, exog=exog, order=order, trend="c",
                        enforce_stationarity=enforce_stationarity, enforce_invertibility=False)
        res = model.fit(disp=False)
        return res
    except Exception:
        return None

def select_best_arima(y, var="unknown", exog=None,
                      p_list=(0,1,2),
                      d_list=(0,1),
                      q_list=(0,1,2),
                      check_adf=True):
    best = {"aic": np.inf, "res": None, "order": None}

    for p in p_list:
        for d in d_list:
            for q in q_list:
                res = fit_arima(y, (p,d,q), exog=exog, enforce_stationarity=False)
                if res is None or not np.isfinite(res.aic):
                    continue

                if res.aic < best["aic"]:
                    is_candidate = True

                    if check_adf:
                        adf_res = adf_test_on_residuals(
                            res.resid,
                            var=var,
                            name=f"resid_check_({p},{d},{q})",
                            alpha=0.05,
                            verbose=True
                        )
                        if adf_res is None or adf_res['pvalue'] >= 0.05:
                            is_candidate = False

                    if is_candidate:
                        best = {"aic": res.aic, "res": res, "order": (p,d,q)}
                        if check_adf:
                            print(f"({p},{d},{q}) AIC:{res.aic:.3f} ADF Pass")

    return best

# ============================================================
# Rolling window evaluation
# ============================================================
def evaluate_target(dat: pd.DataFrame, target: str, sentiment_col: str,
                    p_list=(0,1,2), d_list=(0,1), q_list=(0,1,2),
                    base_window=36, small_window=8,
                    use_fixed_ar_order=True,
                    fixed_ar_order=(1,0,0),
                    arx_use_prev=True):

    assert target in ("cpi", "anfci", "nfci")
    if sentiment_col not in dat.columns:
        raise KeyError(f"[evaluate_target] sentiment_col '{sentiment_col}' missing.")

    y_col_map = {
        "cpi": "cpi_next",
        "anfci": "anfci_next",
        "nfci": "nfci_next"
    }
    prev_col_map = {
        "cpi": "cpi_prev",
        "anfci": "anfci_prev",
        "nfci": "nfci_prev"
    }

    y_col = y_col_map[target]
    prev_col = prev_col_map[target]
    for c in [y_col, prev_col]:
        if c not in dat.columns:
            raise KeyError(f"[evaluate_target] Missing column: {c}")

    df_all = dat[["month", y_col, prev_col, sentiment_col]].dropna().copy()
    if df_all.empty:
        raise ValueError("[evaluate_target] No data after dropna.")
    df_all = df_all.sort_values("month").reset_index(drop=True)
    n = len(df_all)
    if n < 15:
        raise ValueError(f"[evaluate_target] Too few rows: n={n}")

    window_size = base_window if n >= 31 else small_window

    ar_lags  = fixed_ar_order[0]
    arx_lags = fixed_ar_order[0]

    rows = []

    for te in range(window_size, n):
        tr = df_all.iloc[te - window_size: te]
        ts = df_all.iloc[te:te+1]

        y_tr    = tr[y_col].values.astype(float)
        sent_tr = tr[sentiment_col].values.astype(float)
        prev_tr = tr[prev_col].values.astype(float)

        y_ts    = float(ts[y_col].values[0])
        sent_ts = float(ts[sentiment_col].values[0])
        prev_ts = float(ts[prev_col].values[0])

        # ---------------- AutoReg AR ----------------
        ar_model = AutoReg(endog=y_tr, lags=ar_lags, trend='c', exog=None,
                           hold_back=None, old_names=False)
        ar_res = ar_model.fit()
        ar_pred = float(ar_res.predict(start=len(y_tr), end=len(y_tr), exog=None))

        # ---------------- AutoReg ARX ----------------
        exog_tr_arx = np.column_stack([sent_tr])
        exog_ts_arx = np.array([[sent_ts]])

        arx_model = AutoReg(endog=y_tr, lags=arx_lags, trend='c', exog=exog_tr_arx,
                            hold_back=None, old_names=False)
        arx_res = arx_model.fit()
        arx_pred = float(arx_res.predict(start=len(y_tr), end=len(y_tr), exog_oos=exog_ts_arx))

        # ---------------- ARIMAX ----------------
        exog_tr_glob = np.column_stack([sent_tr])
        exog_ts_glob = np.array([[sent_ts]])

        best_arimax = select_best_arima(y_tr, var="eval_arimax", exog=exog_tr_glob,
                                        p_list=p_list, d_list=d_list, q_list=q_list,
                                        check_adf=False)

        if best_arimax["res"] is not None:
            resid_stats = adf_test_on_residuals(best_arimax["res"].resid, verbose=False)
            is_stationary = (resid_stats is not None and resid_stats['pvalue'] < 0.05)
            enforce = False if is_stationary else True

            final_model_arimax = SARIMAX(endog=y_tr, exog=exog_tr_glob,
                                         order=best_arimax["order"], trend="c",
                                         enforce_stationarity=enforce,
                                         enforce_invertibility=False).fit(disp=False)
            pred_arimax = _first_pred(final_model_arimax.get_forecast(steps=1, exog=exog_ts_glob).predicted_mean)
        else:
            pred_arimax = np.nan

        # ---------------- ARIMA ----------------
        best_arima = select_best_arima(y_tr, var="eval_arima", exog=None,
                                       p_list=p_list, d_list=d_list, q_list=q_list,
                                       check_adf=False)

        if best_arima["res"] is not None:
            resid_stats = adf_test_on_residuals(best_arima["res"].resid, verbose=False)
            is_stationary = (resid_stats is not None and resid_stats['pvalue'] < 0.05)
            enforce = False if is_stationary else True

            final_model_arima = SARIMAX(endog=y_tr, exog=None,
                                        order=best_arima["order"], trend="c",
                                        enforce_stationarity=enforce,
                                        enforce_invertibility=False).fit(disp=False)
            pred_arima = _first_pred(final_model_arima.get_forecast(steps=1).predicted_mean)
        else:
            pred_arima = np.nan

        rows.append({
            "month": ts["month"].values[0],
            "y": y_ts,
            "arx_mle": arx_pred,
            "ar_mle": ar_pred,
            "arima_w": pred_arimax,
            "arima_wo": pred_arima,
            "ar_lags": ar_lags,
            "arx_lags": arx_lags
        })

    results = pd.DataFrame(rows)

    def _metric_row(col):
        _rmse = rmse(results["y"], results[col])
        _mae  = mae(results["y"], results[col])
        if target in ("cpi",):
            _mape = mape(results["y"], results[col])
        else:
            _mape = mape_eps(results["y"], results[col], eps=0.1)
        return _rmse, _mae, _mape

    labels = [
        ("ARX AutoReg (lags={}, sentiment{})"
         .format(arx_lags, "" if arx_use_prev else ""), "arx_mle"),
        ("AR AutoReg (lags={})".format(ar_lags), "ar_mle"),
        ("ARIMAX (Best AIC per window, sentiment only)", "arima_w"),
        ("ARIMA  (Best AIC per window, no exog)", "arima_wo"),
    ]
    metrics = pd.DataFrame([
        {
            "Model": name,
            "RMSE": _metric_row(col)[0],
            "MAE":  _metric_row(col)[1],
            "MAPE": _metric_row(col)[2]
        }
        for name, col in labels
    ])

    return {
        "oos": results,
        "metrics": metrics,
        "window_size": window_size
    }

# ============================================================
# Granger causality (lag 1) - Modified to return "Stat (p-val)"
# ============================================================
def _pvalue_linear_granger(y, x, maxlag=1): # simple linear regression
    df = pd.DataFrame({"y": y, "x": x}).dropna()
    if len(df) < maxlag + 5:
        return np.nan
    try:
        res = grangercausalitytests(df[["y","x"]].values, maxlag=maxlag, verbose=False) # SSR 기반 F-검정
        # [수정] F-stat과 p-value를 문자열 포맷으로 반환
        f_stat = res[maxlag][0]["ssr_ftest"][0]
        p_val = res[maxlag][0]["ssr_ftest"][1]
        return f"{f_stat:.4f} ({p_val:.4f})"
    except Exception:
        return np.nan

def _pvalue_arima_granger(y, x,  var = "default", maxlag=1):
    var = var
    df = pd.DataFrame({"y": pd.Series(y), "x": pd.Series(x)}).dropna()
    if len(df) < maxlag + 5:
        return np.nan
    try:
        best_y = select_best_arima(df["y"].values, var, exog=None, check_adf=True)
        best_x = {"aic": 0, "res": x, "order": (0,0,0)}

        if best_y["res"] is None or best_x["res"] is None:
            return np.nan

        resid_y = pd.Series(best_y["res"].resid, index=df.index)
        resid_x = pd.Series(x, index=df.index)

        df_res = pd.DataFrame({"y": resid_y, "x": resid_x}).dropna()

        if len(df_res) < maxlag + 5:
            return np.nan

        res = grangercausalitytests(df_res[["y", "x"]].values, maxlag=maxlag, verbose=False)
        # [수정] F-stat과 p-value를 문자열 포맷으로 반환
        f_stat = res[maxlag][0]["ssr_ftest"][0]
        p_val = res[maxlag][0]["ssr_ftest"][1]
        return f"{f_stat:.4f} ({p_val:.4f})"
    except Exception:
        return np.nan

def _pvalue_arima_granger2(y, x, var,  maxlag=1):
    df = pd.DataFrame({"y": pd.Series(y), "x": pd.Series(x)}).dropna()
    if len(df) < maxlag + 5:
        return np.nan
    try:
        best_x = select_best_arima(df["x"].values, var, exog=None, check_adf=True)
        best_y = {"aic": 0, "res": y, "order": (0,0,0)}

        if best_y["res"] is None or best_x["res"] is None:
            return np.nan

        resid_y = pd.Series(y, index=df.index)
        resid_x = pd.Series(best_x["res"].resid, index=df.index)
        df_res = pd.DataFrame({"y": resid_y, "x": resid_x}).dropna()

        if len(df_res) < maxlag + 5:
            return np.nan

        res = grangercausalitytests(df_res[["y", "x"]].values, maxlag=maxlag, verbose=False)
        # [수정] F-stat과 p-value를 문자열 포맷으로 반환
        f_stat = res[maxlag][0]["ssr_ftest"][0]
        p_val = res[maxlag][0]["ssr_ftest"][1]
        return f"{f_stat:.4f} ({p_val:.4f})"
    except Exception:
        return np.nan

def granger_table(dat: pd.DataFrame, sentiment_col: str):
    required = [
        sentiment_col,
        "cpi_prev","cpi_next",
        "anfci_prev","anfci_next",
        "nfci_prev","nfci_next",
    ]
    for r in required:
        if r not in dat.columns:
            raise KeyError(f"Missing column {r} for Granger.")
    df = dat[[
        "month", sentiment_col,
        "cpi_prev","cpi_next",
        "anfci_prev","anfci_next",
        "nfci_prev","nfci_next",
    ]].dropna().sort_values("month").reset_index(drop=True)
    S = df[sentiment_col]

    # 반환값이 이제 문자열(stat (p-value))임
    def both(y, x, var):
        p_lin = _pvalue_linear_granger(y, x, maxlag=1)
        p_arima = _pvalue_arima_granger(y, x, var, maxlag=1)
        return p_lin, p_arima

    def both2(y, x, var):
        p_lin = _pvalue_linear_granger(y, x, maxlag=1)
        p_arima = _pvalue_arima_granger2(y, x, var, maxlag=1)
        return p_lin, p_arima

    rows = []
    # CPI
    p_lin, p_ar = both2(S, df["cpi_prev"], "cpi_prev")
    rows.append({"Model": "Previous Month CPI -> Sentiment Index",
                 "Linear Granger (Stat, p-val)": p_lin,
                 "ARIMA Granger (Stat, p-val)": p_ar})
    p_lin, p_ar = both(df["cpi_next"], S, "cpi_next")
    rows.append({"Model": "Sentiment Index -> Next Month CPI",
                 "Linear Granger (Stat, p-val)": p_lin,
                 "ARIMA Granger (Stat, p-val)": p_ar})
    # ANFCI
    p_lin, p_ar = both2(S, df["anfci_prev"], "anfci_prev")
    rows.append({"Model": "Previous Month ANFCI -> Sentiment Index",
                 "Linear Granger (Stat, p-val)": p_lin,
                 "ARIMA Granger (Stat, p-val)": p_ar})
    p_lin, p_ar = both(df["anfci_next"], S, "anfic_next")
    rows.append({"Model": "Sentiment Index -> Next Month ANFCI",
                 "Linear Granger (Stat, p-val)": p_lin,
                 "ARIMA Granger (Stat, p-val)": p_ar})
    # NFCI
    p_lin, p_ar = both2(S, df["nfci_prev"], "nfci_prev")
    rows.append({"Model": "Previous Month NFCI -> Sentiment Index",
                 "Linear Granger (Stat, p-val)": p_lin,
                 "ARIMA Granger (Stat, p-val)": p_ar})
    p_lin, p_ar = both(df["nfci_next"], S, "nfci_next")
    rows.append({"Model": "Sentiment Index -> Next Month NFCI",
                 "Linear Granger (Stat, p-val)": p_lin,
                 "ARIMA Granger (Stat, p-val)": p_ar})

    out = pd.DataFrame(rows)
    # [수정] 문자열이므로 round() 제거
    return out

# ============================================================
# Data loading & Tables (Missing functions restored)
# ============================================================
def load_and_prepare(start : str, end : str):
    nfci_anfci_ev = read_event_nfci_anfci("../data/anfci_nfci.csv")
    cpi_ev        = read_event_cpi("../data/cpi.csv")
    sent          = read_sentiment("../data/sentiment_13+19.csv")

    dat = (
        nfci_anfci_ev
        .merge(cpi_ev, on="month", how="inner")
        .merge(sent, on="month", how="inner")
        .sort_values("month")
        .reset_index(drop=True)
    )
    start = pd.Timestamp(start)
    end   = pd.Timestamp(end)
    dat = dat[(dat["month"] >= start) & (dat["month"] <= end)].reset_index(drop=True)

    # Convenience duplicates
    dat["cpi"]     = dat["cpi_next"]
    dat["anfci"]   = dat["anfci_next"]
    dat["nfci"]    = dat["nfci_next"]

    if len(dat) < 21:
        raise ValueError(f"Need at least 31 event rows after filtering; got {len(dat)}.")
    return dat

def build_table_12(dat: pd.DataFrame, sentiment_col: str):
    res_cpi      = evaluate_target(dat, "cpi",       sentiment_col)["metrics"]
    res_anfci    = evaluate_target(dat, "anfci",     sentiment_col)["metrics"]
    res_nfci     = evaluate_target(dat, "nfci",      sentiment_col)["metrics"]

    cpi_tbl     = res_cpi.rename(columns={"RMSE":"CPI_RMSE","MAE":"CPI_MAE","MAPE":"CPI_MAPE"})
    anfci_tbl   = res_anfci.rename(columns={"RMSE":"ANFCI_RMSE","MAE":"ANFCI_MAE","MAPE":"ANFCI_MAPE"})
    nfci_tbl    = res_nfci.rename(columns={"RMSE":"NFCI_RMSE","MAE":"NFCI_MAE","MAPE":"NFCI_MAPE"})

    tbl12 = (
        cpi_tbl
        .merge(anfci_tbl, on="Model", how="inner")
        .merge(nfci_tbl, on="Model", how="inner")

    )
    metric_cols = [c for c in tbl12.columns if c != "Model"]
    tbl12[metric_cols] = tbl12[metric_cols].astype(float).round(3)
    return tbl12

# ============================================================
# Main
# ============================================================
def main(start: str, end: str):
    dat = load_and_prepare(start, end)
    sentiments = ["statement", "Theory", "Policy"]
    period_label = f"{pd.Timestamp(start).year}-{pd.Timestamp(end).year}"

    for s_col in sentiments:
        if s_col not in dat.columns:
            continue

        print(f"\n=== Table 11: Granger causality (lag 1) using sentiment: {s_col} ({period_label}) ===")
        tbl11 = granger_table(dat, sentiment_col=s_col)
        print(tbl11.to_string(index=False))

        print(f"\n=== Table 12: Forecasting accuracy (Rolling, next-event) using sentiment: {s_col} ({period_label}) ===")
        tbl12 = build_table_12(dat, sentiment_col=s_col)
        cols_order = [
            "Model",
            "CPI_RMSE","CPI_MAE","CPI_MAPE",
            "ANFCI_RMSE","ANFCI_MAE","ANFCI_MAPE",
            "NFCI_RMSE","NFCI_MAE","NFCI_MAPE",
        ]
        print(tbl12[cols_order].to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARIMA/ARIMAX analysis")
    parser.add_argument("--start", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-12-31", help="End date (YYYY-MM-DD)")
    args = parser.parse_args()
    main(args.start, args.end)
