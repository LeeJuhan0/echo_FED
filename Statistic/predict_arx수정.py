import warnings
warnings.filterwarnings("ignore")
import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import re
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import norm
import matplotlib.pyplot as plt
import argparse
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit


# ============================================================
# Utilities
# ============================================================
def _first_pred(predicted_mean):
    return float(np.asarray(predicted_mean).ravel()[0])

def _safe_forecast(best, steps, exog, fallback_value):
    try:
        if best is None or best.get("res") is None:
            return float(fallback_value)
        pm = best["res"].get_forecast(steps=steps, exog=exog).predicted_mean
        return _first_pred(pm)
    except Exception:
        return float(fallback_value)

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

def read_event_unrate(path="data/unrate.csv"):
    df = pd.read_csv(path, sep=None, engine="python")
    cols = {c.lower().strip(): c for c in df.columns}
    req = ["prev_unrate_value", "next_unrate_value", "fomc_date"]
    missing = [r for r in req if r not in cols]
    if missing:
        raise KeyError(f"Missing required columns in {path}: {missing}. Found: {list(df.columns)}")
    out = pd.DataFrame({
        "event_date_unrate": _parse_event_date(df[cols["fomc_date"]]),
        "unrate_prev": pd.to_numeric(df[cols["prev_unrate_value"]], errors="coerce"),
        "unrate_next": pd.to_numeric(df[cols["next_unrate_value"]], errors="coerce"),
    }).dropna(subset=["event_date_unrate"])
    out["month"] = out["event_date_unrate"].dt.to_period("M").dt.to_timestamp()
    out = out.dropna(subset=["month"]).sort_values(["month", "event_date_unrate"]).reset_index(drop=True)
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
# ARIMA selection (p,d,q)
# ============================================================
def fit_arima(y, order, exog=None):
    # order = (p,d,q)
    try:
        model = SARIMAX(endog=y, exog=exog, order=order, trend="c",
                        enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        return res
    except Exception:
        return None

def select_best_arima(y, exog=None,
                      p_list=(0,1,2),
                      d_list=(0,1),
                      q_list=(0,1,2)):
    best = {"aic": np.inf, "res": None, "order": None}
    for p in p_list:
        for d in d_list:
            for q in q_list:
                res = fit_arima(y, (p,d,q), exog=exog)
                if res is None or not np.isfinite(res.aic):
                    continue
                if res.aic < best["aic"]:
                    best = {"aic": res.aic, "res": res, "order": (p,d,q)}
    return best

# ------------------------------------------------------------
# auto_aic: AIC 기반 ARMA(p,d,q) 탐색 과정을 출력하면서 최적 모델 선택
# ------------------------------------------------------------
def auto_aic(y, exog=None,
             p_list=(0,1,2),
             d_list=(0,1),
             q_list=(0,1,2),
             name=None,
             print_progress=True):
    """
    Iterate over (p,d,q) grid, print each candidate's AIC, and track the best.
    Returns dict: {'aic': best_aic, 'res': fitted_result, 'order': (p,d,q)}
    """
    title = f"[auto_aic] {name}" if name else "[auto_aic]"
    best = {"aic": np.inf, "res": None, "order": None}

    for p in p_list:
        for d in d_list:
            for q in q_list:
                order = (p, d, q)
                try:
                    res = fit_arima(y, order, exog=exog)
                    aic = res.aic if (res is not None and np.isfinite(res.aic)) else np.inf
                    if print_progress:
                        if np.isfinite(aic):
                            print(f"{title} try order={order} -> AIC={aic:.3f}")
                        else:
                            print(f"{title} try order={order} -> AIC=nan (skip)")
                    if np.isfinite(aic) and aic < best["aic"]:
                        best = {"aic": aic, "res": res, "order": order}
                        if print_progress:
                            print(f"{title} NEW BEST order={order} (AIC={aic:.3f})")
                except Exception as e:
                    if print_progress:
                        print(f"{title} try order={order} -> failed: {e}")
                    continue

    if print_progress:
        if best["order"] is not None:
            print(f"{title} BEST order={best['order']} with AIC={best['aic']:.3f}")
        else:
            print(f"{title} No valid model found.")
    return best

# ============================================================
# Residual diagnostics: ACF, PACF, Ljung–Box, optional plots
# ============================================================
def summarize_residuals(resid, nlags=12, plot=False, plot_prefix="resid"):
    resid = pd.Series(np.asarray(resid, dtype=float)).dropna().values
    if resid.size < nlags + 5:
        print(f"[resid] Not enough data for diagnostics (n={resid.size}, nlags={nlags}).")
        return

    # ACF
    acf_vals = acf(resid, nlags=nlags, fft=False, missing="drop")

    # PACF: 버전 호환을 위해 여러 방법 시도
    pacf_vals = None
    pacf_methods = ["yw", "ywmle", "burg", "ols", "ols-adjusted"]
    last_err = None
    for m in pacf_methods:
        try:
            pacf_vals = pacf(resid, nlags=nlags, method=m)
            print(f"[resid] PACF computed with method='{m}'")
            break
        except Exception as e:
            last_err = e
            continue
    if pacf_vals is None:
        print(f"[resid] PACF failed with all methods {pacf_methods}. Last error: {last_err}")
        pacf_vals = np.zeros(nlags + 1)

    # Ljung–Box
    lb_df = acorr_ljungbox(resid, lags=list(range(1, nlags+1)), return_df=True)

    # 표 출력
    diag = pd.DataFrame({
        "Lag": np.arange(1, nlags+1),
        "ACF": acf_vals[1:nlags+1],
        "PACF": pacf_vals[1:nlags+1],
        "LjungBox_pvalue": lb_df["lb_pvalue"].values
    })
    print("\n[Residual diagnostics] ACF/PACF and Ljung–Box p-values")
    print(diag.to_string(index=False, float_format=lambda x: f"{x: .4f}"))

    # 플롯 저장(선택)
    if plot:
        try:
            fig, ax = plt.subplots(figsize=(7,4))
            plot_acf(resid, lags=nlags, zero=False, ax=ax)
            ax.set_title("Residual ACF")
            fig.tight_layout()
            fig.savefig(f"{plot_prefix}_acf.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            pacf_plot_done = False
            for m in pacf_methods:
                try:
                    fig, ax = plt.subplots(figsize=(7,4))
                    plot_pacf(resid, lags=nlags, zero=False, method=m, ax=ax)
                    ax.set_title(f"Residual PACF (method={m})")
                    fig.tight_layout()
                    fig.savefig(f"{plot_prefix}_pacf.png", dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    pacf_plot_done = True
                    break
                except Exception:
                    plt.close('all')
                    continue

            if pacf_plot_done:
                print(f"[plots] Saved: {plot_prefix}_acf.png, {plot_prefix}_pacf.png")
            else:
                print("[plots] Could not generate PACF plot with any compatible method.")
        except Exception as e:
            print(f"[plots] Could not generate residual plots: {e}")

# ============================================================
# Rolling window evaluation
# ============================================================
import numpy as np
import pandas as pd
import statsmodels.api as sm

def _global_best_orders(dat: pd.DataFrame, y_col: str, sentiment_col: str, prev_col: str,
                        p_list=(0,1,2,3), d_list=(0,1), q_list=(0,1,2,3)):
    """
    전체 표본에서 ARIMAX(외생: sentiment + prev)와 ARIMA(외생 없음)의 AIC 최소 (p,d,q) order를 한 번만 선택.
    dat: 반드시 'month' 컬럼을 포함해야 함.
    """
    needed = [y_col, sentiment_col, prev_col]
    for c in needed:
        if c not in dat.columns:
            raise KeyError(f"[global_orders] Missing column: {c}")
    if "month" not in dat.columns:
        raise KeyError("[global_orders] 'month' column is required for sorting.")

    df_full = dat[["month"] + needed].dropna().copy()
    if df_full.empty:
        raise ValueError("[global_orders] After dropna, no rows remain.")

    # 정렬: 'date' 대신 'month'
    try:
        df_full = df_full.sort_values("month").reset_index(drop=True)
    except Exception:
        # 만약 예외 발생 시 정렬 생략
        pass

    y_full = df_full[y_col].values
    exog_full = np.column_stack([df_full[sentiment_col].values, df_full[prev_col].values])

    best_arimax = select_best_arima(y_full, exog=exog_full,
                                    p_list=p_list, d_list=d_list, q_list=q_list)
    best_arima = select_best_arima(y_full, exog=None,
                                   p_list=p_list, d_list=d_list, q_list=q_list)

    order_arimax = best_arimax["order"] if best_arimax["order"] is not None else (0,0,0)
    order_arima  = best_arima["order"]  if best_arima["order"]  is not None else (0,0,0)

    aic_arimax = best_arimax["aic"] if np.isfinite(best_arimax["aic"]) else float("inf")
    aic_arima  = best_arima["aic"]  if np.isfinite(best_arima["aic"])  else float("inf")

    print(f"[global_orders] ARIMAX(best)={order_arimax}, AIC={aic_arimax:.3f} | "
          f"ARIMA(best)={order_arima}, AIC={aic_arima:.3f}")

    return order_arimax, order_arima


# --- 수정된 evaluate_target (글로벌 order 고정) ---
def evaluate_target(dat: pd.DataFrame, target: str, sentiment_col: str,
                    p_list=(0,1,2), d_list=(0,1), q_list=(0,1,2),
                    base_window=30, small_window=12):
    """
    Rolling OOS 평가 (윈도우마다 ARIMA/ARIMAX 차수 재선택):
      - OLS AR(1): y_t = c + φ y_{t-1}
      - OLS ARX(1): y_t = c + φ y_{t-1} + β sentiment_t
      - ARIMAX (window별 order 선택, exog=[sentiment, prev])
      - ARIMA  (window별 order 선택, no exog)

    변경 사항:
      기존: 전체 표본에서 한 번 global order 선택 후 모든 롤링 시점 재적합
      현재: 각 롤링 학습창(tr)에 대해 select_best_arima(...) 호출로
            ARIMAX / ARIMA 각각의 (p,d,q) 최적 AIC order 선택 → 해당 윈도우 예측

    주의:
      - 매 시점 그리드 탐색 → 계산량 증가 (성능 문제 시 p/d/q 축소 권장)
      - 예외 발생 시 fallback으로 prev_ts 사용
    """
    assert target in ("cpi", "anfci", "nfci", "unrate")
    if sentiment_col not in dat.columns:
        raise KeyError(f"[evaluate_target] sentiment_col '{sentiment_col}' missing.")

    y_col_map = {
        "cpi": "cpi_next",
        "anfci": "anfci_next",
        "nfci": "nfci_next",
        "unrate": "unrate_next"
    }
    prev_col_map = {
        "cpi": "cpi_prev",
        "anfci": "anfci_prev",
        "nfci": "nfci_prev",
        "unrate": "unrate_prev"
    }

    y_col   = y_col_map[target]
    prev_col= prev_col_map[target]

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

    rows = []
    per_window_orders = []  # (arimax_order, arima_order) 저장 (디버깅용)

    for te in range(window_size, n):
        tr = df_all.iloc[te - window_size: te]
        ts = df_all.iloc[te:te+1]

        y_tr   = tr[y_col].values.astype(float)
        sent_tr= tr[sentiment_col].values.astype(float)
        prev_tr= tr[prev_col].values.astype(float)

        y_ts    = float(ts[y_col].values[0])
        sent_ts = float(ts[sentiment_col].values[0])
        prev_ts = float(ts[prev_col].values[0])

        # ---------------- OLS AR(1) ----------------
        if len(y_tr) < 2:
            pred_ar_ols  = y_tr[-1]
            pred_arx_ols = y_tr[-1]
        else:
            y_dep = y_tr[1:]
            y_lag = y_tr[:-1]

            # AR(1)
            X_ar = sm.add_constant(y_lag, has_constant="add")
            try:
                res_ar = sm.OLS(y_dep, X_ar).fit()
                c_ar, phi_ar = res_ar.params
            except Exception:
                c_ar, phi_ar = 0.0, 1.0
            y_prev_last = y_tr[-1]
            pred_ar_ols = c_ar + phi_ar * y_prev_last

            # ARX(1) (sentiment 동시)
            sent_curr = sent_tr[1:]
            X_arx = sm.add_constant(np.column_stack([y_lag, sent_curr]), has_constant="add")
            try:
                res_arx = sm.OLS(y_dep, X_arx).fit()
                c_ax, phi_ax, beta_ax = res_arx.params
            except Exception:
                c_ax, phi_ax, beta_ax = c_ar, phi_ar, 0.0
            pred_arx_ols = c_ax + phi_ax * y_prev_last + beta_ax * sent_ts

        # ---------------- Window별 ARIMAX 차수 선택 ----------------
        exog_tr = np.column_stack([sent_tr, prev_tr])
        best_arimax = select_best_arima(y_tr,
                                        exog=exog_tr,
                                        p_list=p_list,
                                        d_list=d_list,
                                        q_list=q_list)
        order_arimax = best_arimax["order"] if best_arimax["order"] else (0,0,0)
        try:
            if best_arimax["res"] is None:
                # 다시 한 번 fit (order_arimax)
                model_arimax = SARIMAX(endog=y_tr, exog=exog_tr, order=order_arimax, trend="c",
                                       enforce_stationarity=False, enforce_invertibility=False)
                res_arimax = model_arimax.fit(disp=False)
            else:
                res_arimax = best_arimax["res"]
            pred_arimax = _first_pred(
                res_arimax.get_forecast(steps=1, exog=np.array([[sent_ts, prev_ts]])).predicted_mean
            )
        except Exception:
            pred_arimax = prev_ts

        # ---------------- Window별 ARIMA 차수 선택 ----------------
        best_arima = select_best_arima(y_tr,
                                       exog=None,
                                       p_list=p_list,
                                       d_list=d_list,
                                       q_list=q_list)
        order_arima = best_arima["order"] if best_arima["order"] else (0,0,0)
        try:
            if best_arima["res"] is None:
                model_arima = SARIMAX(endog=y_tr, exog=None, order=order_arima, trend="c",
                                      enforce_stationarity=False, enforce_invertibility=False)
                res_arima = model_arima.fit(disp=False)
            else:
                res_arima = best_arima["res"]
            pred_arima = _first_pred(
                res_arima.get_forecast(steps=1).predicted_mean
            )
        except Exception:
            pred_arima = prev_ts

        per_window_orders.append((order_arimax, order_arima))

        rows.append({
            "month": ts["month"].values[0],
            "y": y_ts,
            "arx_ols": pred_arx_ols,
            "ar_ols": pred_ar_ols,
            "arima_w": pred_arimax,
            "arima_wo": pred_arima,
            "arimax_order": order_arimax,
            "arima_order": order_arima
        })

    results = pd.DataFrame(rows)

    def _metric_row(col):
        _rmse = rmse(results["y"], results[col])
        _mae  = mae(results["y"], results[col])
        if target in ("cpi", "unrate"):
            _mape = mape(results["y"], results[col])
        else:
            _mape = mape_eps(results["y"], results[col], eps=0.1)
        return _rmse, _mae, _mape

    labels = [
        ("ARX(1) OLS (y_{t-1}+sentiment)", "arx_ols"),
        ("AR(1) OLS (y_{t-1})", "ar_ols"),
        ("ARIMAX (window order, sentiment+prev)", "arima_w"),
        ("ARIMA  (window order, no exog)", "arima_wo"),
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
        "window_size": window_size,
        "per_window_orders": per_window_orders
    }

# ============================================================
# Granger causality (lag 1)
# ============================================================
def _pvalue_linear_granger(y, x, maxlag=1):
    df = pd.DataFrame({"y": y, "x": x}).dropna()
    if len(df) < maxlag + 5:
        return np.nan
    try:
        res = grangercausalitytests(df[["y","x"]].values, maxlag=maxlag, verbose=False) # SSR 기반 F-검정
        return float(res[maxlag][0]["ssr_ftest"][1])
    except Exception:
        return np.nan

def adf_test_on_residuals(
        resid,
        name="best_x residuals",
        alpha=0.05,
        regression="c",   # 잔차가 평균 0에 가깝다면 "nc"도 검토 가능. 기본은 "c"
        autolag="AIC",
        maxlag=None
):
    """
    ADF(단위근) 검정: H0=단위근 존재(비정상), H1=단위근 없음(정상)
    p-value < alpha 이면 '통과(정상)'로 판정
    """
    s = pd.Series(resid).dropna().astype(float).values
    if s.size < 10:
        print(f"[ADF] 표본이 너무 작습니다 (n={s.size}).")
        return None

    stat, pval, usedlag, nobs, crit, icbest = adfuller(
        s, regression=regression, autolag=autolag, maxlag=maxlag
    )
    decision = "통과(정상)" if pval < alpha else "불통과(단위근 기각 실패)"

    print(f"\n=== ADF 단위근 검정: {name} ===")
    print(f"옵션: regression='{regression}', autolag='{autolag}', maxlag={maxlag}")
    print(f"표본 크기 n={s.size}, 사용 시차 used_lag={usedlag}, 유효 관측 nobs={nobs}")
    print(f"ADF 통계량: {stat:.4f}, p-value: {pval:.6f}, ICbest: {icbest:.4f}")
    print("임계값:", ", ".join([f"{k}: {float(v):.4f}" for k, v in crit.items()]))
    print(f"판정 @ alpha={alpha:.2f}: {decision}")

    return {
        "stat": float(stat),
        "pvalue": float(pval),
        "usedlag": int(usedlag),
        "nobs": int(nobs),
        "critical_values": {k: float(v) for k, v in crit.items()},
        "icbest": float(icbest),
        "decision": decision
    }


def _pvalue_arima_granger(y, x, maxlag=1):
    """
    각 시계열에 AIC 최소의 ARIMA를 적합하여 잔차를 만든 뒤(프리화이트닝),
    그 잔차들로 Granger causality(SSR F-test) p-value를 계산한다.
    방향: x -> y (두 번째가 첫 번째를 Granger-원인하는가)
    """
    df = pd.DataFrame({"y": pd.Series(y), "x": pd.Series(x)}).dropna()
    if len(df) < maxlag + 5:
        return np.nan
    try:
        # y, x 각각 ARIMA 선택 및 잔차 생성
        new_y = fit_arima(y, (0,0,0), exog=None)
        best_y = {"aic": 0, "res": new_y, "order": (0,0,0)}
        best_x = select_best_arima(df["x"].values, exog=None)
        adf_test_on_residuals(
            best_x["res"].resid,
            name=f"best_x resid (order={best_x['order']})",
            alpha=0.05,
            regression="c",   # 잔차가 평균 0로 충분히 가깝다면 "nc"로도 시도 가능
            autolag="AIC"
        )
        summarize_residuals(best_x["res"].resid, nlags=15, plot=False, plot_prefix="resid")

        if best_y["res"] is None or best_x["res"] is None:
            return np.nan

        resid_y = pd.Series(best_y["res"].resid, index=df.index)
        resid_x = pd.Series(best_x["res"].resid, index=df.index)
        df_res = pd.DataFrame({"y": resid_y, "x": resid_x}).dropna()

        if len(df_res) < maxlag + 5:
            return np.nan

        res = grangercausalitytests(df_res[["y", "x"]].values, maxlag=maxlag, verbose=False)
        return float(res[maxlag][0]["ssr_ftest"][1])
    except Exception:
        return np.nan

def _pvalue_arima_granger2(y, x, maxlag=1):
    """
    각 시계열에 AIC 최소의 ARIMA를 적합하여 잔차를 만든 뒤(프리화이트닝),
    그 잔차들로 Granger causality(SSR F-test) p-value를 계산한다.
    방향: x -> y (두 번째가 첫 번째를 Granger-원인하는가)
    """
    df = pd.DataFrame({"y": pd.Series(y), "x": pd.Series(x)}).dropna()
    if len(df) < maxlag + 5:
        return np.nan
    try:
        # y, x 각각 ARIMA 선택 및 잔차 생성
        new_x = fit_arima(x, (0,0,0), exog=None)
        best_x = {"aic": 0, "res": new_x, "order": (0,0,0)}
        best_y = select_best_arima(df["y"].values, exog=None)
        adf_test_on_residuals(
            best_y["res"].resid,
            name=f"best_x resid (order={best_y['order']})",
            alpha=0.05,
            regression="c",   # 잔차가 평균 0로 충분히 가깝다면 "nc"로도 시도 가능
            autolag="AIC"
        )
        summarize_residuals(best_y["res"].resid, nlags=10, plot=False, plot_prefix="resid")

        if best_y["res"] is None or best_x["res"] is None:
            return np.nan

        resid_y = pd.Series(best_y["res"].resid, index=df.index)
        resid_x = pd.Series(best_x["res"].resid, index=df.index)
        df_res = pd.DataFrame({"y": resid_y, "x": resid_x}).dropna()

        if len(df_res) < maxlag + 5:
            return np.nan

        res = grangercausalitytests(df_res[["y", "x"]].values, maxlag=maxlag, verbose=False)
        return float(res[maxlag][0]["ssr_ftest"][1])
    except Exception:
        return np.nan

def granger_table(dat: pd.DataFrame, sentiment_col: str):
    required = [
        sentiment_col,
        "cpi_prev","cpi_next",
        "anfci_prev","anfci_next",
        "nfci_prev","nfci_next",
        "unrate_prev","unrate_next"
    ]
    for r in required:
        if r not in dat.columns:
            raise KeyError(f"Missing column {r} for Granger.")
    df = dat[[
        "month", sentiment_col,
        "cpi_prev","cpi_next",
        "anfci_prev","anfci_next",
        "nfci_prev","nfci_next",
        "unrate_prev","unrate_next"
    ]].dropna().sort_values("month").reset_index(drop=True)
    S = df[sentiment_col]

    # 반환값: (Linear Granger p, ARIMA Granger p)
    def both(y, x):
        p_lin = _pvalue_linear_granger(y, x, maxlag=1)
        p_arima = _pvalue_arima_granger(y, x, maxlag=1)
        return p_lin, p_arima

    def both2(y, x):
        p_lin = _pvalue_linear_granger(y, x, maxlag=1)
        p_arima = _pvalue_arima_granger2(y, x, maxlag=1)
        return p_lin, p_arima


    rows = []
    # CPI
    p_lin, p_ar = both2(S, df["cpi_prev"]) # default = both(y,x)
    rows.append({"Model": "Previous Month CPI -> Sentiment Index",
                 "Linear Granger (p-value)": p_lin,
                 "ARIMA Granger (p-value)": p_ar})
    p_lin, p_ar = both(df["cpi_next"], S)
    rows.append({"Model": "Sentiment Index -> Next Month CPI",
                 "Linear Granger (p-value)": p_lin,
                 "ARIMA Granger (p-value)": p_ar})
    # ANFCI
    p_lin, p_ar = both2(S, df["anfci_prev"])
    rows.append({"Model": "Previous Month ANFCI -> Sentiment Index",
                 "Linear Granger (p-value)": p_lin,
                 "ARIMA Granger (p-value)": p_ar})
    p_lin, p_ar = both(df["anfci_next"], S)
    rows.append({"Model": "Sentiment Index -> Next Month ANFCI",
                 "Linear Granger (p-value)": p_lin,
                 "ARIMA Granger (p-value)": p_ar})
    # NFCI
    p_lin, p_ar = both2(S, df["nfci_prev"])
    rows.append({"Model": "Previous Month NFCI -> Sentiment Index",
                 "Linear Granger (p-value)": p_lin,
                 "ARIMA Granger (p-value)": p_ar})
    p_lin, p_ar = both(df["nfci_next"], S)
    rows.append({"Model": "Sentiment Index -> Next Month NFCI",
                 "Linear Granger (p-value)": p_lin,
                 "ARIMA Granger (p-value)": p_ar})
    # Unrate
    p_lin, p_ar = both2(S, df["unrate_prev"])
    rows.append({"Model": "Previous Month Unemployment -> Sentiment Index",
                 "Linear Granger (p-value)": p_lin,
                 "ARIMA Granger (p-value)": p_ar})
    p_lin, p_ar = both(df["unrate_next"], S)
    rows.append({"Model": "Sentiment Index -> Next Month Unemployment",
                 "Linear Granger (p-value)": p_lin,
                 "ARIMA Granger (p-value)": p_ar})

    out = pd.DataFrame(rows)
    for c in ["Linear Granger (p-value)", "ARIMA Granger (p-value)"]:
        out[c] = out[c].round(6)
    return out

# ============================================================
# ARIMA layer comparison table (Table 13)
# ============================================================
def arma_layer_table(dat: pd.DataFrame, verbose_auto_aic: bool = False):
    """
    For each variable: y_next series across full sample.
    1) Best ARIMA on y_t
    2) Residual epsilon_t = y_t - yhat_t
    3) Best ARIMA on epsilon_t
    Summarize differences.
    """
    vars_map = {
        "CPI": "cpi_next",
        "ANFCI": "anfci_next",
        "NFCI": "nfci_next",
        "Unemployment": "unrate_next",
    }
    rows = []
    for name, col in vars_map.items():
        if col not in dat.columns:
            continue
        y = dat[col].dropna().values
        if len(y) < 20:
            continue

        # auto_aic로 AIC 탐색 과정을 보여줌
        best1 = auto_aic(y, exog=None,
                         p_list=(0,1,2),
                         d_list=(0,1),
                         q_list=(0,1,2),
                         name=f"{name} - First ARIMA",
                         print_progress=verbose_auto_aic)
        if best1["res"] is None:
            continue

        yhat = best1["res"].fittedvalues
        eps = y - yhat

        best2 = auto_aic(eps, exog=None,
                         p_list=(0,1,2),
                         d_list=(0,1),
                         q_list=(0,1,2),
                         name=f"{name} - Residual ARIMA",
                         print_progress=verbose_auto_aic)
        order1 = best1["order"]
        order2 = best2["order"] if best2["res"] is not None else None
        rows.append({
            "Variable": name,
            "N": len(y),
            "First_ARIMA(p,d,q)": order1,
            "First_AIC": round(best1["aic"], 3) if np.isfinite(best1["aic"]) else np.nan,
            "Residual_Var": float(np.var(eps, ddof=1)),
            "Original_Var": float(np.var(y, ddof=1)),
            "Variance_Reduction_%": round(100.0 * (1 - np.var(eps, ddof=1) / np.var(y, ddof=1)), 2),
            "Residual_ARIMA(p,d,q)": order2,
            "Residual_AIC": round(best2["aic"], 3) if (best2["res"] is not None and np.isfinite(best2["aic"])) else np.nan,
            "Same_Model?": bool(order1 == order2),
            "Residual_Model_Improves_AIC?": bool(best2["aic"] < best1["aic"]) if (best2["res"] is not None) else False,
            "AIC_Delta_Residual_minus_First": round(best2["aic"] - best1["aic"], 3) if (best2["res"] is not None) else np.nan
        })
    out = pd.DataFrame(rows)
    return out

# ============================================================
# ARIMAX demo using sentiment as exogenous: auto_aic + residual diagnostics
# ============================================================
def arimax_auto_aic_with_sentiment(dat: pd.DataFrame, target: str, sentiment_col: str,
                                   nlags_diag: int = 12, include_prev: bool = False,
                                   p_list=(0,1,2), d_list=(0,1), q_list=(0,1,2),
                                   plot_resid: bool = False):
    assert target in ("cpi", "anfci", "nfci", "unrate")
    y_col = {
        "cpi": "cpi_next",
        "anfci": "anfci_next",
        "nfci": "nfci_next",
        "unrate": "unrate_next"
    }[target]
    prev_col = {
        "cpi": "cpi_prev",
        "anfci": "anfci_prev",
        "nfci": "nfci_prev",
        "unrate": "unrate_prev"
    }[target]
    cols = ["month", y_col, sentiment_col] + ([prev_col] if include_prev else [])
    df = dat[cols].dropna().sort_values("month").reset_index(drop=True)
    y = df[y_col].values
    if include_prev:
        exog = np.column_stack([df[sentiment_col].values, df[prev_col].values])
        exog_name = f"{sentiment_col}+prev"
    else:
        exog = df[sentiment_col].values.reshape(-1, 1)
        exog_name = f"{sentiment_col}"

    print(f"\n=== ARIMAX auto_aic with exogenous [{exog_name}] for target {target.upper()} ===")
    print(f"Sample size after dropna: n={len(y)}")
    best = auto_aic(y, exog=exog,
                    p_list=p_list, d_list=d_list, q_list=q_list,
                    name=f"{target.upper()} ARIMAX exog={exog_name}",
                    print_progress=True)
    if best["res"] is None:
        print("No valid ARIMAX model found.")
        return

    print(f"Best order for {target.upper()} with exog [{exog_name}]: {best['order']} (AIC={best['aic']:.3f})")
    resid = best["res"].resid
    summarize_residuals(resid, nlags=nlags_diag, plot=plot_resid,
                        plot_prefix=f"resid_{target}_exog_{'sentprev' if include_prev else 'sent'}")

# ============================================================
# Data loading
# ============================================================
def load_and_prepare(start : str, end : str):
    nfci_anfci_ev = read_event_nfci_anfci("../data/anfci_nfci.csv")
    cpi_ev        = read_event_cpi("../data/cpi.csv")
    unrate_ev     = read_event_unrate("../data/unrate.csv")
    sent          = read_sentiment("../data/sentiment_13+19.csv")

    dat = (
        nfci_anfci_ev
        .merge(cpi_ev, on="month", how="inner")
        .merge(unrate_ev, on="month", how="inner")
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
    dat["unrate"]  = dat["unrate_next"]

    if len(dat) < 21:
        raise ValueError(f"Need at least 31 event rows after filtering; got {len(dat)}.")
    return dat

# ============================================================
# Table 12 builder
# ============================================================
def build_table_12(dat: pd.DataFrame, sentiment_col: str):
    res_cpi      = evaluate_target(dat, "cpi",      sentiment_col)["metrics"]
    res_anfci    = evaluate_target(dat, "anfci",    sentiment_col)["metrics"]
    res_nfci     = evaluate_target(dat, "nfci",     sentiment_col)["metrics"]
    res_unrate   = evaluate_target(dat, "unrate",   sentiment_col)["metrics"]

    cpi_tbl    = res_cpi.rename(columns={"RMSE":"CPI_RMSE","MAE":"CPI_MAE","MAPE":"CPI_MAPE"})
    anfci_tbl  = res_anfci.rename(columns={"RMSE":"ANFCI_RMSE","MAE":"ANFCI_MAE","MAPE":"ANFCI_MAPE"})
    nfci_tbl   = res_nfci.rename(columns={"RMSE":"NFCI_RMSE","MAE":"NFCI_MAE","MAPE":"NFCI_MAPE"})
    unrate_tbl = res_unrate.rename(columns={"RMSE":"UNRATE_RMSE","MAE":"UNRATE_MAE","MAPE":"UNRATE_MAPE"})

    tbl12 = (
        cpi_tbl
        .merge(anfci_tbl, on="Model", how="inner")
        .merge(nfci_tbl, on="Model", how="inner")
        .merge(unrate_tbl, on="Model", how="inner")
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

    # Table 13 (ARIMA layer comparison) once (independent of sentiment choice)
    print(f"\n=== Table 13: ARIMA Layer Comparison (First vs Residual) {period_label} ===")
    # auto_aic 탐색 과정을 보여주기 위해 verbose_auto_aic=True
    tbl13 = arma_layer_table(dat, verbose_auto_aic=True)
    if not tbl13.empty:
        print(tbl13.to_string(index=False))
    else:
        print("No sufficient data for ARIMA layer comparison.")

    # Demonstrate ARIMAX auto_aic with sentiment exogenous + residual diagnostics
    for s_col in sentiments:
        if s_col not in dat.columns:
            continue
        print(f"\n=== ARIMAX with exogenous [{s_col}] and residual diagnostics ({period_label}) ===")
        for tgt in ["cpi", "anfci", "nfci", "unrate"]:
            arimax_auto_aic_with_sentiment(dat, target=tgt, sentiment_col=s_col,
                                           nlags_diag=12, include_prev=False,
                                           p_list=(0,1,2), d_list=(0,1), q_list=(0,1,2),
                                           plot_resid=False)

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
            "UNRATE_RMSE","UNRATE_MAE","UNRATE_MAPE",
        ]
        print(tbl12[cols_order].to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARIMA/ARIMAX analysis")
    parser.add_argument("--start", type=str, default="2017-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-12-31", help="End date (YYYY-MM-DD)")
    args = parser.parse_args()
    main(args.start, args.end)