# ============================================================
# Forecasting CPI, ANFCI, and NFCI with FOMC Sentiment Index (Python)
# - Linear regression (with/without sentiment)
# - Vine/Bivariate copula regression (with/without sentiment)
# - SARIMAX (ARMA errors) as GCMR proxy with AIC-based ARMA selection
# Rolling window evaluation: train 30 → predict next (event-based, variable length)
# Additionally:
# - Produce Table 11-like Granger causality (lag 1) for CPI, ANFCI, NFCI vs Sentiment
# - Produce Table 12-like mean forecasting accuracy tables (CPI, ANFCI, NFCI)
# - Repeat both tables for each of three sentiment indices: statement, Theory, Policy
# - Use data restricted to 2017-01 through 2022-12
# - CHANGE REQUESTED:
#   1) Vine(s) = residual-on-top: fit vine on (y_next - prev) ~ sentiment, predict yhat = prev + residual_hat
#      + add a small kNN smoother fallback/blend so Vine(s) ≠ Linear and stays stable.
#   2) Use adjusted MAPE for ANFCI/NFCI when denominators are too small: MAPE_eps.
#   3) In Gaussian-copula fallback, make Pearson correlation primary but also reflect
#      Kendall and Spearman (weighted blend with Pearson-dominant).
#   4) FIX: Vine regression (no sentiment) should NOT equal the Linear (no sentiment) baseline.
#      We fit a bivariate copula between (prev, y_next) and predict E[y_next | prev_ts].
# ============================================================

import warnings
warnings.filterwarnings("ignore")
import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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

import re
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import norm

# Optional: pyvinecopulib for bivariate copula; fallback to Gaussian copula if missing
try:
    import pyvinecopulib as pv
    HAS_VINE = True
except Exception:
    HAS_VINE = False


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
# Metrics
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

def smape(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    denom = np.abs(y) + np.abs(yhat)
    out = np.zeros_like(denom)
    m = denom > 0
    out[m] = 200.0 * np.abs(y[m] - yhat[m]) / denom[m]
    return float(np.mean(out))

def mape_eps(y, yhat, eps=0.1):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    denom = np.maximum(np.abs(y), eps)
    return float(100.0 * np.mean(np.abs(y - yhat) / denom))


# -----------------------------
# ARMA selection via AIC (SARIMAX)
# -----------------------------
def fit_sarimax(y, x=None, order=(0, 0, 0)):
    model = SARIMAX(
        endog=y,
        exog=x,
        order=order,
        trend="c",
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    res = model.fit(disp=False)
    return res

def select_best_arma(y, x=None, candidates=((0,0),(0,1),(1,0),(1,1))):
    best = {"aic": np.inf, "res": None, "order": None}
    for p, q in candidates:
        try:
            res = fit_sarimax(y, x, order=(p, 0, q))
            aic = res.aic
        except Exception:
            aic = np.inf
            res = None
        if aic < best["aic"]:
            best = {"aic": aic, "res": res, "order": (p, 0, q)}
    return best


# -----------------------------
# Vine/Bivariate copula regression (1D X)
# Pearson-primary Gaussian copula fallback with Kendall & Spearman blended in.
# IMPORTANT: Keep all methods indented inside the class to avoid AttributeError.
# -----------------------------
class BivariateCopulaRegressor1D:
    def __init__(self, n_grid: int = 2000, corr_mode: str = "pearson_primary", corr_weights=None):
        """
        corr_mode:
          - 'pearson_primary' (default): blend of Pearson (primary), Spearman, Kendall
          - 'pearson' | 'spearman' | 'kendall': use single estimator
          - 'auto': choose estimator that maximizes Gaussian copula pseudo-LL
        corr_weights: tuple/list of (w_pearson, w_spearman, w_kendall) for 'pearson_primary'
                      default = (0.6, 0.25, 0.15)
        """
        self.n_grid = n_grid
        self.x_train = None
        self.y_train = None
        self.has_vine = HAS_VINE
        self.model = None
        self.corr_mode = str(corr_mode).lower()
        self.corr_weights = corr_weights if corr_weights is not None else (0.6, 0.25, 0.15)

    def _ecdf(self, samples):
        xs = np.sort(np.asarray(samples)); n = len(xs)
        def cdf(x):
            idx = np.searchsorted(xs, x, side="right")
            return np.clip(idx / (n + 1.0), 1e-6, 1 - 1e-6)
        return np.vectorize(cdf)

    def _quantile(self, samples):
        xs = np.asarray(samples)
        def q(u):
            u = np.clip(u, 1e-6, 1 - 1e-6)
            return np.quantile(xs, u)
        return np.vectorize(q)

    @staticmethod
    def _rho_from_kendall(tau):
        return float(np.sin(np.pi * float(tau) / 2.0))

    @staticmethod
    def _rho_from_spearman(rho_s):
        return float(2.0 * np.sin(np.pi * float(rho_s) / 6.0))

    @staticmethod
    def _gauss_cop_pseudo_ll(z1, z2, rho):
        rho = float(np.clip(rho, -0.999, 0.999))
        one_m = 1.0 - rho * rho
        return float(-0.5 * np.log(one_m) + np.mean((rho * z1 * z2 - 0.5 * rho * rho * (z1 * z1 + z2 * z2)) / one_m))

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        assert x.shape == y.shape
        self.x_train = x
        self.y_train = y
        self.Fx = self._ecdf(x)
        Fy = self._ecdf(y)
        self.Qy = self._quantile(y)
        u = Fy(y)
        v = self.Fx(x)
        z1 = norm.ppf(u)
        z2 = norm.ppf(v)
        if self.has_vine:
            data = np.column_stack([u, v])
            bicop = pv.Bicop()
            bicop.select(data)
            self.model = bicop
            return self
        tau_k = pd.Series(u).corr(pd.Series(v), method="kendall")
        rho_k = self._rho_from_kendall(tau_k)
        rho_s_raw = pd.Series(u).corr(pd.Series(v), method="spearman")
        rho_s = self._rho_from_spearman(rho_s_raw)
        rho_p = float(np.corrcoef(z1, z2)[0, 1])
        rho_k = float(np.clip(rho_k, -0.999, 0.999))
        rho_s = float(np.clip(rho_s, -0.999, 0.999))
        rho_p = float(np.clip(rho_p, -0.999, 0.999))
        mode = self.corr_mode
        if mode == "pearson":
            rho = rho_p; chosen = "pearson"
        elif mode == "spearman":
            rho = rho_s; chosen = "spearman"
        elif mode == "kendall":
            rho = rho_k; chosen = "kendall"
        elif mode == "auto":
            ll_k = self._gauss_cop_pseudo_ll(z1, z2, rho_k)
            ll_s = self._gauss_cop_pseudo_ll(z1, z2, rho_s)
            ll_p = self._gauss_cop_pseudo_ll(z1, z2, rho_p)
            if ll_p >= ll_s and ll_p >= ll_k:
                rho, chosen = rho_p, "pearson"
            elif ll_s >= ll_k:
                rho, chosen = rho_s, "spearman"
            else:
                rho, chosen = rho_k, "kendall"
        else:
            w_p, w_s, w_k = self.corr_weights
            w_sum = float(w_p + w_s + w_k) if (w_p + w_s + w_k) != 0 else 1.0
            w_p, w_s, w_k = w_p / w_sum, w_s / w_sum, w_k / w_sum
            rho = float(np.clip(w_p * rho_p + w_s * rho_s + w_k * rho_k, -0.999, 0.999))
            chosen = "pearson_primary_blend"
        self.model = {"rho": rho, "rho_p": rho_p, "rho_s": rho_s, "rho_k": rho_k, "mode": chosen, "weights": (self.corr_weights if mode not in ("pearson","spearman","kendall","auto") else None)}
        return self

    def predict_one(self, x_star: float) -> float:
        v_star = float(self.Fx(x_star))
        r = (np.arange(1, self.n_grid + 1) / (self.n_grid + 1.0)).astype(float)
        if self.has_vine and isinstance(self.model, pv.Bicop):
            u_samples = self.model.hinv1(r, v_star)
        else:
            rho = self.model["rho"]
            z2 = norm.ppf(v_star)
            z1 = rho * z2 + np.sqrt(max(1e-12, 1 - rho**2)) * norm.ppf(r)
            u_samples = norm.cdf(z1)
        y_samples = self.Qy(u_samples)
        return float(np.mean(y_samples))

    def predict(self, x_star_array):
        x_star_array = np.asarray(x_star_array).ravel()
        return np.array([self.predict_one(xi) for xi in x_star_array])


# -----------------------------
# Rolling window evaluation (per sentiment)
# -----------------------------
def _knn_residual(x_tr, y_res_tr, x_star, k=5):
    x_tr = np.asarray(x_tr, float).ravel()
    y_res_tr = np.asarray(y_res_tr, float).ravel()
    x_star = float(x_star)
    if len(x_tr) == 0:
        return 0.0
    k = int(max(1, min(k, len(x_tr))))
    dist = np.abs(x_tr - x_star)
    idx = np.argsort(dist)[:k]
    w = 1.0 / (dist[idx] + 1e-6)
    w = w / w.sum()
    return float(np.dot(w, y_res_tr[idx]))

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

    # NEW: keep counts of selected orders for summary logging
    selection_counts = {"with_exog": {}, "no_exog": {}}

    rows = []
    for te in train_ends:
        tr = dat.iloc[:te].copy()
        ts = dat.iloc[te:te+1].copy()
        y_tr = tr[y_col].values
        x_tr_sent = tr[[sentiment_col]].values
        x_ts_sent = ts[[sentiment_col]].values
        prev_ts = float(ts[prev_col].values[0])
        # Linear with sentiment (ARX: sentiment + prev)
        x_tr_arx = np.column_stack([x_tr_sent, tr[[prev_col]].values])
        x_ts_arx = np.array([[x_ts_sent[0, 0], prev_ts]], dtype=float)
        X_tr_w = sm.add_constant(x_tr_arx, has_constant="add")
        ols_w = sm.OLS(y_tr, X_tr_w).fit()
        X_ts_w = sm.add_constant(x_ts_arx, has_constant="add")
        pred_lm_w = float(ols_w.predict(X_ts_w)[0])
        # Linear no sentiment (naive prev)
        pred_lm_wo = prev_ts
        # Vine with sentiment: residual-on-top + kNN blend
        y_res_tr = (tr[y_col].values - tr[prev_col].values).astype(float)
        vine_resid = BivariateCopulaRegressor1D(
            n_grid=2000, corr_mode="pearson_primary", corr_weights=(0.6, 0.25, 0.15)
        ).fit(x_tr_sent.ravel(), y_res_tr)
        res_hat_cop = float(vine_resid.predict(x_ts_sent.ravel())[0])
        res_hat_knn = _knn_residual(x_tr_sent.ravel(), y_res_tr, x_ts_sent.ravel()[0], k=min(7, max(3, te // 10)))
        try:
            corr = float(pd.Series(x_tr_sent.ravel()).corr(pd.Series(y_res_tr), method="pearson"))
            w = 0.7 if abs(corr) >= 0.1 else 0.3
        except Exception:
            w = 0.5
        res_hat = w * res_hat_cop + (1.0 - w) * res_hat_knn
        res_hat = float(np.clip(res_hat, float(np.min(y_res_tr)), float(np.max(y_res_tr))))
        pred_vine_w = prev_ts + res_hat
        # Vine no sentiment: model E[y_next | prev]
        try:
            x_tr_prev = tr[[prev_col]].values.ravel().astype(float)
            vine_prev = BivariateCopulaRegressor1D(
                n_grid=2000, corr_mode="pearson_primary", corr_weights=(0.6, 0.25, 0.15)
            ).fit(x_tr_prev, y_tr.astype(float))
            pred_vine_wo = float(vine_prev.predict([prev_ts])[0])
        except Exception:
            pred_vine_wo = prev_ts
        # GCMR proxy: SARIMAX ARMA(p,q) AIC selection
        best_w = select_best_arma(y_tr, x_tr_arx, candidates=((0,0),(0,1),(1,0),(1,1)))
        best_wo = select_best_arma(y_tr, None, candidates=((0,0),(0,1),(1,0),(1,1)))
        pred_gcmr_w = _safe_forecast(best=best_w, steps=1, exog=x_ts_arx, fallback_value=prev_ts)
        pred_gcmr_wo = _safe_forecast(best=best_wo, steps=1, exog=None, fallback_value=prev_ts)

        # NEW: print which ARMA model got selected at this step
        def _fmt_order(o):
            return str(o) if o is not None else "None"
        def _fmt_aic(a):
            try:
                a = float(a)
                return f"{a:.3f}" if np.isfinite(a) else "inf"
            except Exception:
                return "nan"
        month_val = ts["month"].values[0]
        try:
            month_str = pd.to_datetime(month_val).strftime("%Y-%m")
        except Exception:
            month_str = str(month_val)

        print(
            f"[ARMA-Selection] target={target} sentiment={sentiment_col} month={month_str} | "
            f"with_exog order={_fmt_order(best_w.get('order'))} AIC={_fmt_aic(best_w.get('aic'))} ; "
            f"no_exog order={_fmt_order(best_wo.get('order'))} AIC={_fmt_aic(best_wo.get('aic'))}"
        )

        # NEW: accumulate counts for a brief summary
        sel_w = best_w.get("order")
        sel_wo = best_wo.get("order")
        selection_counts["with_exog"][sel_w] = selection_counts["with_exog"].get(sel_w, 0) + 1
        selection_counts["no_exog"][sel_wo] = selection_counts["no_exog"].get(sel_wo, 0) + 1

        rows.append({
            "month": ts["month"].values[0],
            "y": ts[y_col].values[0],
            "lm_w": pred_lm_w,
            "lm_wo": pred_lm_wo,
            "vine_w": pred_vine_w,
            "vine_wo": pred_vine_wo,
            "gcmr_w": pred_gcmr_w,
            "gcmr_wo": pred_gcmr_wo,
        })
    results = pd.DataFrame(rows)
    # NEW: print summary counts for this target
    def _summary_line(side, d):
        # ensure consistent order display
        order_list = [ (0,0,0), (1,0,0), (0,0,1), (1,0,1) ]
        # our select_best_arma stores (p,0,q)
        pretty = []
        for o in [(0,0,0),(1,0,0),(0,0,1),(1,0,1)]:
            pretty.append(f"{o}: {d.get(o,0)}")
        others = {k:v for k,v in d.items() if k not in [(0,0,0),(1,0,0),(0,0,1),(1,0,1)]}
        if others:
            pretty.append(f"others: {others}")
        return f"{side} -> " + ", ".join(pretty)
    print(f"[ARMA-Selection-Summary] target={target} sentiment={sentiment_col} | "
          f"{_summary_line('with_exog', selection_counts['with_exog'])} ; "
          f"{_summary_line('no_exog', selection_counts['no_exog'])}")

    def _metric_row(pred_col):
        _rmse = rmse(results["y"], results[pred_col])
        _mae = mae(results["y"], results[pred_col])
        if target == "cpi":
            _mape = mape(results["y"], results[pred_col])
        else:
            _mape = mape_eps(results["y"], results[pred_col], eps=0.1)
        return _rmse, _mae, _mape
    labels = [
        ("Linear regression (sentiment)", "lm_w"),
        ("Linear regression (no sentiment)", "lm_wo"),
        ("Vine regression (sentiment)", "vine_w"),
        ("Vine regression (no sentiment)", "vine_wo"),
        ("GCMR (sentiment)", "gcmr_w"),
        ("GCMR (no sentiment)", "gcmr_wo"),
    ]
    metrics = pd.DataFrame([
        {"Model": name, "RMSE": _metric_row(col)[0], "MAE": _metric_row(col)[1], "MAPE": _metric_row(col)[2]}
        for name, col in labels
    ])
    return {"oos": results, "metrics": metrics}


# -----------------------------
# Granger causality tables (lag 1)
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

def _gaussian_copula_transform(s):
    s = pd.Series(s).dropna()
    x = s.values; xs = np.sort(x); n = len(xs)
    ranks = np.searchsorted(xs, x, side="right")
    u = np.clip(ranks / (n + 1.0), 1e-6, 1 - 1e-6)
    z = norm.ppf(u)
    out = pd.Series(index=s.index, data=z)
    return out

def granger_table(dat: pd.DataFrame, sentiment_col: str):
    required = [sentiment_col, "cpi_prev", "cpi_next", "anfci_prev", "anfci_next", "nfci_prev", "nfci_next"]
    for r in required:
        if r not in dat.columns:
            raise KeyError(f"Required column '{r}' not found for Granger table.")
    df = dat[["month", sentiment_col, "cpi_prev", "cpi_next", "anfci_prev", "anfci_next", "nfci_prev", "nfci_next"]].copy()
    df = df.dropna().sort_values("month").reset_index(drop=True)
    S = df[sentiment_col]
    def both_pvalues(y, x):
        p_lin = _pvalue_linear_granger(y, x, maxlag=1)
        y_z = _gaussian_copula_transform(y).reindex_like(pd.Series(y))
        x_z = _gaussian_copula_transform(x).reindex_like(pd.Series(x))
        tmp = pd.concat([y_z, x_z], axis=1).dropna()
        p_gc = _pvalue_linear_granger(tmp.iloc[:,0], tmp.iloc[:,1], maxlag=1) if len(tmp) else np.nan
        return p_gc, p_lin
    rows = []
    p_gc, p_lin = both_pvalues(S, df["cpi_prev"]) # S = dependent variable y , CPi_prev = regressor x
    rows.append({"Model": "Previous Month CPI -> Sentiment Index", "Gaussian copula Granger (p-value)": p_gc, "Linear Granger (p-value)": p_lin})
    p_gc, p_lin = both_pvalues(df["cpi_next"], S)
    rows.append({"Model": "Sentiment Index -> Next Month CPI", "Gaussian copula Granger (p-value)": p_gc, "Linear Granger (p-value)": p_lin})
    p_gc, p_lin = both_pvalues(S, df["anfci_prev"])
    rows.append({"Model": "Previous Month ANFCI -> Sentiment Index", "Gaussian copula Granger (p-value)": p_gc, "Linear Granger (p-value)": p_lin})
    p_gc, p_lin = both_pvalues(df["anfci_next"], S)
    rows.append({"Model": "Sentiment Index -> Next Month ANFCI", "Gaussian copula Granger (p-value)": p_gc, "Linear Granger (p-value)": p_lin})
    p_gc, p_lin = both_pvalues(S, df["nfci_prev"])
    rows.append({"Model": "Previous Month NFCI -> Sentiment Index", "Gaussian copula Granger (p-value)": p_gc, "Linear Granger (p-value)": p_lin})
    p_gc, p_lin = both_pvalues(df["nfci_next"], S)
    rows.append({"Model": "Sentiment Index -> Next Month NFCI", "Gaussian copula Granger (p-value)": p_gc, "Linear Granger (p-value)": p_lin})
    out = pd.DataFrame(rows)
    for c in ["Gaussian copula Granger (p-value)", "Linear Granger (p-value)"]:
        out[c] = out[c].round(4)
    return out


# -----------------------------
# Data loading (2017-2022 restriction) and Runner
# -----------------------------
def load_and_prepare():
    nfci_anfci_ev = read_event_nfci_anfci("../data/anfci_nfci.csv")
    cpi_ev = read_event_cpi("../data/cpi.csv")
    sent = read_sentiment("../data/sentiment.csv")
    dat = (
        nfci_anfci_ev.merge(cpi_ev, on="month", how="inner")
        .merge(sent, on="month", how="inner")
        .sort_values("month")
        .reset_index(drop=True)
    )
    start = pd.Timestamp("2017-01-01")
    end = pd.Timestamp("2022-12-31")
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
        print(f"\n=== Table 11: Granger causality (lag 1, event-based prev/next) using sentiment: {s_col} (2017-2022) ===")
        tbl11 = granger_table(dat, sentiment_col=s_col)
        print(tbl11.to_string(index=False))
        print(f"\n=== Table 12: Mean forecasting accuracy (event-based next targets) using sentiment: {s_col} (2017-2022) ===")
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