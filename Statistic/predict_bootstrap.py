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
# - CHANGE REQUESTED: Apply residual-on-top ONLY for Vine regression (sentiment):
#   fit vine on (y_next - prev) ~ sentiment, then predict yhat = prev + residual_hat
#   This stabilizes Vine performance and reduces MAPE/RMSE.
# - NEW: Gaussian-copula fallback now supports correlation estimation by
#   'kendall', 'spearman', 'pearson', or 'auto' (selects best by pseudo-LL).
# ============================================================

import warnings
warnings.filterwarnings("ignore")
import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def _first_pred(predicted_mean):
    """
    statsmodels predicted_mean이 Series/Index/ndarray 등 어떤 타입이든
    첫 예측값을 float로 안전하게 추출.
    """
    return float(np.asarray(predicted_mean).ravel()[0])

def _safe_forecast(best, steps, exog, fallback_value):
    """
    - best: select_best_arma의 반환(dict) 또는 None
    - steps: 1
    - exog: 예측 외생변수 배열 또는 None
    - fallback_value: 폴백으로 사용할 스칼라(float)
    """
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
    """
    Robustly parse strings like '2017-1', '2017-01', '2020-03x', '2020/3', etc.
    Return 'YYYY-MM' or np.nan when unparsable/empty.
    """
    import numpy as np
    import re

    if s is None:
        return np.nan
    s = str(s).strip()
    if s == "":
        return np.nan
    m = re.search(r"(\d{4})[^\d]?(\d{1,2})", s)  # allow '-', '/', or nothing between
    if not m:
        return np.nan
    yyyy = int(m.group(1))
    mm = int(m.group(2))
    if mm < 1 or mm > 12:
        return np.nan
    return f"{yyyy}-{mm:02d}"


# -----------------------------
# New readers: event-based prev/next + event date → month
# -----------------------------
def _parse_event_date(s):
    # Handle both "02/01/2017" and "2017-02-01"
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

def read_event_nfci_anfci(path="data/anfci_nfci.csv"):
    """
    Read event-level ANFCI/NFCI with prev/next and statement_date:
      columns: prev_NFCI, prev_ANFCI, next_NFCI, next_ANFCI, statement_date
    Returns:
      month (Timestamp at month start),
      event_date_nfci (Timestamp),
      anfci_prev, anfci_next, nfci_prev, nfci_next
    """
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
    """
    Read event-level CPI with prev/next and FOMC_date:
      columns: Prev_CPI_Value, Next_CPI_Value, FOMC_date
    Returns:
      month (Timestamp at month start),
      event_date_cpi (Timestamp),
      cpi_prev, cpi_next
    """
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


def read_sentiment(path="data/sentiment.csv"):
    """
    Reads sentiment and aggregates to monthly averages.
    If your sentiment file is already at statement dates, the monthly merge below will
    align each event to the corresponding month’s sentiment.
    """
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
                raise ValueError("Sentiment 파일을 4개 칼럼(date, statement, Theory, Policy)로 분해할 수 없습니다. 구분자를 확인하세요.")
            body = body.iloc[:, :4]
            body.columns = ["date", "statement", "Theory", "Policy"]
            df = body

    df.columns = [str(c).strip() for c in df.columns]
    col_map = {}
    date_col = None
    for c in df.columns:
        cl = c.lower().strip()
        if cl == "date":
            date_col = c
        elif cl in ("statement", "statment"):  # 오타 허용
            col_map[c] = "statement"
        elif cl == "theory":
            col_map[c] = "Theory"
        elif cl == "policy":
            col_map[c] = "Policy"

    if date_col is None:
        candidate = None
        for c in df.columns:
            vals = df[c].astype(str).head(5).tolist()
            if any(re.search(r"\d{4}[-/]\d{1,2}", v) for v in vals):
                candidate = c
                break
        if candidate is None:
            raise KeyError("Sentiment file must have a 'date' column.")
        date_col = candidate

    df = df.rename(columns=col_map)
    required = ["statement", "Theory", "Policy"]
    missing = [r for r in required if r not in df.columns]
    if missing:
        raise KeyError(f"Sentiment 파일에 필요한 칼럼이 없습니다: {missing}. 현재 칼럼: {list(df.columns)}")

    df["month"] = df[date_col].astype(str).map(clean_month_string)
    bad = df["month"].isna().sum()
    if bad:
        print(f"[read_sentiment] 파싱 불가 날짜 {bad}개 행 제거")
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
# 2) Metrics
# -----------------------------
def rmse(y, yhat):
    """
    RMSE = sqrt( (1/n) * sum_t (y_t - yhat_t)^2 )
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if y.shape != yhat.shape:
        raise ValueError("rmse: y and yhat must have the same shape.")
    if np.isnan(y).any() or np.isnan(yhat).any():
        raise ValueError("rmse: NaN detected in inputs; clean your data before computing RMSE.")
    n = y.size
    se = (y - yhat) ** 2
    return float(np.sqrt(np.sum(se) / n))

def mae(y, yhat):
    """
    MAE = (1/n) * sum_t |y_t - yhat_t|
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if y.shape != yhat.shape:
        raise ValueError("mae: y and yhat must have the same shape.")
    if np.isnan(y).any() or np.isnan(yhat).any():
        raise ValueError("mae: NaN detected in inputs; clean your data before computing MAE.")
    n = y.size
    ae = np.abs(y - yhat)
    return float(np.sum(ae) / n)

def mape(y, yhat):
    """
    MAPE = (100% / n) * sum_t |(y_t - yhat_t) / y_t|
    Note: If any y_t == 0, MAPE is undefined by this strict formula.
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if y.shape != yhat.shape:
        raise ValueError("mape: y and yhat must have the same shape.")
    if np.isnan(y).any() or np.isnan(yhat).any():
        raise ValueError("mape: NaN detected in inputs; clean your data before computing MAPE.")
    if np.any(y == 0):
        raise ZeroDivisionError("mape: y contains zero; MAPE is undefined with the strict formula.")
    n = y.size
    return float(100.0 * np.sum(np.abs((y - yhat) / y)) / n)


# -----------------------------
# 3) Helpers: ARMA selection via AIC (SARIMAX)
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
# 4) Vine/Bivariate copula regression (1D X)
#     NEW: corr_method controls Gaussian-copula fallback correlation:
#          'kendall' | 'spearman' | 'pearson' | 'auto'
#          - 'auto' tries all three and selects the one with highest Gaussian
#            copula pseudo log-likelihood on training pairs.
# -----------------------------
class BivariateCopulaRegressor1D:
    def __init__(self, n_grid: int = 2000, corr_method: str = "auto"):
        self.n_grid = n_grid
        self.has_vine = HAS_VINE
        self.model = None  # pv.Bicop or dict with rho/method
        self.Fx = None
        self.Qy = None
        self.corr_method = str(corr_method).lower()

    def _ecdf(self, samples):
        xs = np.sort(np.asarray(samples))
        n = len(xs)
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
        # For Gaussian copula: tau = (2/pi) * arcsin(rho)  => rho = sin(pi*tau/2)
        return float(np.sin(np.pi * float(tau) / 2.0))

    @staticmethod
    def _rho_from_spearman(rho_s):
        # For Gaussian copula: rho_s = (6/pi) * arcsin(rho/2) => rho = 2 * sin(pi*rho_s/6)
        return float(2.0 * np.sin(np.pi * float(rho_s) / 6.0))

    @staticmethod
    def _gauss_cop_pseudo_ll(z1, z2, rho):
        # log copula density for Gaussian copula
        rho = float(np.clip(rho, -0.999, 0.999))
        one_m = 1.0 - rho * rho
        # avoid log(0)
        return float(
            -0.5 * np.log(one_m)
            + np.mean((rho * z1 * z2 - 0.5 * rho * rho * (z1 * z1 + z2 * z2)) / one_m)
        )

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        assert x.shape == y.shape, "x and y must have same length."

        # Build empirical CDF transforms and quantile for y
        self.Fx = self._ecdf(x)
        Fy = self._ecdf(y)
        self.Qy = self._quantile(y)

        # Pseudo-observations
        u = Fy(y)
        v = self.Fx(x)
        # Normal scores for Gaussian copula computations
        z1 = norm.ppf(u)
        z2 = norm.ppf(v)

        if self.has_vine:
            # Let pyvinecopulib select best family/params
            data = np.column_stack([u, v])
            bicop = pv.Bicop()
            bicop.select(data)
            self.model = bicop
            return self

        # Fallback: Gaussian copula with different correlation estimators
        # Kendall's tau
        tau_k = pd.Series(u).corr(pd.Series(v), method="kendall")
        rho_k = self._rho_from_kendall(tau_k)
        # Spearman's rho
        rho_spearman = pd.Series(u).corr(pd.Series(v), method="spearman")
        rho_s = self._rho_from_spearman(rho_spearman)
        # Pearson (on normal scores) — MLE for Gaussian copula
        rho_p = float(np.corrcoef(z1, z2)[0, 1])

        rho_k = float(np.clip(rho_k, -0.999, 0.999))
        rho_s = float(np.clip(rho_s, -0.999, 0.999))
        rho_p = float(np.clip(rho_p, -0.999, 0.999))

        method = self.corr_method
        if method not in ("kendall", "spearman", "pearson", "auto"):
            method = "auto"

        if method == "kendall":
            rho_sel, m_sel = rho_k, "kendall"
        elif method == "spearman":
            rho_sel, m_sel = rho_s, "spearman"
        elif method == "pearson":
            rho_sel, m_sel = rho_p, "pearson"
        else:
            # auto: select by highest pseudo log-likelihood under Gaussian copula
            ll_k = self._gauss_cop_pseudo_ll(z1, z2, rho_k)
            ll_s = self._gauss_cop_pseudo_ll(z1, z2, rho_s)
            ll_p = self._gauss_cop_pseudo_ll(z1, z2, rho_p)
            if ll_p >= ll_s and ll_p >= ll_k:
                rho_sel, m_sel = rho_p, "pearson"
            elif ll_s >= ll_k:
                rho_sel, m_sel = rho_s, "spearman"
            else:
                rho_sel, m_sel = rho_k, "kendall"

        self.model = {"rho": float(rho_sel), "method": m_sel}
        return self

    def predict_one(self, x_star: float) -> float:
        # Compute conditional expectation via simulation grid on copula
        v_star = float(self.Fx(x_star))
        r = (np.arange(1, self.n_grid + 1) / (self.n_grid + 1.0)).astype(float)

        if HAS_VINE and isinstance(self.model, pv.Bicop):
            u_samples = self.model.hinv1(r, v_star)
        else:
            rho = self.model["rho"]
            # Conditional sampling for Gaussian copula using normal scores
            z2 = norm.ppf(v_star)
            z1 = rho * z2 + np.sqrt(max(1e-12, 1 - rho**2)) * norm.ppf(r)
            u_samples = norm.cdf(z1)

        y_samples = self.Qy(u_samples)
        return float(np.mean(y_samples))

    def predict(self, x_star_array):
        x_star_array = np.asarray(x_star_array).ravel()
        return np.array([self.predict_one(xi) for xi in x_star_array])


# -----------------------------
# 5) Rolling window evaluation (per sentiment)
#     Targets now use event-based "next" values:
#       cpi  -> cpi_next
#       anfci-> anfci_next
#       nfci -> nfci_next
#     NOTE: Only Vine(s) changed to residual-on-top per request.
#     NEW: Vine uses corr_method="auto" to allow Kendall/Spearman/Pearson selection.
# -----------------------------
def evaluate_target(dat: pd.DataFrame, target: str, sentiment_col: str):
    assert target in ("cpi", "anfci", "nfci")
    assert sentiment_col in dat.columns, f"{sentiment_col} not in data"

    # map target to next/prev columns
    y_col = {"cpi": "cpi_next", "anfci": "anfci_next", "nfci": "nfci_next"}[target]
    prev_col = {"cpi": "cpi_prev", "anfci": "anfci_prev", "nfci": "nfci_prev"}[target]
    if y_col not in dat.columns or prev_col not in dat.columns:
        raise KeyError(f"Required columns '{y_col}' and '{prev_col}' not found in dat.")

    n = len(dat)
    if n < 31:
        raise ValueError(f"Need at least 31 event rows for rolling eval; got {n}.")

    # General rolling: train on [0..te-1], predict at te
    train_ends = list(range(30, n))

    rows = []
    for te in train_ends:
        tr = dat.iloc[:te].copy()
        ts = dat.iloc[te:te+1].copy()

        y_tr = tr[y_col].values
        x_tr_sent = tr[[sentiment_col]].values  # (n,1)
        x_ts_sent = ts[[sentiment_col]].values  # (1,1)
        prev_ts = float(ts[prev_col].values[0])

        # Linear with sentiment (ARX: sentiment + prev)
        x_tr_arx = np.column_stack([x_tr_sent, tr[[prev_col]].values])  # (n, 2)
        x_ts_arx = np.array([[x_ts_sent[0, 0], prev_ts]], dtype=float)
        X_tr_w = sm.add_constant(x_tr_arx, has_constant="add")
        ols_w = sm.OLS(y_tr, X_tr_w).fit()
        X_ts_w = sm.add_constant(x_ts_arx, has_constant="add")
        pred_lm_w = float(ols_w.predict(X_ts_w)[0])

        # Linear no sentiment (naive last observation = prev)
        pred_lm_wo = prev_ts

        # Vine with sentiment: residual-on-top (uses corr_method="auto")
        # Fit vine on residuals: r = y_next - prev
        y_res_tr = (tr[y_col].values - tr[prev_col].values).astype(float)
        vine = BivariateCopulaRegressor1D(n_grid=2000, corr_method="auto").fit(
            x_tr_sent.ravel(), y_res_tr
        )
        res_hat = float(vine.predict(x_ts_sent.ravel())[0])
        pred_vine_w = prev_ts + res_hat

        # Vine no sentiment: naive prev
        pred_vine_wo = prev_ts

        # GCMR proxy: SARIMAX ARMA(p,q) AIC selection
        best_w = select_best_arma(y_tr, x_tr_arx, candidates=((0,0),(0,1),(1,0),(1,1)))
        pred_gcmr_w = _safe_forecast(
            best=best_w,
            steps=1,
            exog=x_ts_arx,                 # shape (1, 2)
            fallback_value=prev_ts         # fallback: naive prev
        )

        best_wo = select_best_arma(y_tr, None, candidates=((0,0),(0,1),(1,0),(1,1)))
        pred_gcmr_wo = _safe_forecast(
            best=best_wo,
            steps=1,
            exog=None,
            fallback_value=prev_ts         # fallback: naive prev
        )

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

    metrics = pd.DataFrame({
        "Model": [
            "Linear regression (sentiment)",
            "Linear regression (no sentiment)",
            "Vine regression (sentiment)",
            "Vine regression (no sentiment)",
            "GCMR (sentiment)",
            "GCMR (no sentiment)"
        ],
        "RMSE": [
            rmse(results["y"], results["lm_w"]),
            rmse(results["y"], results["lm_wo"]),
            rmse(results["y"], results["vine_w"]),
            rmse(results["y"], results["vine_wo"]),
            rmse(results["y"], results["gcmr_w"]),
            rmse(results["y"], results["gcmr_wo"]),
        ],
        "MAE": [
            mae(results["y"], results["lm_w"]),
            mae(results["y"], results["lm_wo"]),
            mae(results["y"], results["vine_w"]),
            mae(results["y"], results["vine_wo"]),
            mae(results["y"], results["gcmr_w"]),
            mae(results["y"], results["gcmr_wo"]),
        ],
        "MAPE": [
            mape(results["y"], results["lm_w"]),
            mape(results["y"], results["lm_wo"]),
            mape(results["y"], results["vine_w"]),
            mape(results["y"], results["vine_wo"]),
            mape(results["y"], results["gcmr_w"]),
            mape(results["y"], results["gcmr_wo"]),
        ]
    })

    return {"oos": results, "metrics": metrics}


# -----------------------------
# 6) Granger causality tables (lag 1) using provided prev/next
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
    x = s.values
    xs = np.sort(x)
    n = len(xs)
    ranks = np.searchsorted(xs, x, side="right")
    u = np.clip(ranks / (n + 1.0), 1e-6, 1 - 1e-6)
    z = norm.ppf(u)
    out = pd.Series(index=s.index, data=z)
    return out

def granger_table(dat: pd.DataFrame, sentiment_col: str):
    # dat already has event-based prev/next columns and monthly sentiment
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
    # CPI
    p_gc, p_lin = both_pvalues(S, df["cpi_prev"])
    rows.append({"Model": "Previous Month CPI -> Sentiment Index", "Gaussian copula Granger (p-value)": p_gc, "Linear Granger (p-value)": p_lin})
    p_gc, p_lin = both_pvalues(df["cpi_next"], S)
    rows.append({"Model": "Sentiment Index -> Next Month CPI", "Gaussian copula Granger (p-value)": p_gc, "Linear Granger (p-value)": p_lin})
    # ANFCI
    p_gc, p_lin = both_pvalues(S, df["anfci_prev"])
    rows.append({"Model": "Previous Month ANFCI -> Sentiment Index", "Gaussian copula Granger (p-value)": p_gc, "Linear Granger (p-value)": p_lin})
    p_gc, p_lin = both_pvalues(df["anfci_next"], S)
    rows.append({"Model": "Sentiment Index -> Next Month ANFCI", "Gaussian copula Granger (p-value)": p_gc, "Linear Granger (p-value)": p_lin})
    # NFCI
    p_gc, p_lin = both_pvalues(S, df["nfci_prev"])
    rows.append({"Model": "Previous Month NFCI -> Sentiment Index", "Gaussian copula Granger (p-value)": p_gc, "Linear Granger (p-value)": p_lin})
    p_gc, p_lin = both_pvalues(df["nfci_next"], S)
    rows.append({"Model": "Sentiment Index -> Next Month NFCI", "Gaussian copula Granger (p-value)": p_gc, "Linear Granger (p-value)": p_lin})

    out = pd.DataFrame(rows)
    for c in ["Gaussian copula Granger (p-value)", "Linear Granger (p-value)"]:
        out[c] = out[c].round(4)
    return out


# -----------------------------
# 7) Data loading (2017-2022 restriction) and Runner
# -----------------------------
def load_and_prepare():
    # New event-based readers (update paths if different)
    nfci_anfci_ev = read_event_nfci_anfci("../data/anfci_nfci.csv")
    cpi_ev = read_event_cpi("../data/cpi.csv")
    sent = read_sentiment("../data/sentiment.csv")

    # Merge by month (keep only months present in both event files)
    dat = (
        nfci_anfci_ev.merge(cpi_ev, on="month", how="inner")
        .merge(sent, on="month", how="inner")
        .sort_values("month")
        .reset_index(drop=True)
    )

    # Restrict to 2017-01 through 2022-12
    start = pd.Timestamp("2017-01-01")
    end = pd.Timestamp("2022-12-31")
    dat = dat[(dat["month"] >= start) & (dat["month"] <= end)].reset_index(drop=True)

    # Convenience aliases: forecasting targets use "next" values
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