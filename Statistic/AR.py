
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
def rmse(y, yhat):
    return float(np.sqrt(mean_squared_error(y, yhat)))

def mae(y, yhat):
    return float(mean_absolute_error(y, yhat))

def mape(y, yhat, eps=1e-6):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    denom = np.clip(np.abs(y), eps, None)
    return float(100.0 * np.mean(np.abs(y - yhat) / denom))

# ------------------------------------------------------------
# Synthetic data generator (forces corr >= target_corr)
# ------------------------------------------------------------
def generate_data(n=220, seed=42, target_corr=0.7):
    """
    y_t: piecewise smooth trend + strong AR(1) + noise
    x_t: constructed to have Pearson correlation >= target_corr with y.
         x shares the same trend + blended y + small independent noise.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    up_part = 0.02 * t
    down_part = 0.02 * (n - t)
    trend = np.where(t < n // 2, up_part, down_part)

    phi = 0.85
    noise = rng.normal(0, 0.6, size=n)
    # Variance spike around midpoint
    spike_idx = (t > n//2 - 10) & (t < n//2 + 10)
    noise[spike_idx] += rng.normal(0, 1.5, spike_idx.sum())

    y = np.zeros(n)
    for i in range(1, n):
        y[i] = phi * y[i-1] + (1 - phi) * trend[i] + noise[i]

    x_noise = rng.normal(0, 0.25, size=n)
    x = 0.5 * trend + 0.5 * (y - y.mean()) + 0.1 * x_noise

    # Adjust scaling iteratively if correlation below target
    corr = np.corrcoef(y, x)[0, 1]
    attempts = 0
    while corr < target_corr and attempts < 20:
        x = 0.6 * x + 0.4 * (y - y.mean())
        corr = np.corrcoef(y, x)[0, 1]
        attempts += 1

    df = pd.DataFrame({"y": y, "x": x})
    return df

# ------------------------------------------------------------
# Rolling forecast comparison AR vs ARX using SARIMAX (MLE)
# ------------------------------------------------------------
def rolling_forecast_mle(df, window=80, horizon=1, use_diff=False,
                         order_ar=(1,0,0), order_arx=(1,0,0)):
    """
    MLE-based rolling one-step-ahead forecasts using SARIMAX.

    Parameters
    ----------
    df : DataFrame with columns y, x
    window : training window length
    horizon : forecast horizon (only horizon=1 is used here)
    use_diff : if True, difference y and x first, fit AR / ARX on diffs and integrate
    order_ar : (p,d,q) for AR model
    order_arx: (p,d,q) for ARX model

    Returns
    -------
    DataFrame: idx, y_true, y_hat_ar, y_hat_arx
    """
    y_all = df["y"].values
    x_all = df["x"].values
    n = len(df)

    preds_ar = []
    preds_arx = []
    actuals = []
    idxs = []

    # Adjust window if too large
    if window >= n - horizon - 1:
        window = max(5, (n - horizon) // 2)

    for t in range(window, n - horizon + 1):
        y_tr = y_all[t - window: t]
        x_tr = x_all[t - window: t]
        y_true = y_all[t + horizon - 1]

        if use_diff:
            # Differenced approach
            y_tr_diff = np.diff(y_tr)
            x_tr_diff = np.diff(x_tr)

            # Fit AR on differenced series
            if len(y_tr_diff) < 3:  # too few points
                pred_ar = y_tr[-1]
            else:
                try:
                    ar_model = SARIMAX(endog=y_tr_diff,
                                       exog=None,
                                       order=order_ar,
                                       trend="c",
                                       enforce_stationarity=False,
                                       enforce_invertibility=False).fit(disp=False)
                    diff_fore = float(ar_model.get_forecast(steps=horizon).predicted_mean[-1])
                    pred_ar = y_tr[-1] + diff_fore
                except Exception:
                    pred_ar = y_tr[-1]

            # Fit ARX on differenced series
            if len(y_tr_diff) < 3:
                pred_arx = y_tr[-1]
            else:
                try:
                    arx_model = SARIMAX(endog=y_tr_diff,
                                        exog=x_tr_diff.reshape(-1,1),
                                        order=order_arx,
                                        trend="c",
                                        enforce_stationarity=False,
                                        enforce_invertibility=False).fit(disp=False)
                    # use last exog diff for next step (naive)
                    last_xd = x_tr_diff[-1]
                    diff_fore_x = float(arx_model.get_forecast(
                        steps=horizon,
                        exog=np.array([[last_xd]])
                    ).predicted_mean[-1])
                    pred_arx = y_tr[-1] + diff_fore_x
                except Exception:
                    pred_arx = y_tr[-1]

        else:
            # Level modeling
            # AR(1) MLE
            try:
                ar_model = SARIMAX(endog=y_tr,
                                   exog=None,
                                   order=order_ar,
                                   trend="c",
                                   enforce_stationarity=False,
                                   enforce_invertibility=False).fit(disp=False)
                ar_fore = float(ar_model.get_forecast(steps=horizon).predicted_mean[-1])
                pred_ar = ar_fore
            except Exception:
                pred_ar = y_tr[-1]

            # ARX(1) MLE with contemporaneous x
            try:
                arx_model = SARIMAX(endog=y_tr,
                                    exog=x_tr.reshape(-1,1),
                                    order=order_arx,
                                    trend="c",
                                    enforce_stationarity=False,
                                    enforce_invertibility=False).fit(disp=False)
                x_future = x_all[t : t + horizon].reshape(-1,1)  # horizon=1 → shape (1,1)
                arx_fore = float(arx_model.get_forecast(steps=horizon, exog=x_future).predicted_mean[-1])
                pred_arx = arx_fore
            except Exception:
                pred_arx = y_tr[-1]

        preds_ar.append(float(pred_ar))
        preds_arx.append(float(pred_arx))
        actuals.append(float(y_true))
        idxs.append(t + horizon - 1)

    out = pd.DataFrame({
        "idx": idxs,
        "y_true": actuals,
        "y_hat_ar": preds_ar,
        "y_hat_arx": preds_arx
    })
    return out

# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
def plot_error_bars(res, max_points=60):
    err_ar = np.abs(res["y_true"] - res["y_hat_ar"])
    err_arx = np.abs(res["y_true"] - res["y_hat_arx"])
    plot_df = res.copy()
    plot_df["err_ar"] = err_ar
    plot_df["err_arx"] = err_arx
    if len(plot_df) > max_points:
        plot_df = plot_df.iloc[:max_points]

    x = np.arange(len(plot_df))
    width = 0.4

    plt.figure(figsize=(13,5))
    plt.bar(x - width/2, plot_df["err_ar"], width=width, label="|Error| AR MLE")
    plt.bar(x + width/2, plot_df["err_arx"], width=width, label="|Error| ARX MLE")
    plt.title(f"Per-point Absolute Forecast Errors (first {len(plot_df)} points)")
    plt.xlabel("Forecast point (rolling index order)")
    plt.ylabel("Absolute Error")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# Run experiment
# ------------------------------------------------------------
def main():
    df = generate_data(n=220, seed=123, target_corr=0.7)
    corr = np.corrcoef(df["y"], df["x"])[0,1]
    print(f"Global Pearson corr(y,x) = {corr:.3f} (target >= 0.70)")

    # Use SARIMAX rolling
    res = rolling_forecast_mle(df,
                               window=80,
                               horizon=1,
                               use_diff=False,
                               order_ar=(1,0,0),
                               order_arx=(1,0,0))

    ar_rmse = rmse(res["y_true"], res["y_hat_ar"])
    arx_rmse = rmse(res["y_true"], res["y_hat_arx"])
    ar_mae  = mae(res["y_true"], res["y_hat_ar"])
    arx_mae = mae(res["y_true"], res["y_hat_arx"])
    ar_mape = mape(res["y_true"], res["y_hat_ar"])
    arx_mape= mape(res["y_true"], res["y_hat_arx"])

    metrics_df = pd.DataFrame({
        "Model": ["AR_MLE", "ARX_MLE"],
        "RMSE": [ar_rmse, arx_rmse],
        "MAE":  [ar_mae, arx_mae],
        "MAPE": [ar_mape, arx_mape]
    })
    print("\nRolling OOS Metrics (window=80):")
    print(metrics_df.to_string(index=False))

    # Plot 1: Actual vs Forecasts
    plt.figure(figsize=(11,5))
    plt.plot(res["idx"], res["y_true"], label="Actual y", color="black", linewidth=2)
    plt.plot(res["idx"], res["y_hat_ar"], label="AR(1) MLE forecast", alpha=0.85)
    plt.plot(res["idx"], res["y_hat_arx"], label="ARX(1) MLE forecast (with exog)", alpha=0.85)
    plt.title("Actual vs Forecasts (AR MLE vs ARX MLE)\nExogenous highly correlated (corr ≥ 0.7) but limited incremental value")
    plt.xlabel("Time index")
    plt.ylabel("y")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot 2: Error time series
    err_ar  = res["y_true"] - res["y_hat_ar"]
    err_arx = res["y_true"] - res["y_hat_arx"]
    plt.figure(figsize=(11,5))
    plt.plot(res["idx"], err_ar, label="Error AR MLE", alpha=0.85)
    plt.plot(res["idx"], err_arx, label="Error ARX MLE", alpha=0.85)
    plt.title("Forecast Errors Over Time (MLE)")
    plt.xlabel("Time index")
    plt.ylabel("Error (y_true - y_hat)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot 3: Scatter y vs x
    plt.figure(figsize=(6,5))
    plt.scatter(df["x"], df["y"], s=20, alpha=0.6)
    plt.title(f"Scatter of y vs x (Pearson corr ≈ {corr:.2f})")
    plt.xlabel("x (exogenous)")
    plt.ylabel("y")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot 4: Error bars
    plot_error_bars(res, max_points=60)

    # Interpretation
    print("\nInterpretation:")
    print("- Switching from OLS to SARIMAX(MLE) enables expansion to richer (p,d,q) if desired.")
    print("- Identical or near-identical performance to AR OLS indicates exogenous adds little conditional value.")
    print("- Consider testing sentiment lags or differencing if ARX continues to match AR.")
    print("- For more complexity, expand order_ar / order_arx grids and compare AIC / OOS errors.")

if __name__ == "__main__":
    main()