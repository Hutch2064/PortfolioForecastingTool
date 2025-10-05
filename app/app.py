import sys
import warnings
import random
import os
import numpy as np
import pandas as pd
import yfinance as yf
from typing import List
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import shap
import streamlit as st
from sklearn.metrics import mean_squared_error
import optuna
from scipy.stats import t as student_t  # <-- Student-t distribution

warnings.filterwarnings("ignore")

# ---------- Global Seed Fix ----------
GLOBAL_SEED = 42
os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# ---------- Config ----------
DEFAULT_START = "2000-01-01"
ENSEMBLE_SEEDS = 100        # number of seeds in the ensemble
SIMS_PER_SEED = 10000       # simulations per seed
FORECAST_YEARS = 1          # 12-month horizon

# ---------- Helpers ----------
def to_weights(raw: List[float]) -> np.ndarray:
    arr = np.array(raw, dtype=np.float32)
    s = arr.sum()
    if s <= 0:
        raise ValueError("Weights must sum to a positive number.")
    return arr / s

def annualized_return_monthly(monthly_returns: pd.Series) -> float:
    m = monthly_returns.dropna()
    if m.empty: return np.nan
    compounded = (1 + m).prod()
    years = len(m) / 12.0
    return compounded ** (1 / years) - 1 if years > 0 else np.nan

def annualized_vol_monthly(monthly_returns: pd.Series) -> float:
    m = monthly_returns.dropna()
    return m.std(ddof=0) * np.sqrt(12) if len(m) > 1 else np.nan

def annualized_sharpe_monthly(monthly_returns: pd.Series, rf_monthly: float = 0.0) -> float:
    m = monthly_returns.dropna()
    if m.empty: return np.nan
    excess = m - rf_monthly
    mu, sigma = excess.mean(), excess.std(ddof=0)
    return (mu / sigma) * np.sqrt(12) if sigma and sigma > 0 else np.nan

def max_drawdown_from_rets(returns: pd.Series) -> float:
    cum = (1 + returns.fillna(0)).cumprod()
    roll_max = cum.cummax()
    dd = cum / roll_max - 1.0
    return dd.min()

def compute_current_drawdown(returns: pd.Series) -> pd.Series:
    cum = (1 + returns.fillna(0)).cumprod()
    roll_max = cum.cummax()
    return (cum / roll_max - 1.0).astype(np.float32)

# ---------- Data Fetch ----------
def fetch_prices_monthly(tickers: List[str], start=DEFAULT_START) -> pd.DataFrame:
    data = yf.download(
        tickers,
        start=start,
        auto_adjust=False,
        progress=False,
        interval="1mo",
        threads=False
    )
    if data.empty:
        raise ValueError("No price data returned from Yahoo Finance.")
    if isinstance(data.columns, pd.MultiIndex):
        for field in ["Adj Close", "Close"]:
            if field in data.columns.get_level_values(0):
                close = data[field].copy()
                break
        else:
            raise ValueError("Could not find Close/Adj Close in Yahoo data.")
    else:
        colname = "Adj Close" if "Adj Close" in data.columns else "Close"
        close = pd.DataFrame(data[colname]); close.columns = tickers
    close = close.ffill().dropna(how="all").astype(np.float32)
    first_valids = [close[col].first_valid_index() for col in close.columns]
    valid_starts = [d for d in first_valids if d is not None]
    if not valid_starts:
        raise ValueError("No valid price history found for tickers.")
    non_na_start = max(valid_starts)
    return close.loc[non_na_start:]

def fetch_macro_features(start=DEFAULT_START) -> pd.DataFrame:
    tickers = ["^VIX", "^MOVE", "^TNX", "^IRX"]
    data = yf.download(
        tickers, start=start, auto_adjust=False, progress=False, interval="1mo", threads=False
    )
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        close = data.copy()
    close = close.ffill().astype(np.float32)
    df = pd.DataFrame(index=close.index)
    df["VIX"] = close["^VIX"]
    df["MOVE"] = close["^MOVE"]
    df["YC_Spread"] = close["^TNX"] - close["^IRX"]
    return df

# ---------- Portfolio ----------
def portfolio_returns_monthly(prices: pd.DataFrame, weights: np.ndarray, rebalance: str) -> pd.Series:
    rets = prices.pct_change().dropna(how="all").astype(np.float32)
    if rebalance == "N":
        vals = (1 + rets).cumprod()
        port_vals = vals.dot(weights)
        port_vals = port_vals / port_vals.iloc[0]
        return port_vals.pct_change().fillna(0.0).astype(np.float32)
    else:
        freq_map = {"M": "M", "Q": "Q", "S": "2Q", "Y": "A"}
        rule = freq_map.get(rebalance)
        if rule is None: raise ValueError("Invalid rebalance option")
        port_val, port_vals, current_weights = 1.0, [], weights.copy()
        rebalance_dates = rets.resample(rule).last().index
        for i, date in enumerate(rets.index):
            if i > 0: port_val *= (1 + (rets.iloc[i] @ current_weights))
            port_vals.append(port_val)
            if date in rebalance_dates: current_weights = weights.copy()
        return pd.Series(port_vals, index=rets.index, name="Portfolio").pct_change().fillna(0.0).astype(np.float32)

# ---------- Feature Builders ----------
def build_features(returns: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame()
    df["mom_3m"] = returns.rolling(3).apply(lambda x: (1+x).prod()-1, raw=True)
    df["mom_6m"] = returns.rolling(6).apply(lambda x: (1+x).prod()-1, raw=True)
    df["mom_12m"] = returns.rolling(12).apply(lambda x: (1+x).prod()-1, raw=True)
    df["dd_state"] = compute_current_drawdown(returns)

    macro = fetch_macro_features()
    df = df.join(macro, how="left").ffill()

    valid_start = max([df[c].first_valid_index() for c in df.columns if df[c].first_valid_index() is not None])
    df = df.loc[valid_start:].dropna()
    return df.astype(np.float32)

# ---------- OOS Helper Functions ----------
def _oos_years_available(idx: pd.DatetimeIndex, max_years=5) -> List[int]:
    years = sorted(set(idx.year))
    complete_years = []
    for y in years:
        months = set(idx[idx.year == y].month)
        if all(m in months for m in range(1, 13)):
            complete_years.append(y)
    return complete_years[-max_years:] if len(complete_years) > max_years else complete_years

def _split_train_test_for_year(X: pd.DataFrame, Y: pd.Series, test_year: int):
    train_end = pd.Timestamp(f"{test_year-1}-12-31")
    test_start = pd.Timestamp(f"{test_year}-01-01")
    test_end = pd.Timestamp(f"{test_year}-12-31")
    train_X = X.loc[:train_end]
    train_Y = Y.loc[train_X.index]
    test_X = X.loc[test_start:test_end]
    test_Y = Y.loc[test_X.index]
    return train_X, train_Y, test_X, test_Y

def _tune_on_explicit_split(train_X, train_Y, test_X, test_Y, seed=GLOBAL_SEED, n_trials=50):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 1000, 8000),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "df": trial.suggest_int("df", 3, 30),
            "random_state": seed,
            "n_jobs": 1
        }
        model_params = {k:v for k,v in params.items() if k != "df"}
        model = LGBMRegressor(**model_params)
        model.fit(train_X, train_Y)
        preds = model.predict(test_X)
        actual_cum = (1 + test_Y).cumprod()
        pred_cum = (1 + preds).cumprod()
        rmse = np.sqrt(mean_squared_error(actual_cum, pred_cum))
        directional_acc = (np.sign(test_Y.values) == np.sign(preds)).mean()
        return rmse, -directional_acc

    study = optuna.create_study(directions=["minimize", "minimize"], sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_trials[0]
    return dict(best.params), float(best.values[0]), -float(best.values[1])

def _median_params(param_dicts: List[dict]) -> dict:
    if not param_dicts: return {}
    all_keys = set().union(*[d.keys() for d in param_dicts])
    consensus = {}
    for k in all_keys:
        vals = [d[k] for d in param_dicts if k in d]
        if not vals: continue
        if isinstance(vals[0], (int, np.integer)): consensus[k] = int(np.round(np.median(vals)))
        elif isinstance(vals[0], (float, np.floating)): consensus[k] = float(np.median(vals))
        else: consensus[k] = max(set(vals), key=vals.count)
    consensus["random_state"] = GLOBAL_SEED
    consensus["n_jobs"] = 1
    return consensus

def _eval_params_on_split(params: dict, train_X, train_Y, test_X, test_Y, seed=GLOBAL_SEED):
    lgbm_params = {k:v for k,v in params.items() if k not in ("df",)}
    lgbm_params["random_state"] = seed
    lgbm_params["n_jobs"] = 1
    model = LGBMRegressor(**lgbm_params)
    model.fit(train_X, train_Y)
    preds = model.predict(test_X)
    actual_cum = (1 + test_Y).cumprod()
    pred_cum = (1 + preds).cumprod()
    rmse = np.sqrt(mean_squared_error(actual_cum, pred_cum))
    directional_acc = (np.sign(test_Y.values) == np.sign(preds)).mean()
    return rmse, directional_acc

def tune_across_recent_oos_years(X: pd.DataFrame, Y: pd.Series, years_back: int = 5, seed: int = GLOBAL_SEED, n_trials: int = 50):
    years = _oos_years_available(Y.index, max_years=years_back)
    param_runs, details = [], []
    progress_outer = st.progress(0)
    for i, y in enumerate(years):
        train_X, train_Y, test_X, test_Y = _split_train_test_for_year(X, Y, y)
        if len(train_X) < 24 or len(test_X) < 6: continue
        best_params, rmse, da = _tune_on_explicit_split(train_X, train_Y, test_X, test_Y, seed=seed, n_trials=n_trials)
        details.append({"year": y, "rmse": rmse, "da": da, "best_params": best_params})
        param_runs.append(best_params)
        progress_outer.progress((i+1)/len(years))
    progress_outer.empty()
    consensus_params = _median_params(param_runs)
    last_rmse, last_da = np.nan, np.nan
    if years:
        last_year = years[-1]
        trX, trY, teX, teY = _split_train_test_for_year(X, Y, last_year)
        if len(trX) > 0 and len(teX) > 0:
            last_rmse, last_da = _eval_params_on_split(consensus_params, trX, trY, teX, teY, seed=seed)
    return consensus_params, details, last_rmse, last_da

# ---------- Monte Carlo ----------
def run_monte_carlo_paths(model, X_base, Y_base, residuals, sims_per_seed, rng, seed_id=None, df=5):
    horizon_months = FORECAST_YEARS * 12
    log_paths = np.zeros((sims_per_seed, horizon_months), dtype=np.float32)
    mu, sigma = residuals.mean(), residuals.std(ddof=0)
    snapshot_X = X_base.iloc[[-1]].values.astype(np.float32)
    last_X = np.repeat(snapshot_X, sims_per_seed, axis=0)
    base_pred = model.predict(last_X).astype(np.float32)
    for t in range(horizon_months):
        shocks = student_t.rvs(df, loc=mu, scale=sigma, size=sims_per_seed, random_state=rng).astype(np.float32)
        log_paths[:, t] = (log_paths[:, t-1] if t > 0 else 0) + base_pred + shocks
    return np.exp(log_paths, dtype=np.float32)

# ---------- Global Median-Ending Modal Line ----------
def find_global_modal_medoid(all_paths: np.ndarray):
    endings = all_paths[:, -1]
    median_ending = np.median(endings)
    tol = 0.01 * median_ending
    subset_idx = np.where(np.abs(endings - median_ending) <= tol)[0]
    if len(subset_idx) == 0:
        subset_idx = np.argsort(np.abs(endings - median_ending))[:max(1, len(all_paths)//20)]
    subset = all_paths[subset_idx]
    median_traj = np.median(subset, axis=0)
    diffs = np.sum(np.abs(subset - median_traj), axis=1)
    modal_idx = np.argmin(diffs)
    return subset[modal_idx]

# ---------- Forecast Stats ----------
def compute_forecast_stats_from_path(path: np.ndarray, start_capital: float, last_date: pd.Timestamp):
    if path is None or len(path) == 0:
        return {"CAGR": np.nan, "Volatility": np.nan, "Sharpe": np.nan, "Max Drawdown": np.nan}
    norm_path = path / path[0]
    forecast_index = pd.date_range(start=last_date, periods=len(norm_path)+1, freq="M")
    price = pd.Series(norm_path, index=forecast_index[:-1]) * start_capital
    monthly = price.pct_change().dropna()
    return {
        "CAGR": annualized_return_monthly(monthly),
        "Volatility": annualized_vol_monthly(monthly),
        "Sharpe": annualized_sharpe_monthly(monthly),
        "Max Drawdown": max_drawdown_from_rets(monthly)
    }

# ---------- SHAP ----------
def plot_feature_attributions(model, X, final_X):
    explainer = shap.TreeExplainer(model)
    shap_values_hist = explainer.shap_values(X)
    shap_mean_hist = np.abs(shap_values_hist).mean(axis=0)
    shap_values_fore = explainer.shap_values(final_X)
    shap_mean_fore = np.abs(shap_values_fore).reshape(-1)
    features = X.columns
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(np.arange(len(features))-0.2, shap_mean_hist, width=0.4, label="Backtest Avg")
    ax.bar(np.arange(len(features))+0.2, shap_mean_fore, width=0.4, label="Forecast Snapshot")
    ax.set_xticks(np.arange(len(features)))
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.set_ylabel("Average |SHAP Value|")
    ax.set_title("Feature Contributions: Backtest vs Forecast Snapshot")
    ax.legend()
    st.pyplot(fig)

# ---------- Plot Forecasts ----------
def plot_forecasts(port_rets, start_capital, central, rebalance_label):
    port_cum = (1 + port_rets).cumprod() * start_capital
    last_date = port_cum.index[-1]
    forecast_path = port_cum.iloc[-1] * (central / central[0])
    forecast_dates = pd.date_range(start=last_date, periods=len(central), freq="M")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(port_cum.index, port_cum.values, label="Portfolio Backtest")
    ax.plot([last_date, *forecast_dates], [port_cum.iloc[-1], *forecast_path], linewidth=2, label="Forecast")
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Balance ($)")
    st.pyplot(fig)

# ---------- Streamlit App ----------
def main():
    st.title("Portfolio Forecasting Tool (Global Median-Modal Path)")
    tickers = st.text_input("Tickers (comma-separated, e.g. VTI,AGG)", "VTI,AGG")
    weights_str = st.text_input("Weights (comma-separated, must sum > 0)", "0.6,0.4")
    start_capital = st.number_input("Starting Value ($)", min_value=1000.0, value=10000.0, step=1000.0)
    freq_map = {"M": "Monthly", "Q": "Quarterly", "S": "Semiannual", "Y": "Yearly", "N": "None"}
    rebalance_label = st.selectbox("Rebalance", list(freq_map.values()), index=0)
    rebalance_choice = [k for k,v in freq_map.items() if v == rebalance_label][0]

    if st.button("Run Forecast"):
        try:
            weights = to_weights([float(x) for x in weights_str.split(",")])
            tickers = [t.strip() for t in tickers.split(",") if t.strip()]
            prices = fetch_prices_monthly(tickers, start=DEFAULT_START)
            port_rets = portfolio_returns_monthly(prices, weights, rebalance_choice)
            df = build_features(port_rets)
            if df.empty:
                st.error("Feature engineering returned no data.")
                return
            Y = np.log(1 + port_rets.loc[df.index]).astype(np.float32)
            X = df.shift(1).dropna()
            Y = Y.loc[X.index]

            consensus_params, _, last_rmse, last_da = tune_across_recent_oos_years(X, Y, years_back=5, seed=GLOBAL_SEED, n_trials=50)
            st.json(consensus_params)
            st.write(f"OOS RMSE: {last_rmse:.6f} | DirAcc: {last_da:.2%}")

            df_opt = int(consensus_params.get("df", 5))
            lgbm_params = {k:v for k,v in consensus_params.items() if k != "df"}
            lgbm_params["random_state"] = GLOBAL_SEED
            lgbm_params["n_jobs"] = 1
            final_model = LGBMRegressor(**lgbm_params)
            final_model.fit(X, Y)
            residuals = (Y.values - final_model.predict(X)).astype(np.float32)

            all_sims = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            for i in range(ENSEMBLE_SEEDS):
                rng = np.random.default_rng(GLOBAL_SEED + i)
                sims = run_monte_carlo_paths(final_model, X, Y, residuals, SIMS_PER_SEED, rng, df=df_opt)
                all_sims.append(sims)
                progress_bar.progress((i+1)/ENSEMBLE_SEEDS)
                status_text.text(f"Running forecasts... {i+1}/{ENSEMBLE_SEEDS}")
            progress_bar.empty()

            final_path = find_global_modal_medoid(np.vstack(all_sims))
            stats = compute_forecast_stats_from_path(final_path, start_capital, port_rets.index[-1])
            backtest_stats = {
                "CAGR": annualized_return_monthly(port_rets),
                "Volatility": annualized_vol_monthly(port_rets),
                "Sharpe": annualized_sharpe_monthly(port_rets),
                "Max Drawdown": max_drawdown_from_rets(port_rets)
            }

            st.subheader("Results")
            c1, c2 = st.columns(2)
            for k,v in backtest_stats.items(): c1.metric(k, f"{v:.2%}" if 'Sharpe' not in k else f"{v:.2f}")
            for k,v in stats.items(): c2.metric(k, f"{v:.2%}" if 'Sharpe' not in k else f"{v:.2f}")
            st.metric("Forecasted Portfolio Value", f"${float(final_path[-1])*start_capital:,.2f}")

            plot_forecasts(port_rets, start_capital, final_path, rebalance_label)
            plot_feature_attributions(final_model, X, X.iloc[[-1]])

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()