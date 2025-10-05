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
from scipy.stats import t as student_t  # Student-t distribution

warnings.filterwarnings("ignore")

# ---------- Global Seed Fix ----------
GLOBAL_SEED = 42
os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# ---------- Config ----------
DEFAULT_START = "2000-01-01"
ENSEMBLE_SEEDS = 100       # number of seeds in the ensemble
SIMS_PER_SEED = 10000      # simulations per seed
FORECAST_YEARS = 1         # 12-month horizon

# ---------- Helpers ----------
def to_weights(raw: List[float]) -> np.ndarray:
    arr = np.array(raw, dtype=np.float32)
    s = arr.sum()
    if s <= 0:
        raise ValueError("Weights must sum to a positive number.")
    return arr / s

def annualized_return_monthly(m):
    m = m.dropna()
    if m.empty: return np.nan
    compounded = (1 + m).prod()
    years = len(m) / 12.0
    return compounded ** (1 / years) - 1 if years > 0 else np.nan

def annualized_vol_monthly(m):
    m = m.dropna()
    return m.std(ddof=0) * np.sqrt(12) if len(m) > 1 else np.nan

def annualized_sharpe_monthly(m, rf_monthly=0.0):
    m = m.dropna()
    if m.empty: return np.nan
    excess = m - rf_monthly
    mu, sigma = excess.mean(), excess.std(ddof=0)
    return (mu / sigma) * np.sqrt(12) if sigma and sigma > 0 else np.nan

def max_drawdown_from_rets(r):
    cum = (1 + r.fillna(0)).cumprod()
    roll_max = cum.cummax()
    dd = cum / roll_max - 1.0
    return dd.min()

def compute_current_drawdown(returns):
    cum = (1 + returns.fillna(0)).cumprod()
    roll_max = cum.cummax()
    return (cum / roll_max - 1.0).astype(np.float32)

# ---------- Data Fetch ----------
def fetch_prices_monthly(tickers, start=DEFAULT_START):
    data = yf.download(tickers, start=start, auto_adjust=False, progress=False, interval="1mo", threads=False)
    if data.empty:
        raise ValueError("No price data returned.")
    if isinstance(data.columns, pd.MultiIndex):
        for f in ["Adj Close", "Close"]:
            if f in data.columns.get_level_values(0):
                close = data[f].copy()
                break
        else:
            raise ValueError("Missing Close field.")
    else:
        col = "Adj Close" if "Adj Close" in data.columns else "Close"
        close = pd.DataFrame(data[col]); close.columns = tickers
    close = close.ffill().dropna(how="all").astype(np.float32)
    start_idx = max(c.first_valid_index() for _, c in close.items())
    return close.loc[start_idx:]

def fetch_macro_features(start=DEFAULT_START):
    tickers = ["^VIX", "^MOVE", "^TNX", "^IRX"]
    data = yf.download(tickers, start=start, auto_adjust=False, progress=False, interval="1mo", threads=False)
    if isinstance(data.columns, pd.MultiIndex): 
        close = data["Close"].copy()
    else:
        close = data.copy()
    close = close.ffill().astype(np.float32)
    df = pd.DataFrame(index=close.index)
    df["VIX"] = close["^VIX"]; df["MOVE"] = close["^MOVE"]
    df["YC_Spread"] = close["^TNX"] - close["^IRX"]
    return df

# ---------- Portfolio ----------
def portfolio_returns_monthly(prices, weights, rebalance):
    rets = prices.pct_change().dropna(how="all").astype(np.float32)
    if rebalance == "N":
        vals = (1 + rets).cumprod()
        port_vals = vals.dot(weights)
        port_vals = port_vals / port_vals.iloc[0]
        return port_vals.pct_change().fillna(0.0).astype(np.float32)
    freq_map = {"M": "M", "Q": "Q", "S": "2Q", "Y": "A"}
    rule = freq_map.get(rebalance)
    port_val, port_vals, cw = 1.0, [], weights.copy()
    rebalance_dates = rets.resample(rule).last().index
    for i, date in enumerate(rets.index):
        if i > 0:
            port_val *= (1 + (rets.iloc[i] @ cw))
        port_vals.append(port_val)
        if date in rebalance_dates:
            cw = weights.copy()
    return pd.Series(port_vals, index=rets.index, name="Portfolio").pct_change().fillna(0.0).astype(np.float32)

# ---------- Features ----------
def build_features(returns):
    df = pd.DataFrame()
    df["mom_3m"] = returns.rolling(3).apply(lambda x: (1+x).prod()-1, raw=True)
    df["mom_6m"] = returns.rolling(6).apply(lambda x: (1+x).prod()-1, raw=True)
    df["mom_12m"] = returns.rolling(12).apply(lambda x: (1+x).prod()-1, raw=True)
    df["dd_state"] = compute_current_drawdown(returns)
    macro = fetch_macro_features()
    df = df.join(macro, how="left").ffill()
    valid_start = max(df[c].first_valid_index() for c in df.columns if df[c].first_valid_index())
    return df.loc[valid_start:].dropna().astype(np.float32)

# ---------- Parameter Utilities ----------
def _median_params(param_dicts):
    if not param_dicts: return {}
    keys = set().union(*[d.keys() for d in param_dicts])
    out = {}
    for k in keys:
        vals = [d[k] for d in param_dicts if k in d]
        if isinstance(vals[0], (int, np.integer)): out[k] = int(np.median(vals))
        elif isinstance(vals[0], (float, np.floating)): out[k] = float(np.median(vals))
        else: out[k] = max(set(vals), key=vals.count)
    return out

# ---------- Optuna Tuning ----------
def tune_across_recent_oos_years(X, Y, years_back=5, seed=GLOBAL_SEED, n_trials=100):
    years = sorted(set(Y.index.year))[-years_back:]
    param_runs, details = [], []
    total_jobs = len(years)*n_trials
    progress_bar = st.progress(0)
    status_text = st.empty()
    completed = 0
    for y in years:
        train_X, test_X = X.loc[:f"{y-1}-12-31"], X.loc[f"{y}-01-01":f"{y}-12-31"]
        train_Y, test_Y = Y.loc[train_X.index], Y.loc[test_X.index]
        if len(train_X)<24 or len(test_X)<6: continue
        def objective(trial):
            nonlocal completed
            p = {
                "n_estimators": trial.suggest_int("n_estimators", 1000, 8000),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "df": trial.suggest_int("df", 3, 30)
            }
            m = LGBMRegressor(**{k:v for k,v in p.items() if k!="df"})
            m.fit(train_X, train_Y)
            preds = m.predict(test_X)
            rmse = np.sqrt(mean_squared_error((1+test_Y).cumprod(), (1+preds).cumprod()))
            da = (np.sign(test_Y.values)==np.sign(preds)).mean()
            completed += 1
            if completed % 20 == 0:
                progress_bar.progress(completed/total_jobs)
                status_text.text(f"Tuning... {int(100*completed/total_jobs)}%")
            return rmse, -da
        study = optuna.create_study(directions=["minimize","minimize"], sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = study.best_trials[0]
        details.append({"year":y,"rmse":float(best.values[0]),"da":-float(best.values[1]),"best_params":dict(best.params)})
        param_runs.append(dict(best.params))
    consensus = _median_params(param_runs)
    progress_bar.empty()
    return consensus, details, np.nan, np.nan

# ---------- Streamed Global Modal Medoid ----------
def find_streamed_global_modal_medoid(seed_medoids):
    all_medoids = np.vstack(seed_medoids).astype(np.float32)
    endings = all_medoids[:, -1]
    median_ending = np.median(endings)
    tol = 0.01 * abs(median_ending)
    subset_idx = np.where(np.abs(endings - median_ending) <= tol)[0]
    if len(subset_idx) == 0:
        subset_idx = np.argsort(np.abs(endings - median_ending))[:max(1, len(all_medoids)//20)]
    subset = all_medoids[subset_idx]
    median_traj = np.median(subset, axis=0)
    diffs = np.sum(np.abs(subset - median_traj), axis=1)
    best_idx = np.argmin(diffs)
    return subset[best_idx]

# ---------- Monte Carlo ----------
def run_monte_carlo_paths(model, X_base, Y_base, residuals, sims_per_seed, rng, seed_id=None, df=5):
    horizon = FORECAST_YEARS*12
    log_paths = np.zeros((sims_per_seed,horizon),dtype=np.float32)
    mu,sigma=residuals.mean(),residuals.std(ddof=0)
    last_X=np.repeat(X_base.iloc[[-1]].values.astype(np.float32),sims_per_seed,axis=0)
    base_pred=model.predict(last_X).astype(np.float32)
    for t in range(horizon):
        shocks=student_t.rvs(df,loc=mu,scale=sigma,size=sims_per_seed,random_state=rng).astype(np.float32)
        log_paths[:,t]=(log_paths[:,t-1] if t>0 else 0)+base_pred+shocks
    return np.exp(log_paths,dtype=np.float32)

# ---------- Forecast Stats ----------
def compute_forecast_stats_from_path(path,start_cap,last_date):
    if path is None or len(path)==0:
        return {"CAGR":np.nan,"Volatility":np.nan,"Sharpe":np.nan,"Max Drawdown":np.nan}
    norm=path/path[0]; idx=pd.date_range(start=last_date,periods=len(norm)+1,freq="M")
    price=pd.Series(norm,index=idx[:-1])*start_cap
    m=price.pct_change().dropna()
    return {"CAGR":annualized_return_monthly(m),"Volatility":annualized_vol_monthly(m),
            "Sharpe":annualized_sharpe_monthly(m),"Max Drawdown":max_drawdown_from_rets(m)}

# ---------- SHAP ----------
def plot_feature_attributions(model,X,final_X):
    expl=shap.TreeExplainer(model)
    hist=np.abs(expl.shap_values(X)).mean(axis=0)
    fore=np.abs(expl.shap_values(final_X)).reshape(-1)
    feats=X.columns
    fig,ax=plt.subplots(figsize=(10,6))
    ax.bar(np.arange(len(feats))-0.2,hist,width=0.4,label="Backtest Avg")
    ax.bar(np.arange(len(feats))+0.2,fore,width=0.4,label="Forecast Snapshot")
    ax.set_xticks(np.arange(len(feats))); ax.set_xticklabels(feats,rotation=45,ha="right")
    ax.legend(); ax.set_ylabel("Avg |SHAP|"); ax.set_title("Feature Contributions")
    st.pyplot(fig)

# ---------- Plot ----------
def plot_forecasts(port_rets,start_cap,central,rebalance_label):
    port_cum=(1+port_rets).cumprod()*start_cap
    last=port_cum.index[-1]
    forecast=port_cum.iloc[-1]*(central/central[0])
    dates=pd.date_range(start=last,periods=len(central),freq="M")
    fig,ax=plt.subplots(figsize=(12,6))
    ax.plot(port_cum.index,port_cum.values,label="Portfolio Backtest")
    ax.plot([last,*dates],[port_cum.iloc[-1],*forecast],linewidth=2,label="Forecast")
    ax.legend(); ax.set_xlabel("Date"); ax.set_ylabel("Balance ($)")
    st.pyplot(fig)

# ---------- Main ----------
def main():
    st.title("Portfolio Forecasting Tool (Optimized Global Modal)")

    tickers=st.text_input("Tickers","VTI,AGG")
    weights_str=st.text_input("Weights","0.6,0.4")
    start_cap=st.number_input("Starting Value ($)",min_value=1000.0,value=10000.0,step=1000.0)
    freq_map={"M":"Monthly","Q":"Quarterly","S":"Semiannual","Y":"Yearly","N":"None"}
    rebalance_label=st.selectbox("Rebalance",list(freq_map.values()),index=0)
    rebalance=[k for k,v in freq_map.items() if v==rebalance_label][0]

    if st.button("Run Forecast"):
        try:
            weights=to_weights([float(x) for x in weights_str.split(",")])
            tickers=[t.strip() for t in tickers.split(",") if t.strip()]
            prices=fetch_prices_monthly(tickers)
            port_rets=portfolio_returns_monthly(prices,weights,rebalance)
            df=build_features(port_rets)
            if df.empty: return st.error("No features.")
            Y=np.log(1+port_rets.loc[df.index]).astype(np.float32)
            X=df.shift(1).dropna(); Y=Y.loc[X.index]

            params,details,rmse,da=tune_across_recent_oos_years(X,Y,5,GLOBAL_SEED,100)
            st.json(params)
            st.write(f"OOS Tuning Complete | Trials: 100")

            df_opt=int(params.get("df",5))
            model=LGBMRegressor(**{k:v for k,v in params.items() if k!="df"})
            model.fit(X,Y)
            residuals=(Y.values-model.predict(X)).astype(np.float32)

            seed_medoids=[]
            progress_bar=st.progress(0); status_text=st.empty()
            for i in range(ENSEMBLE_SEEDS):
                rng=np.random.default_rng(GLOBAL_SEED+i)
                sims=run_monte_carlo_paths(model,X,Y,residuals,SIMS_PER_SEED,rng,df=df_opt)
                seed_medoids.append(find_streamed_global_modal_medoid(sims))
                del sims  # immediately free memory
                progress_bar.progress((i+1)/ENSEMBLE_SEEDS)
                status_text.text(f"Running forecasts... {i+1}/{ENSEMBLE_SEEDS}")
            progress_bar.empty()

            final_path=find_streamed_global_modal_medoid(seed_medoids)
            stats=compute_forecast_stats_from_path(final_path,start_cap,port_rets.index[-1])
            back={"CAGR":annualized_return_monthly(port_rets),"Volatility":annualized_vol_monthly(port_rets),
                  "Sharpe":annualized_sharpe_monthly(port_rets),"Max Drawdown":max_drawdown_from_rets(port_rets)}

            st.subheader("Results")
            c1,c2=st.columns(2)
            for k,v in back.items(): c1.metric(k,f"{v:.2%}" if 'Sharpe' not in k else f"{v:.2f}")
            for k,v in stats.items(): c2.metric(k,f"{v:.2%}" if 'Sharpe' not in k else f"{v:.2f}")
            st.metric("Forecasted Portfolio Value",f"${float(final_path[-1])*start_cap:,.2f}")
            plot_forecasts(port_rets,start_cap,final_path,rebalance_label)
            plot_feature_attributions(model,X,X.iloc[[-1]])

        except Exception as e:
            st.error(f"Error: {e}")

if __name__=="__main__":
    main()