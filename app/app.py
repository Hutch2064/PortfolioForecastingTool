import sys
import warnings
import random
import os
import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Dict
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import shap
import streamlit as st
from sklearn.metrics import mean_squared_error
import optuna

warnings.filterwarnings("ignore")

# ---------- Global Seed Fix ----------
GLOBAL_SEED = 42
os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# ---------- Config ----------
DEFAULT_START = "2000-01-01"
ENSEMBLE_SEEDS = 100
SIMS_PER_SEED = 10000
FORECAST_YEARS = 1

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

def max_drawdown_from_rets(returns):
    cum = (1 + returns.fillna(0)).cumprod()
    roll_max = cum.cummax()
    return (cum / roll_max - 1.0).min()

def compute_current_drawdown(returns):
    cum = (1 + returns.fillna(0)).cumprod()
    roll_max = cum.cummax()
    return (cum / roll_max - 1.0).astype(np.float32)

# ---------- Data Fetch ----------
def fetch_prices_monthly(tickers, start=DEFAULT_START):
    data = yf.download(tickers, start=start, interval="1mo", auto_adjust=False, progress=False, threads=False)
    if data.empty:
        raise ValueError("No price data returned.")
    if isinstance(data.columns, pd.MultiIndex):
        for f in ["Adj Close", "Close"]:
            if f in data.columns.get_level_values(0):
                close = data[f].copy(); break
        else:
            raise ValueError("No valid price field.")
    else:
        close = data.copy()
    close = close.ffill().astype(np.float32)
    df = pd.DataFrame(close)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(1)
    return df.dropna(how="all")

def fetch_macro_features(start=DEFAULT_START):
    tickers = ["^VIX", "^MOVE", "^TNX", "^IRX"]
    data = yf.download(tickers, start=start, interval="1mo", progress=False, threads=False)
    close = data["Close"].ffill().astype(np.float32)
    df = pd.DataFrame(index=close.index)
    df["VIX"] = close["^VIX"]
    df["MOVE"] = close["^MOVE"]
    df["YC_Spread"] = close["^TNX"] - close["^IRX"]
    return df

# ---------- Portfolio ----------
def portfolio_returns_monthly(prices, weights, rebalance):
    rets = prices.pct_change().dropna().astype(np.float32)
    if rebalance == "N":
        vals = (1 + rets).cumprod()
        port_vals = vals.dot(weights)
        port_vals = port_vals / port_vals.iloc[0]
        return port_vals.pct_change().fillna(0.0)
    freq_map = {"M": "M", "Q": "Q", "S": "2Q", "Y": "A"}
    rule = freq_map.get(rebalance)
    if not rule: raise ValueError("Invalid rebalance option.")
    port_val, port_vals, current_weights = 1.0, [], weights.copy()
    rebalance_dates = rets.resample(rule).last().index
    for i, date in enumerate(rets.index):
        if i > 0: port_val *= (1 + (rets.iloc[i] @ current_weights))
        port_vals.append(port_val)
        if date in rebalance_dates: current_weights = weights.copy()
    return pd.Series(port_vals, index=rets.index, name="Portfolio").pct_change().fillna(0.0)

# ---------- Features ----------
def build_features(returns):
    df = pd.DataFrame()
    df["mom_3m"] = returns.rolling(3).apply(lambda x: (1+x).prod()-1, raw=True)
    df["mom_6m"] = returns.rolling(6).apply(lambda x: (1+x).prod()-1, raw=True)
    df["mom_12m"] = returns.rolling(12).apply(lambda x: (1+x).prod()-1, raw=True)
    df["dd_state"] = compute_current_drawdown(returns)
    macro = fetch_macro_features()
    df = df.join(macro, how="left").ffill()
    valid_start = max([df[c].first_valid_index() for c in df.columns if df[c].first_valid_index()])
    df = df.loc[valid_start:].dropna()
    return df.astype(np.float32)

# ---------- Optuna ----------
def _oos_years_available(idx, max_years=5):
    years = sorted(set(idx.year))
    full = []
    for y in years:
        months = set(idx[idx.year == y].month)
        if all(m in months for m in range(1, 13)): full.append(y)
    return full[-max_years:]

def _split_train_test_for_year(X, Y, y):
    train_end = pd.Timestamp(f"{y-1}-12-31")
    test_start = pd.Timestamp(f"{y}-01-01")
    train_X = X.loc[:train_end]
    train_Y = Y.loc[train_X.index]
    test_X = X.loc[test_start:]
    test_Y = Y.loc[test_X.index]
    return train_X, train_Y, test_X, test_Y

def _median_params(params_list):
    if not params_list: return {}
    keys = set().union(*[d.keys() for d in params_list])
    out = {}
    for k in keys:
        vals = [d[k] for d in params_list if k in d]
        if not vals: continue
        if isinstance(vals[0], (int, np.integer)):
            out[k] = int(np.median(vals))
        elif isinstance(vals[0], (float, np.floating)):
            out[k] = float(np.median(vals))
        else:
            out[k] = max(set(vals), key=vals.count)
    out["random_state"] = GLOBAL_SEED
    out["n_jobs"] = 1
    return out

def tune_across_recent_oos_years(X, Y, years_back=5, seed=GLOBAL_SEED, n_trials=50):
    years = _oos_years_available(Y.index, years_back)
    params_all, details = [], []
    total_jobs = len(years)*n_trials
    bar = st.progress(0); txt = st.empty(); done = 0
    for y in years:
        Xtr, Ytr, Xte, Yte = _split_train_test_for_year(X, Y, y)
        if len(Xtr)<24 or len(Xte)<6: continue
        def objective(trial):
            nonlocal done
            params = {
                "n_estimators": trial.suggest_int("n_estimators",100,2500),
                "learning_rate": trial.suggest_float("learning_rate",0.001,1,log=True),
                "max_depth": trial.suggest_int("max_depth",2,6),
                "subsample": trial.suggest_float("subsample",0.5,1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree",0.5,1.0),
                "block_length": trial.suggest_int("block_length",3,6),
                "random_state": seed, "n_jobs":1
            }
            mdl = LGBMRegressor(**{k:v for k,v in params.items() if k!="block_length"})
            mdl.fit(Xtr,Ytr)
            preds = mdl.predict(Xte)
            rmse = np.sqrt(mean_squared_error((1+Yte).cumprod(),(1+preds).cumprod()))
            da = (np.sign(Yte)==np.sign(preds)).mean()
            done+=1; bar.progress(done/total_jobs)
            txt.text(f"Tuning models... {int(done/total_jobs*100)}%")
            return rmse, -da
        study=optuna.create_study(directions=["minimize","minimize"],sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective,n_trials=n_trials,show_progress_bar=False)
        best=study.best_trials[0]
        details.append({"year":y,"rmse":float(best.values[0]),"da":-float(best.values[1]),"best_params":dict(best.params)})
        params_all.append(dict(best.params))
    bar.empty(); txt.empty()
    return _median_params(params_all), details, np.nan, np.nan

# ---------- Indicator Models ----------
def train_indicator_models(X, feats):
    models={}
    for f in feats:
        if f not in X.columns: continue
        y=X[f].shift(-1).dropna(); idx=y.index.intersection(X.index)
        if len(idx)<24: continue
        mdl=LGBMRegressor(n_estimators=500,max_depth=3,learning_rate=0.05,
                          subsample=0.8,colsample_bytree=0.8,random_state=GLOBAL_SEED,n_jobs=1)
        mdl.fit(X.loc[idx],y.loc[idx])
        models[f]=mdl
    return models

# ---------- Block Bootstrap (Overlapping) ----------
def block_bootstrap_residuals(residuals, size, block_len, rng):
    n = len(residuals)
    indices = []
    for _ in range(size):
        start = rng.integers(0, n)
        block = [residuals[(start + j) % n] for j in range(block_len)]
        indices.extend(block)
        if len(indices) >= size: break
    arr = np.array(indices[:size], dtype=np.float32)
    return np.ascontiguousarray(arr)

# ---------- Monte Carlo ----------
def run_monte_carlo_paths(model, X_base, Y_base, residuals, sims_per_seed, rng, seed_id=None, block_len=12, indicator_models=None):
    horizon = FORECAST_YEARS*12
    log_paths = np.zeros((sims_per_seed,horizon),dtype=np.float32)
    ar_returns = np.zeros_like(log_paths)
    state = pd.Series(X_base.iloc[-1].values, index=X_base.columns).astype(np.float32)
    cum_val = np.ones(sims_per_seed,dtype=np.float32)
    cum_max = np.ones_like(cum_val)

    def _mom_med(k,t):
        start=max(0,t-k+1)
        window=ar_returns[:,start:t+1]
        if window.shape[1]==0: return 0.0
        prod=np.prod(1+window,axis=1)-1
        return float(np.median(prod))

    hist_std = np.std(residuals, ddof=0)

    for t in range(horizon):
        mu_t=float(model.predict(state.values.reshape(1,-1))[0])
        shocks=block_bootstrap_residuals(residuals,sims_per_seed,block_len,rng)

        # Scale shocks to match historical residual volatility
        sim_std = np.std(shocks, ddof=0)
        if hist_std > 0 and sim_std > 0:
            shocks *= hist_std / sim_std

        log_step=mu_t+shocks
        log_paths[:,t]=(log_paths[:,t-1] if t>0 else 0)+log_step
        ar_t=np.expm1(log_step); ar_returns[:,t]=ar_t
        cum_val*=(1+ar_t); cum_max=np.maximum(cum_max,cum_val)
        dd_med=float(np.median(cum_val/cum_max-1))
        state["mom_3m"]=np.float32(_mom_med(3,t))
        state["mom_6m"]=np.float32(_mom_med(6,t))
        state["mom_12m"]=np.float32(_mom_med(12,t))
        state["dd_state"]=np.float32(dd_med)
        if indicator_models:
            for feat in("VIX","MOVE","YC_Spread"):
                if feat in indicator_models:
                    state[feat]=float(indicator_models[feat].predict(state.values.reshape(1,-1))[0])
    return np.exp(log_paths,dtype=np.float32)

# ---------- Median Path Selection ----------
def find_median_path(paths):
    final_vals = paths[:,-1]
    median_val = np.median(final_vals)
    idx = np.argmin(np.abs(final_vals - median_val))
    return paths[idx]

# ---------- Stats ----------
def compute_forecast_stats_from_path(path,start_cap,last_date):
    norm=path/path[0]
    idx=pd.date_range(start=last_date,periods=len(norm)+1,freq="M")
    price=pd.Series(norm,index=idx[:-1])*start_cap
    rets=price.pct_change().dropna()
    return {
        "CAGR":annualized_return_monthly(rets),
        "Volatility":annualized_vol_monthly(rets),
        "Sharpe":annualized_sharpe_monthly(rets),
        "Max Drawdown":max_drawdown_from_rets(rets)
    }

# ---------- SHAP ----------
def plot_feature_attributions(model,X,final_X):
    expl=shap.TreeExplainer(model)
    sh_hist=np.abs(expl.shap_values(X)).mean(axis=0)
    sh_fore=np.abs(expl.shap_values(final_X)).reshape(-1)
    feats=X.columns; pos=np.arange(len(feats))
    fig,ax=plt.subplots(figsize=(10,6))
    ax.bar(pos-0.2,sh_hist,width=0.4,label="Backtest Avg")
    ax.bar(pos+0.2,sh_fore,width=0.4,label="Forecast Snapshot")
    ax.set_xticks(pos); ax.set_xticklabels(feats,rotation=45,ha="right")
    ax.set_ylabel("Average |SHAP Value|"); ax.legend()
    st.pyplot(fig)

# ---------- Plot ----------
def plot_forecasts(port_rets,start_cap,central,reb_label):
    port_cum=(1+port_rets).cumprod()*start_cap
    last=port_cum.index[-1]
    fore=port_cum.iloc[-1]*(central/central[0])
    dates=pd.date_range(start=last,periods=len(central),freq="M")
    fig,ax=plt.subplots(figsize=(12,6))
    ax.plot(port_cum.index,port_cum.values,label="Portfolio Backtest")
    ax.plot([last,*dates],[port_cum.iloc[-1],*fore],label="Forecast (Median Path)",lw=2)
    ax.legend(); st.pyplot(fig)

# ---------- Streamlit ----------
def main():
    st.title("Portfolio Forecasting Tool – Ensemble Median Path")
    tickers=st.text_input("Tickers","VTI,AGG")
    weights_str=st.text_input("Weights","0.6,0.4")
    start_cap=st.number_input("Starting Value ($)",1000.0,1000000.0,10000.0,1000.0)
    freq_map={"M":"Monthly","Q":"Quarterly","S":"Semiannual","Y":"Yearly","N":"None"}
    reb_label=st.selectbox("Rebalance",list(freq_map.values()),index=0)
    reb=[k for k,v in freq_map.items() if v==reb_label][0]

    if st.button("Run Forecast"):
        try:
            weights=to_weights([float(x) for x in weights_str.split(",")])
            tickers=[t.strip() for t in tickers.split(",") if t.strip()]
            prices=fetch_prices_monthly(tickers,DEFAULT_START)
            port_rets=portfolio_returns_monthly(prices,weights,reb)
            df=build_features(port_rets)
            Y=np.log(1+port_rets.loc[df.index]).astype(np.float32)
            X=df.shift(1).dropna(); Y=Y.loc[X.index]

            cons,_,_,_=tune_across_recent_oos_years(X,Y,5,GLOBAL_SEED,50)
            st.json(cons)
            blk_len=int(cons.get("block_length",12))
            lgb_params={k:v for k,v in cons.items() if k!="block_length"}
            model=LGBMRegressor(**lgb_params); model.fit(X,Y)
            res=(Y.values-model.predict(X)).astype(np.float32)
            res=np.ascontiguousarray(res[~np.isnan(res)])

            indicators=train_indicator_models(X,["VIX","MOVE","YC_Spread"])
            hist_vol=annualized_vol_monthly(port_rets)

            all_paths=[]; bar=st.progress(0); txt=st.empty()
            for i in range(ENSEMBLE_SEEDS):
                rng=np.random.default_rng(GLOBAL_SEED+i)
                sims=run_monte_carlo_paths(model,X,Y,res,SIMS_PER_SEED,rng,i,blk_len,indicators)
                all_paths.append(sims)
                bar.progress((i+1)/ENSEMBLE_SEEDS)
                txt.text(f"Running forecasts... {i+1}/{ENSEMBLE_SEEDS}")
            bar.empty(); txt.empty()

            paths=np.vstack(all_paths)
            final=find_median_path(paths)
            stats=compute_forecast_stats_from_path(final,start_cap,port_rets.index[-1])
            back={"CAGR":annualized_return_monthly(port_rets),
                  "Volatility":annualized_vol_monthly(port_rets),
                  "Sharpe":annualized_sharpe_monthly(port_rets),
                  "Max Drawdown":max_drawdown_from_rets(port_rets)}

            st.subheader("Results")
            c1,c2=st.columns(2)
            with c1:
                st.markdown("**Backtest**")
                for k,v in back.items():
                    st.metric(k,f"{v:.2%}" if "Sharpe" not in k else f"{v:.2f}")
            with c2:
                st.markdown("**Forecast (Median Path)**")
                for k,v in stats.items():
                    st.metric(k,f"{v:.2%}" if "Sharpe" not in k else f"{v:.2f}")
            st.metric("Forecasted Portfolio Value",f"${final[-1]*start_cap:,.2f}")

            plot_forecasts(port_rets,start_cap,final,reb_label)
            plot_feature_attributions(model,X,X.iloc[[-1]])

        except Exception as e:
            st.error(f"Error: {e}")

if __name__=="__main__":
    main()

