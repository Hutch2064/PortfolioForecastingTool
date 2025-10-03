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
import math
import streamlit as st
from sklearn.metrics import mean_squared_error
import optuna
from pandas_datareader import data as pdr  # for FRED sentiment

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

# ---------- Data Fetch ----------
@st.cache_data(show_spinner=False)
def fetch_umich_sentiment(start=DEFAULT_START) -> pd.Series:
    try:
        umcsent = pdr.DataReader("UMCSENT", "fred", start)
        umcsent = umcsent.resample("M").last().ffill()
        umcsent.name = "umcsent"
        st.write("DEBUG: UMCSENT fetched rows:", len(umcsent))
        st.write("DEBUG: UMCSENT head:", umcsent.head())
        return umcsent.astype(np.float32).squeeze()
    except Exception as e:
        st.write("DEBUG: UMCSENT fetch failed:", e)
        return pd.Series(dtype=np.float32, name="umcsent")

# ---------- Feature Builders ----------
def build_features(returns: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame(index=returns.index)
    df["mom_3m"] = returns.rolling(3).apply(lambda x: (1+x).prod()-1, raw=True)
    df["mom_6m"] = returns.rolling(6).apply(lambda x: (1+x).prod()-1, raw=True)
    df["mom_12m"] = returns.rolling(12).apply(lambda x: (1+x).prod()-1, raw=True)
    df["dd_state"] = (1 + returns.fillna(0)).cumprod() / (1 + returns.fillna(0)).cumprod().cummax() - 1

    umcsent = fetch_umich_sentiment(start=returns.index.min().to_pydatetime().date())
    st.write("DEBUG: UMCSENT returned to build_features:", "empty" if umcsent.empty else "non-empty")
    if not umcsent.empty:
        umcsent = umcsent.loc[df.index.min():df.index.max()]
        df["umcsent"] = umcsent.reindex(df.index).astype(np.float32).fillna(method="ffill").fillna(method="bfill")

    st.write("DEBUG: build_features df columns:", list(df.columns))
    st.write("DEBUG: build_features df head:", df.head())
    return df.dropna().astype(np.float32)

# ---------- Model Training ----------
def tune_and_fit_best_model(X: pd.DataFrame, Y: pd.Series, seed=GLOBAL_SEED):
    model = LGBMRegressor(random_state=seed, n_jobs=1)
    model.fit(X, Y, feature_name=list(X.columns))
    st.write("DEBUG: Model features seen:", model.booster_.feature_name())
    return model

# ---------- SHAP ----------
def plot_feature_attributions(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    st.write("DEBUG: SHAP features:", list(X.columns))
    st.write("DEBUG: SHAP values shape:", np.array(shap_values).shape)

    shap_mean = np.abs(shap_values).mean(axis=0)
    x_pos = np.arange(len(X.columns))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_pos, shap_mean, width=0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(X.columns, rotation=45, ha="right")
    ax.set_ylabel("Average |SHAP Value|")
    ax.set_title("Feature Contributions (DEBUG)")
    st.pyplot(fig)

# ---------- Main ----------
def main():
    st.title("UMCSENT Debugging")

    # Fake returns just for debug
    dates = pd.date_range("2010-01-01", periods=200, freq="M")
    rets = pd.Series(np.random.randn(len(dates)) / 100, index=dates)

    df = build_features(rets)
    X = df.shift(1).dropna()
    Y = np.log(1 + rets.loc[X.index])

    st.write("DEBUG: Final X columns:", list(X.columns))
    st.write("DEBUG: Final X head:", X.head())

    model = tune_and_fit_best_model(X, Y)
    plot_feature_attributions(model, X)

if __name__ == "__main__":
    main()