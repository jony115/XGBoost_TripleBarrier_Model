# -*- coding: utf-8 -*-
"""
Triple Barrier Forex Model (Final Production Training)
Workflow: 
1. Load Data
2. Feature Engineering
3. Validation Check (80/20 Split)
4. Production Training (100% Data) -> Save Model
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pytz
import os
import glob
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ==========================================
# 1. CONFIGURATIONS
# ==========================================
# TRADING LOGIC
RISK_REWARD_RATIO = 1.2  # 1.2x profit
BARRIER_WIDTH = 1.2      # 1.2x ATR Stop Loss
TIME_HORIZON = 24        

# MODEL SETTINGS
TRADE_THRESHOLD = 0.60   

# DATA SETTINGS
PREFERRED_FILES = [
    "EURUSD_extracted.csv", 
    "EURUSD_H1_201501020900_202512040900.csv"
]

# ==========================================
# 2. DATA EXTRACTION & LOADING
# ==========================================

def get_data_from_mt5(symbol="EURUSD", timeframe=mt5.TIMEFRAME_H1, start_year=2010, end_year=2025):
    print(f"\n--- Attempting to Connect to MetaTrader 5 ---")
    if not mt5.initialize():
        print(f"Warning: MT5 initialize() failed. Error: {mt5.last_error()}")
        return None

    timezone = pytz.timezone("Etc/UTC")
    utc_from = datetime(start_year, 1, 1, tzinfo=timezone)
    utc_to = datetime(end_year, 12, 31, tzinfo=timezone)
    
    print(f"Fetching {symbol} data...")
    rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
    return df

def find_and_load_local_csv():
    """ Smarter loader: Looks for preferred files first, then any CSV. """
    filename = None
    for f in PREFERRED_FILES:
        if os.path.exists(f):
            filename = f
            break
            
    if filename is None:
        csv_files = glob.glob("*.csv")
        if csv_files:
            filename = csv_files[0]
            
    if filename is None:
        return None
    
    print(f"Loading local file: {filename}")
    try:
        with open(filename, 'r') as f:
            header = f.readline()
        
        if '\t' in header:
            df = pd.read_csv(filename, sep='\t', header=0)
            df.columns = df.columns.str.strip().str.lower().str.replace('<|>', '', regex=True)
            if 'date' in df.columns and 'time' in df.columns:
                df['datetime'] = df['date'] + ' ' + df['time']
                df['time'] = pd.to_datetime(df['datetime'])
                df = df.drop(columns=['date', 'datetime', 'vol', 'spread'], errors='ignore')
            if 'tickvol' in df.columns:
                df = df.rename(columns={'tickvol': 'tick_volume'})
        else:
            df = pd.read_csv(filename)
            for col in ['time', 'datetime', 'Date', 'Time']:
                if col in df.columns:
                    df['time'] = pd.to_datetime(df[col])
                    break
        
        if 'time' in df.columns:
            df = df.set_index('time')
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================

def engineer_features(df):
    df = df.copy()
    
    # 1. Indicators
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd_hist'] = ema12 - ema26
    
    # Efficiency Ratio
    change = df['close'].diff(10).abs()
    volatility = df['close'].diff().abs().rolling(10).sum()
    df['efficiency_ratio'] = change / volatility
    
    # ADX
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    trs = tr.rolling(14).sum().replace(0, np.nan)
    pdm = pd.Series(plus_dm, index=df.index).rolling(14).sum()
    mdm = pd.Series(minus_dm, index=df.index).rolling(14).sum()
    
    plus_di = 100 * (pdm / trs)
    minus_di = 100 * (mdm / trs)
    dx = 100 * np.abs((plus_di - minus_di) / (plus_di + minus_di))
    df['adx'] = dx.rolling(14).mean()
    
    # Volatility Regime
    df['vol_regime'] = df['log_return'].rolling(24).std()
    
    # Cyclical Time
    if 'time' in df.columns:
        df['hour_sin'] = np.sin(2 * np.pi * df['time'].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['time'].dt.hour / 24)
        df = df.set_index('time')
    else:
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    
    # Lags
    cols_to_lag = ['log_return', 'rsi', 'tick_volume', 'efficiency_ratio', 'adx']
    for col in cols_to_lag:
        if col in df.columns:
            for lag in [1, 2, 3, 5, 24]:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
    return df

# ==========================================
# 4. LABELING
# ==========================================

def triple_barrier_labeling(df, width=1.5, horizon=24):
    print("Applying Triple Barrier Labeling...")
    data = df.copy()
    
    data['prev_close'] = data['close'].shift(1)
    data['tr'] = np.maximum(data['high'] - data['low'], np.abs(data['high'] - data['prev_close']))
    data['atr'] = data['tr'].rolling(window=14).mean()
    data['volatility_threshold'] = data['atr'] * width
    
    highs = data['high'].values
    lows = data['low'].values
    closes = data['close'].values
    thresholds = data['volatility_threshold'].values
    
    labels = []
    length = len(data)
    
    for i in range(length - horizon):
        current_price = closes[i]
        vol = thresholds[i]
        if np.isnan(vol):
            labels.append(np.nan)
            continue
            
        upper = current_price + (vol * RISK_REWARD_RATIO)
        lower = current_price - vol
        
        hit_upper = np.where(highs[i+1 : i+horizon+1] >= upper)[0]
        hit_lower = np.where(lows[i+1 : i+horizon+1] <= lower)[0]
        
        first_upper = hit_upper[0] if len(hit_upper) > 0 else 999
        first_lower = hit_lower[0] if len(hit_lower) > 0 else 999
        
        if first_upper < first_lower: labels.append(1)
        elif first_lower < first_upper: labels.append(0)
        else: labels.append(np.nan)
            
    labels.extend([np.nan] * horizon)
    data['target'] = labels 
    return data

# ==========================================
# 5. EXECUTION
# ==========================================

if __name__ == "__main__":
    print("=== STARTING PRODUCTION PIPELINE ===")
    
    # 1. Load Data
    df = get_data_from_mt5(start_year=2010, end_year=2026)
    if df is not None:
        df.to_csv("EURUSD_fresh_export.csv", index=False)
    else:
        print("MT5 not active. Searching local...")
        df = find_and_load_local_csv()
        
    if df is None:
        print("CRITICAL ERROR: No data found.")
        exit()
        
    # 2. Process
    print(f"Data Loaded: {len(df)} rows.")
    df = engineer_features(df)
    df = triple_barrier_labeling(df, width=BARRIER_WIDTH, horizon=TIME_HORIZON)
    df_clean = df.dropna(subset=['target', 'rsi', 'efficiency_ratio', 'adx']).copy()
    
    # Select Features
    exclude = ['open', 'high', 'low', 'close', 'tick_volume', 'target', 'prev_close', 'tr', 'atr', 'volatility_threshold']
    features = [c for c in df_clean.columns if c not in exclude]
    X = df_clean[features]
    y = df_clean['target']
    
    # --- PHASE 1: VALIDATION (Check health) ---
    print("\n--- Phase 1: Validation Check (80/20 Split) ---")
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    neg, pos = np.sum(y_train == 0), np.sum(y_train == 1)
    scale_weight = neg / pos if pos > 0 else 1.0
    
    val_model = XGBClassifier(
        n_estimators=500, learning_rate=0.015, max_depth=5, 
        subsample=0.8, colsample_bytree=0.8, gamma=0.2, min_child_weight=7,      
        scale_pos_weight=scale_weight, random_state=42, n_jobs=-1
    )
    val_model.fit(X_train, y_train)
    
    probs = val_model.predict_proba(X_test)[:, 1]
    trades = (probs > TRADE_THRESHOLD).astype(int)
    mask = probs > TRADE_THRESHOLD
    
    if mask.sum() > 0:
        acc = accuracy_score(y_test[mask], trades[mask])
        print(f"Validation Accuracy on Taken Trades: {acc:.2%}")
        wins = np.sum(trades[mask] == y_test[mask])
        losses = mask.sum() - wins
        net = (wins * RISK_REWARD_RATIO) - (losses * 1.0)
        print(f"Validation Net Profit: {net:.2f} R")
    else:
        print("Validation: No trades taken (System too conservative?)")

    # --- PHASE 2: PRODUCTION TRAINING ---
    print("\n--- Phase 2: Training Master Model (100% Data) ---")
    neg_full, pos_full = np.sum(y == 0), np.sum(y == 1)
    scale_weight_full = neg_full / pos_full if pos_full > 0 else 1.0
    
    prod_model = XGBClassifier(
        n_estimators=500, learning_rate=0.015, max_depth=5, 
        subsample=0.8, colsample_bytree=0.8, gamma=0.2, min_child_weight=7,      
        scale_pos_weight=scale_weight_full, random_state=42, n_jobs=-1
    )
    prod_model.fit(X, y)
    
    prod_model.save_model("xgb_model_final.json")
    print("\nSUCCESS: Model saved as 'xgb_model_final.json'")
    print("Status: READY FOR LIVE TRADING.")
    
    