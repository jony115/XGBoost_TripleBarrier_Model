# Project Overview

This project implements a sophisticated Machine Learning pipeline designed to predict high-probability trading setups in the Forex market (specifically EUR/USD). Moving beyond simple "price-up/price-down" classification, this system utilizes Marcos López de Prado’s Triple Barrier Method to align model predictions with professional risk management parameters.

Key Highlights:
Dynamic Labeling: Replaces fixed-horizon labels with volatility-adjusted profit and loss barriers.
Feature Engineering: Includes Fourier-style temporal encoding (Sin/Cos) and Kaufman’s Efficiency Ratio.
Production-Ready: Integrated with the MetaTrader 5 API for real-time data ingestion.

High-Level Architecture
The system operates as a multi-stage pipeline:
Data Acquisition: Automated ingestion from MT5 or local CSV fail-safes.
Feature Engineering: Transformation of non-stationary price data into 30+ statistical features.
Labeling: Application of the Triple Barrier Method using a 1.2x Risk/Reward ratio.
Training: Gradient Boosted Trees (XGBoost) with hyperparameter tuning to prevent overfitting.
Validation: Out-of-sample testing using a 60% confidence threshold for trade execution.

The "Triple Barrier" Logic
Traditional models fail because they predict price at a fixed time t+n. This model predicts outcomes:
Barrier 1 (Take Profit): 1.2×ATR
Barrier 2 (Stop Loss): 1.0×ATR
Barrier 3 (Time Expiry): 24-hour horizontal window.

Engineered Features
To give the model a statistical edge, we engineered features that capture market "context" rather than just price:
Stationarity: Log Returns for stable mean/variance.
Temporal Cycles: Hour-of-day encoded as Sine/Cosine waves to capture session liquidity.
Efficiency Ratio: Measures the "cleanliness" of a trend vs. market noise.
Lagged Momentum: T-1 through T-24 lags for RSI, ADX, and Volume.

Model Configuration (XGBoost)
We optimized for precision over recall, ensuring the model only fires on high-conviction setups.
Learning Rate: 0.015 (Slow, robust convergence)
Max Depth: 5 (To prevent memorization/overfitting)
Scale Pos Weight: Dynamic balancing to handle rare "Buy" signals.
Validation: Walk-forward check on the final 20% of unseen data.
