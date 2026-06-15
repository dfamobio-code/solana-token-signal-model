# Early Pump Prediction for New Solana Memecoins

This project explores whether early token-level data can be used to predict large short-term price movements in newly observed Solana memecoins.

The goal is not to build a production trading system, but to create a clear research workflow for data collection, label construction, baseline modelling, and evaluation.

## Research Question

Can early token features from newly observed Solana tokens be used to predict whether a token will increase by at least 50% after the initial observation period?

## Project Overview

The project collects newly observed Solana token data, tracks those tokens over a follow-up window, and labels each token based on its later price movement. A token is labelled as a pump if its follow-up USD price is at least 50% higher than its initial USD price.

The first baseline model is logistic regression. The model uses only initial snapshot features to avoid future-data leakage.

## Features Used

The baseline model uses the following initial token features:

- Liquidity
- Fully diluted valuation
- Holder count
- Initial USD price
- Initial SOL price
- Circulating supply
- Total supply
- Transaction count
- Whale Dominance Score
- Whale Dominance Score missingness indicator

## Methodology

1. Collect a candidate pool of newly observed Solana tokens.
2. Randomly sample a tracking set.
3. Collect an initial snapshot for each tracked token.
4. Collect a follow-up snapshot approximately 24 hours later.
5. Merge initial and follow-up snapshots by token address.
6. Construct a binary pump label.
7. Train and evaluate a logistic regression baseline.

## Model Evaluation

The notebook evaluates the model using:

- Train/test split
- Stratified cross-validation when class balance allows
- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix
- Prediction probabilities
- Threshold testing
- Error analysis

## Limitations

This is an early baseline and should not be interpreted as a trading-ready model. Important limitations include:

- Small dataset size
- Highly noisy memecoin price behaviour
- Possible class imbalance
- Missing holder or Whale Dominance Score data
- API rate limits affecting collection completeness
- A simple 50% pump threshold
- Logistic regression may be too simple for nonlinear market behaviour

## Next Steps

Future improvements may include:

- Collecting more token observations
- Testing alternative pump thresholds
- Comparing logistic regression against Random Forest and XGBoost
- Adding more liquidity, holder concentration, and transaction-flow features
- Evaluating simulated paper-trading performance
- Ranking tokens by predicted probability rather than only using hard classification