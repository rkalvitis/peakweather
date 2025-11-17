# Metrics Reference Guide

This document explains what each metric means and where to find their definitions in the codebase.

## Metric Definitions Location

All metrics are defined in the `lib/metrics/` directory:

- **Point prediction metrics**: `lib/metrics/point_predictions.py`
- **Sample-based metrics**: `lib/metrics/sample_metrics.py`
- **Energy Score (CRPS)**: `lib/metrics/crps.py` and `lib/metrics/functional/crps.py`
- **Wind-specific metrics**: `lib/metrics/wind.py`
- **Functional implementations**: `lib/metrics/functional/sampling.py`

## Temperature Forecasting Metrics

For temperature forecasting, you'll see these metrics:

### Point Prediction Metrics

#### **MAE (Mean Absolute Error)**
- **File**: Uses `torch_metrics.MaskedMAE()` from TSL library
- **Definition**: Average absolute difference between predictions and actual values
- **Formula**: `MAE = mean(|predicted - actual|)`
- **Units**: Same as target (e.g., °C for temperature)
- **Interpretation**: Lower is better. Represents average error magnitude.
- **Example**: `test_mae = 1.69` means average error of 1.69°C

#### **MSE (Mean Squared Error)**
- **File**: Uses `torch_metrics.MaskedMSE()` from TSL library
- **Definition**: Average squared difference between predictions and actual values
- **Formula**: `MSE = mean((predicted - actual)²)`
- **Units**: Squared units (e.g., °C² for temperature)
- **Interpretation**: Lower is better. Penalizes large errors more than MAE.
- **Example**: `test_mse = 5.40` means average squared error of 5.40°C²

#### **MAE/MSE at specific horizons**
- **Format**: `mae_1h`, `mae_3h`, `mae_6h`, `mae_12h`, `mae_18h`, `mae_24h`
- **Definition**: MAE/MSE computed only at that specific forecast horizon
- **Interpretation**: Shows how error increases with forecast lead time
- **Example**: `test_mae_1h = 0.77` (1 hour ahead) vs `test_mae_24h = 2.22` (24 hours ahead)

### Probabilistic/Sample-Based Metrics

These metrics evaluate the full predictive distribution, not just point predictions.

#### **SMAE (Sample Mean Absolute Error)**
- **File**: `lib/metrics/point_predictions.py` → `SampleMAE`
- **Implementation**: `lib/metrics/functional/sampling.py` → `sampling_MAE()`
- **Definition**: MAE computed using the **median** of predictive samples
- **Formula**: `SMAE = mean(|median(samples) - actual|)`
- **Interpretation**: Lower is better. Evaluates point prediction from probabilistic model.
- **Note**: Uses median of samples, not mean (more robust to outliers)

#### **SMSE (Sample Mean Squared Error)**
- **File**: `lib/metrics/point_predictions.py` → `SampleMSE`
- **Implementation**: `lib/metrics/functional/sampling.py` → `sampling_MSE()`
- **Definition**: MSE computed using the **mean** of predictive samples
- **Formula**: `SMSE = mean((mean(samples) - actual)²)`
- **Interpretation**: Lower is better. Evaluates point prediction from probabilistic model.

#### **ENS (Energy Score)**
- **File**: `lib/metrics/crps.py` → `EnergyScore`
- **Implementation**: `lib/metrics/functional/crps.py` → `energy_score()`
- **Definition**: Multivariate generalization of CRPS (Continuous Ranked Probability Score)
- **Formula**: 
  ```
  ES(F,y) = E(||X-y||) - 0.5 E(||X-X'||)
           X~F              X,X'~F
  ```
  Where:
  - First term: Expected distance from samples to target
  - Second term: Expected distance between sample pairs (measures spread)
- **Interpretation**: Lower is better. Proper scoring rule that evaluates both:
  - **Calibration**: How well the distribution matches reality
  - **Sharpness**: How concentrated the distribution is
- **Units**: Same as target (e.g., °C for temperature)
- **Example**: `test_ens = 1.22` means good probabilistic forecast quality

#### **ENS at specific horizons**
- **Format**: `ens_1h`, `ens_3h`, `ens_6h`, `ens_12h`, `ens_18h`, `ens_24h`
- **Definition**: Energy Score computed only at that specific forecast horizon
- **Interpretation**: Shows how probabilistic forecast quality changes with lead time

## Understanding Your Results

Based on your test results:

```
test_mae = 1.69°C          # Average absolute error
test_mae_1h = 0.77°C       # 1-hour ahead forecast error
test_mae_24h = 2.22°C      # 24-hour ahead forecast error
test_ens = 1.22°C          # Probabilistic forecast quality
```

### Key Insights:
1. **Error increases with horizon**: 0.77°C at 1h → 2.22°C at 24h (expected)
2. **ENS < MAE**: Energy Score (1.22) is lower than MAE (1.69), suggesting the probabilistic model is well-calibrated
3. **Good short-term accuracy**: 1-hour forecast has <1°C error

## Metric Comparison

| Metric | Type | Evaluates | Best For |
|--------|------|-----------|----------|
| MAE | Point | Single prediction | Simple error assessment |
| MSE | Point | Single prediction | Penalizing large errors |
| SMAE | Sample | Distribution (median) | Point prediction from samples |
| SMSE | Sample | Distribution (mean) | Point prediction from samples |
| ENS | Sample | Full distribution | Probabilistic forecast quality |

## Code References

- **TSL Metrics**: `tsl.metrics.torch_metrics` (MaskedMAE, MaskedMSE)
- **Custom Metrics**: `lib.metrics` module
- **Functional**: `lib/metrics/functional/` directory

## Further Reading

- Energy Score paper: Gneiting & Raftery (2007) "Strictly Proper Scoring Rules, Prediction, and Estimation"
- CRPS: Continuous Ranked Probability Score (univariate Energy Score)
- Proper scoring rules: Metrics that encourage honest probabilistic forecasts

