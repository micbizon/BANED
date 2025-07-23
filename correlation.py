from typing import Any, Tuple

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from uncertainty_toolbox.utils import (
    assert_is_flat_same_shape,
)

# import results files
MC_II = np.load("cnn_test_probabilities_monte_carlo_dp_0.2.npy")
MC_III = np.load("cnn_test_probabilities_monte_carlo_dp_0.3.npy")
MC_IV = np.load("cnn_test_probabilities_monte_carlo_dp_0.4.npy")
MC_V = np.load("cnn_test_probabilities_monte_carlo_dp_0.5.npy")

CNN_prob = np.load("cnn_test_probabilities.npy")

KB_prob = np.load("cnn_test_probabilities_kb_dropout.npy")
KB_I_prob = np.load("cnn_test_probabilities_kb_dropout_0_1.npy")
KB_II_prob = np.load("cnn_test_probabilities_kb_dropout_0_2.npy")
KB_III_prob = np.load("cnn_test_probabilities_kb_dropout_0_3.npy")
KB_IV_prob = np.load("cnn_test_probabilities_kb_dropout_0_4.npy")

MC_II_KB_I = np.load("cnn_test_probabilities_monte_carlo_dp_0.2_kb_dropout_0_1.npy")
MC_II_KB_II = np.load("cnn_test_probabilities_monte_carlo_dp_0.2_kb_dropout_0_2.npy")
MC_II_KB_III = np.load("cnn_test_probabilities_monte_carlo_dp_0.2_kb_dropout_0_3.npy")
MC_II_KB_IV = np.load("cnn_test_probabilities_monte_carlo_dp_0.2_kb_dropout_0_4.npy")

MC_III_KB_I = np.load("cnn_test_probabilities_monte_carlo_dp_0.3_kb_dropout_0_1.npy")
MC_III_KB_II = np.load("cnn_test_probabilities_monte_carlo_dp_0.3_kb_dropout_0_2.npy")
MC_III_KB_III = np.load("cnn_test_probabilities_monte_carlo_dp_0.3_kb_dropout_0_3.npy")
MC_III_KB_IV = np.load("cnn_test_probabilities_monte_carlo_dp_0.3_kb_dropout_0_4.npy")

MC_IV_KB_I = np.load("cnn_test_probabilities_monte_carlo_dp_0.4_kb_dropout_0_1.npy")
MC_IV_KB_II = np.load("cnn_test_probabilities_monte_carlo_dp_0.4_kb_dropout_0_2.npy")
MC_IV_KB_III = np.load("cnn_test_probabilities_monte_carlo_dp_0.4_kb_dropout_0_3.npy")
MC_IV_KB_IV = np.load("cnn_test_probabilities_monte_carlo_dp_0.4_kb_dropout_0_4.npy")

MC_V_KB_I = np.load("cnn_test_probabilities_monte_carlo_dp_0.5_kb_dropout_0_1.npy")
MC_V_KB_II = np.load("cnn_test_probabilities_monte_carlo_dp_0.5_kb_dropout_0_2.npy")
MC_V_KB_III = np.load("cnn_test_probabilities_monte_carlo_dp_0.5_kb_dropout_0_3.npy")
MC_V_KB_IV = np.load("cnn_test_probabilities_monte_carlo_dp_0.5_kb_dropout_0_4.npy")

y_std_mc_II = np.load("cnn_test_pred_std_monte_carlo_dp_0.2.npy").flatten()
y_std_mc_III = np.load("cnn_test_pred_std_monte_carlo_dp_0.3.npy").flatten()
y_std_mc_IV = np.load("cnn_test_pred_std_monte_carlo_dp_0.4.npy").flatten()
y_std_mc_V = np.load("cnn_test_pred_std_monte_carlo_dp_0.5.npy").flatten()

y_std_kb = np.load("cnn_test_pred_std_kb_dropout.npy").flatten()
y_std_kb_I = np.load("cnn_test_pred_std_kb_dropout_0_1.npy").flatten()
y_std_kb_II = np.load("cnn_test_pred_std_kb_dropout_0_2.npy").flatten()
y_std_kb_III = np.load("cnn_test_pred_std_kb_dropout_0_3.npy").flatten()
y_std_kb_IV = np.load("cnn_test_pred_std_kb_dropout_0_4.npy").flatten()

y_std_mc_II_kb_1 = np.load(
    "cnn_test_pred_std_monte_carlo_dp_0.2_kb_dropout_0_1.npy"
).flatten()
y_std_mc_II_kb_2 = np.load(
    "cnn_test_pred_std_monte_carlo_dp_0.2_kb_dropout_0_2.npy"
).flatten()
y_std_mc_II_kb_3 = np.load(
    "cnn_test_pred_std_monte_carlo_dp_0.2_kb_dropout_0_3.npy"
).flatten()
y_std_mc_II_kb_4 = np.load(
    "cnn_test_pred_std_monte_carlo_dp_0.2_kb_dropout_0_4.npy"
).flatten()

y_std_mc_III_kb_1 = np.load(
    "cnn_test_pred_std_monte_carlo_dp_0.3_kb_dropout_0_1.npy"
).flatten()
y_std_mc_III_kb_2 = np.load(
    "cnn_test_pred_std_monte_carlo_dp_0.3_kb_dropout_0_2.npy"
).flatten()
y_std_mc_III_kb_3 = np.load(
    "cnn_test_pred_std_monte_carlo_dp_0.3_kb_dropout_0_3.npy"
).flatten()
y_std_mc_III_kb_4 = np.load(
    "cnn_test_pred_std_monte_carlo_dp_0.3_kb_dropout_0_4.npy"
).flatten()

y_std_mc_IV_kb_1 = np.load(
    "cnn_test_pred_std_monte_carlo_dp_0.4_kb_dropout_0_1.npy"
).flatten()
y_std_mc_IV_kb_2 = np.load(
    "cnn_test_pred_std_monte_carlo_dp_0.4_kb_dropout_0_2.npy"
).flatten()
y_std_mc_IV_kb_3 = np.load(
    "cnn_test_pred_std_monte_carlo_dp_0.4_kb_dropout_0_3.npy"
).flatten()
y_std_mc_IV_kb_4 = np.load(
    "cnn_test_pred_std_monte_carlo_dp_0.4_kb_dropout_0_4.npy"
).flatten()

y_std_mc_V_kb_1 = np.load(
    "cnn_test_pred_std_monte_carlo_dp_0.5_kb_dropout_0_1.npy"
).flatten()
y_std_mc_V_kb_2 = np.load(
    "cnn_test_pred_std_monte_carlo_dp_0.5_kb_dropout_0_2.npy"
).flatten()
y_std_mc_V_kb_3 = np.load(
    "cnn_test_pred_std_monte_carlo_dp_0.5_kb_dropout_0_3.npy"
).flatten()
y_std_mc_V_kb_4 = np.load(
    "cnn_test_pred_std_monte_carlo_dp_0.5_kb_dropout_0_4.npy"
).flatten()

# KB_file = pd.read_csv('results.csv')
# KB_fake = KB_file['results_fake'].tolist()
# KB_real = KB_file['results_real'].tolist()


# pair labels with indexes (labels_all does not include the headers!)
# one is real, 0 is fake
labels_all = [1] * (len(pd.read_csv("fnn_real_clean.csv")) - 1) + [0] * (
    len(pd.read_csv("fnn_fake_clean.csv")) - 1
)

indices = np.load("cnn_test_indices.npy")
labels_test = []

for number in indices:
    labels_test.append(labels_all[number])
labels_arr = np.array(labels_test)


# check if indices of MCMC and KB are the same
# print(len(KB_prob))
# print(len(MCMC))
# print(len(labels_test))

# predictions_mcmc =  []
# predictions_kb = []
# n=0

# for label in labels_test:
#    predictions_mcmc.append(MCMC[n,0])
#    predictions_kb.append(KB_fake[n])
#    if label == 0:
#    if CNN_prob[n,0] >= [0.5]:
#        predictions_kb.append(KB_real[n])
#        predictions_kb.append(KB_fake[n])
#    else:
#        predictions_kb.append(KB_real[n])
#        predictions_kb.append(KB_fake[n])
#    n=n+1


# sharpness
def sharpness(y_std: np.ndarray) -> float:
    """Return sharpness (a single measure of the overall confidence)."""
    assert y_std.ndim == 1, "Input array must be 1D"
    assert np.all(y_std >= 0), "All standard deviations must be positive"
    sharp_metric = np.sqrt(np.mean(y_std**2))
    return sharp_metric


# Calculate sharpness for both models
sharp_metric_mc = sharpness(y_std_mc_II)
sharp_metric_mc_III = sharpness(y_std_mc_III)
sharp_metric_mc_IV = sharpness(y_std_mc_IV)
sharp_metric_mc_V = sharpness(y_std_mc_V)
print(f"Sharpness Monte Carlo 0.2 : {sharp_metric_mc}")
print(f"Sharpness Monte Carlo 0.3 : {sharp_metric_mc_III}")
print(f"Sharpness Monte Carlo 0.4 : {sharp_metric_mc_IV}")
print(f"Sharpness Monte Carlo 0.5 : {sharp_metric_mc_V}")

sharp_metric_kb_I = sharpness(y_std_kb_I)
sharp_metric_kb_II = sharpness(y_std_kb_II)
sharp_metric_kb_III = sharpness(y_std_kb_III)
sharp_metric_kb_IV = sharpness(y_std_kb_IV)
print(f"Sharpness Knowledge Base Monte Carlo: {sharp_metric_kb_I}")
print(f"Sharpness Knowledge Base Monte Carlo: {sharp_metric_kb_II}")
print(f"Sharpness Knowledge Base Monte Carlo: {sharp_metric_kb_III}")
print(f"Sharpness Knowledge Base Monte Carlo: {sharp_metric_kb_IV}")

sharp_metric_mc_II_kb_I = sharpness(y_std_mc_II_kb_1)
sharp_metric_mc_II_kb_II = sharpness(y_std_mc_II_kb_2)
sharp_metric_mc_II_kb_III = sharpness(y_std_mc_II_kb_3)
sharp_metric_mc_II_kb_IV = sharpness(y_std_mc_II_kb_4)
print(f"Sharpness Monte Carlo 0.2 with Knowledge Base 0.1: {sharp_metric_mc_II_kb_I}")
print(f"Sharpness Monte Carlo 0.2 with Knowledge Base 0.2: {sharp_metric_mc_II_kb_II}")
print(f"Sharpness Monte Carlo 0.2 with Knowledge Base 0.3: {sharp_metric_mc_II_kb_III}")
print(f"Sharpness Monte Carlo 0.2 with Knowledge Base 0.4: {sharp_metric_mc_II_kb_IV}")

sharp_metric_mc_III_kb_I = sharpness(y_std_mc_III_kb_1)
sharp_metric_mc_III_kb_II = sharpness(y_std_mc_III_kb_2)
sharp_metric_mc_III_kb_III = sharpness(y_std_mc_III_kb_3)
sharp_metric_mc_III_kb_IV = sharpness(y_std_mc_III_kb_4)
print(f"Sharpness Monte Carlo 0.3 with Knowledge Base 0.1: {sharp_metric_mc_III_kb_I}")
print(f"Sharpness Monte Carlo 0.3 with Knowledge Base 0.2: {sharp_metric_mc_III_kb_II}")
print(
    f"Sharpness Monte Carlo 0.3 with Knowledge Base 0.3: {sharp_metric_mc_III_kb_III}"
)
print(f"Sharpness Monte Carlo 0.3 with Knowledge Base 0.4: {sharp_metric_mc_III_kb_IV}")

sharp_metric_mc_IV_kb_I = sharpness(y_std_mc_IV_kb_1)
sharp_metric_mc_IV_kb_II = sharpness(y_std_mc_IV_kb_2)
sharp_metric_mc_IV_kb_III = sharpness(y_std_mc_IV_kb_3)
sharp_metric_mc_IV_kb_IV = sharpness(y_std_mc_IV_kb_4)
print(f"Sharpness Monte Carlo 0.4 with Knowledge Base 0.1: {sharp_metric_mc_IV_kb_I}")
print(f"Sharpness Monte Carlo 0.4 with Knowledge Base 0.2: {sharp_metric_mc_IV_kb_II}")
print(f"Sharpness Monte Carlo 0.4 with Knowledge Base 0.3: {sharp_metric_mc_IV_kb_III}")
print(f"Sharpness Monte Carlo 0.4 with Knowledge Base 0.4: {sharp_metric_mc_IV_kb_IV}")

sharp_metric_mc_V_kb_I = sharpness(y_std_mc_V_kb_1)
sharp_metric_mc_V_kb_II = sharpness(y_std_mc_V_kb_2)
sharp_metric_mc_V_kb_III = sharpness(y_std_mc_V_kb_3)
sharp_metric_mc_V_kb_IV = sharpness(y_std_mc_V_kb_4)
print(f"Sharpness Monte Carlo 0.5 with Knowledge Base 0.1: {sharp_metric_mc_V_kb_I}")
print(f"Sharpness Monte Carlo 0.5 with Knowledge Base 0.2: {sharp_metric_mc_V_kb_II}")
print(f"Sharpness Monte Carlo 0.5 with Knowledge Base 0.3: {sharp_metric_mc_V_kb_III}")
print(f"Sharpness Monte Carlo 0.5 with Knowledge Base 0.4: {sharp_metric_mc_V_kb_IV}")


# Calibration
def root_mean_squared_calibration_error(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    num_bins: int = 100,
    vectorized: bool = False,
    recal_model: IsotonicRegression = None,
    prop_type: str = "interval",
) -> float:
    # Check that input arrays are flat
    assert_is_flat_same_shape(y_pred, y_std, y_true)
    # Check that input std is positive
    # assert_is_positive(y_std)
    # Check that prop_type is one of 'interval' or 'quantile'
    assert prop_type in ["interval", "quantile"]

    # Get lists of expected and observed proportions for a range of quantiles
    if vectorized:
        (exp_proportions, obs_proportions) = get_proportion_lists_vectorized(
            y_pred, y_std, y_true, num_bins, recal_model, prop_type
        )
    else:
        (exp_proportions, obs_proportions) = get_proportion_lists(
            y_pred, y_std, y_true, num_bins, recal_model, prop_type
        )

    squared_diff_proportions = np.square(exp_proportions - obs_proportions)
    rmsce = np.sqrt(np.mean(squared_diff_proportions))

    return rmsce


def get_proportion_lists(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    num_bins: int = 100,
    recal_model: IsotonicRegression = None,
    prop_type: str = "interval",
) -> Tuple[np.ndarray, np.ndarray]:
    # Check that input arrays are flat
    assert_is_flat_same_shape(y_pred, y_std, y_true)
    # Check that input std is positive
    # assert_is_positive(y_std)
    # Check that prop_type is one of 'interval' or 'quantile'
    assert prop_type in ["interval", "quantile"]

    # Compute proportions
    exp_proportions = np.linspace(0, 1, num_bins)
    # If we are recalibrating, input proportions are recalibrated proportions
    if recal_model is not None:
        in_exp_proportions = recal_model.predict(exp_proportions)
    else:
        in_exp_proportions = exp_proportions

    if prop_type == "interval":
        obs_proportions = [
            get_proportion_in_interval(y_pred, y_std, y_true, quantile)
            for quantile in in_exp_proportions
        ]
    elif prop_type == "quantile":
        obs_proportions = [
            get_proportion_under_quantile(y_pred, y_std, y_true, quantile)
            for quantile in in_exp_proportions
        ]

    return exp_proportions, obs_proportions


def get_proportion_lists_vectorized(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    num_bins: int = 100,
    recal_model: Any = None,
    prop_type: str = "interval",
) -> Tuple[np.ndarray, np.ndarray]:
    """Arrays of expected and observed proportions

    Returns the expected proportions and observed proportion of points falling into
    intervals corresponding to a range of quantiles.
    Computations here are vectorized for faster execution, but this function is
    not suited when there are memory constraints.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        num_bins: number of discretizations for the probability space [0, 1].
        recal_model: an sklearn isotonic regression model which recalibrates the predictions.
        prop_type: "interval" to measure observed proportions for centered prediction intervals,
                   and "quantile" for observed proportions below a predicted quantile.

    Returns:
        A tuple of two numpy arrays, expected proportions and observed proportions

    """

    # Check that input arrays are flat
    assert_is_flat_same_shape(y_pred, y_std, y_true)
    # Check that input std is positive
    # assert_is_positive(y_std)
    # Check that prop_type is one of 'interval' or 'quantile'
    assert prop_type in ["interval", "quantile"]

    # Compute proportions
    exp_proportions = np.linspace(0, 1, num_bins)
    # If we are recalibrating, input proportions are recalibrated proportions
    if recal_model is not None:
        in_exp_proportions = recal_model.predict(exp_proportions)
    else:
        in_exp_proportions = exp_proportions

    residuals = y_pred - y_true
    normalized_residuals = (residuals.flatten() / y_std.flatten()).reshape(-1, 1)
    norm = stats.norm(loc=0, scale=1)
    if prop_type == "interval":
        gaussian_lower_bound = norm.ppf(0.5 - in_exp_proportions / 2.0)
        gaussian_upper_bound = norm.ppf(0.5 + in_exp_proportions / 2.0)

        above_lower = normalized_residuals >= gaussian_lower_bound
        below_upper = normalized_residuals <= gaussian_upper_bound

        within_quantile = above_lower * below_upper
        obs_proportions = np.sum(within_quantile, axis=0).flatten() / len(residuals)
    elif prop_type == "quantile":
        gaussian_quantile_bound = norm.ppf(in_exp_proportions)
        below_quantile = normalized_residuals <= gaussian_quantile_bound
        obs_proportions = np.sum(below_quantile, axis=0).flatten() / len(residuals)

    return exp_proportions, obs_proportions


def get_proportion_in_interval(
    y_pred: np.ndarray, y_std: np.ndarray, y_true: np.ndarray, quantile: float
) -> float:
    """For a specified quantile, return the proportion of points falling into
    an interval corresponding to that quantile.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        quantile: a specified quantile level

    Returns:
        A single scalar which is the proportion of the true labels falling into the
        prediction interval for the specified quantile.
    """

    # Check that input arrays are flat
    assert_is_flat_same_shape(y_pred, y_std, y_true)
    # Check that input std is positive
    # assert_is_positive(y_std)

    # Computer lower and upper bound for quantile
    norm = stats.norm(loc=0, scale=1)
    lower_bound = norm.ppf(0.5 - quantile / 2)
    upper_bound = norm.ppf(0.5 + quantile / 2)

    # Compute proportion of normalized residuals within lower to upper bound
    residuals = y_pred - y_true

    normalized_residuals = residuals.reshape(-1) / y_std.reshape(-1)

    num_within_quantile = 0
    for resid in normalized_residuals:
        if lower_bound <= resid and resid <= upper_bound:
            num_within_quantile += 1.0
    proportion = num_within_quantile / len(residuals)

    return proportion


def mean_absolute_calibration_error(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    num_bins: int = 100,
    vectorized: bool = False,
    recal_model: IsotonicRegression = None,
    prop_type: str = "interval",
) -> float:
    # Check that input arrays are flat
    assert_is_flat_same_shape(y_pred, y_std, y_true)
    # Check that input std is positive
    # assert_is_positive(y_std)
    # Check that prop_type is one of 'interval' or 'quantile'
    assert prop_type in ["interval", "quantile"]

    # Get lists of expected and observed proportions for a range of quantiles
    if vectorized:
        (exp_proportions, obs_proportions) = get_proportion_lists_vectorized(
            y_pred, y_std, y_true, num_bins, recal_model, prop_type
        )
    else:
        (exp_proportions, obs_proportions) = get_proportion_lists(
            y_pred, y_std, y_true, num_bins, recal_model, prop_type
        )

    abs_diff_proportions = np.abs(exp_proportions - obs_proportions)
    mace = np.mean(abs_diff_proportions)

    return mace


# Calculate callibration
mcmc_y, mcmc_x = calibration_curve(labels_test, MC_II[:, 0], n_bins=12)
kb_y, kb_x = calibration_curve(labels_test, KB_prob[:, 0], n_bins=12)

cal_abs_mc_II = mean_absolute_calibration_error(MC_II[:, 0], y_std_mc_II, labels_arr)
print(f"Calibration error for Monte Carlo 0.2: {cal_abs_mc_II}")

cal_abs_mc_III = mean_absolute_calibration_error(MC_III[:, 0], y_std_mc_III, labels_arr)
print(f"Calibration error for Monte Carlo 0.3: {cal_abs_mc_III}")

cal_abs_mc_IV = mean_absolute_calibration_error(MC_IV[:, 0], y_std_mc_IV, labels_arr)
print(f"Calibration error for Monte Carlo 0.4: {cal_abs_mc_IV}")

cal_abs_mc_V = mean_absolute_calibration_error(MC_V[:, 0], y_std_mc_V, labels_arr)
print(f"Calibration error for Monte Carlo 0.5: {cal_abs_mc_V}")

cal_abs_KB = mean_absolute_calibration_error(KB_prob[:, 0], y_std_kb, labels_arr)
print(f"Calibration error for knowledge base: {cal_abs_KB}")

cal_abs_KB_I = mean_absolute_calibration_error(KB_I_prob[:, 0], y_std_kb_I, labels_arr)
print(f"Calibration error for knowledge base 0.1: {cal_abs_KB_I}")

cal_abs_KB_II = mean_absolute_calibration_error(
    KB_II_prob[:, 0], y_std_kb_II, labels_arr
)
print(f"Calibration error for knowledge base 0.2: {cal_abs_KB_II}")

cal_abs_KB_III = mean_absolute_calibration_error(
    KB_III_prob[:, 0], y_std_kb_III, labels_arr
)
print(f"Calibration error for knowledge base 0.3: {cal_abs_KB_III}")

cal_abs_KB_IV = mean_absolute_calibration_error(
    KB_IV_prob[:, 0], y_std_kb_IV, labels_arr
)
print(f"Calibration error for knowledge base 0.4: {cal_abs_KB_IV}")

cal_abs_MC_II_KB_I = mean_absolute_calibration_error(
    MC_II_KB_I[:, 0], y_std_mc_II_kb_1, labels_arr
)
cal_abs_MC_II_KB_II = mean_absolute_calibration_error(
    MC_II_KB_II[:, 0], y_std_mc_II_kb_2, labels_arr
)
cal_abs_MC_II_KB_III = mean_absolute_calibration_error(
    MC_II_KB_III[:, 0], y_std_mc_II_kb_3, labels_arr
)
cal_abs_MC_II_KB_IV = mean_absolute_calibration_error(
    MC_II_KB_IV[:, 0], y_std_mc_II_kb_4, labels_arr
)
print(
    f"Calibration error for Monte Carlo 0.2 and knowledge base: {cal_abs_MC_II_KB_I}"
    f"{cal_abs_MC_II_KB_II}"
    f"{cal_abs_MC_II_KB_III}"
    f"{cal_abs_MC_II_KB_IV}"
)

cal_abs_MC_III_KB_I = mean_absolute_calibration_error(
    MC_III_KB_I[:, 0], y_std_mc_III_kb_1, labels_arr
)
cal_abs_MC_III_KB_II = mean_absolute_calibration_error(
    MC_III_KB_II[:, 0], y_std_mc_III_kb_2, labels_arr
)
cal_abs_MC_III_KB_III = mean_absolute_calibration_error(
    MC_III_KB_III[:, 0], y_std_mc_III_kb_3, labels_arr
)
cal_abs_MC_III_KB_IV = mean_absolute_calibration_error(
    MC_III_KB_IV[:, 0], y_std_mc_III_kb_4, labels_arr
)
print(
    f"Calibration error for Monte Carlo 0.3 and knowledge base: {cal_abs_MC_III_KB_I}"
    f"{cal_abs_MC_III_KB_II}"
    f"{cal_abs_MC_III_KB_III}"
    f"{cal_abs_MC_III_KB_IV}"
)

cal_abs_MC_IV_KB_I = mean_absolute_calibration_error(
    MC_IV_KB_I[:, 0], y_std_mc_IV_kb_1, labels_arr
)
cal_abs_MC_IV_KB_II = mean_absolute_calibration_error(
    MC_IV_KB_II[:, 0], y_std_mc_IV_kb_2, labels_arr
)
cal_abs_MC_IV_KB_III = mean_absolute_calibration_error(
    MC_IV_KB_III[:, 0], y_std_mc_IV_kb_3, labels_arr
)
cal_abs_MC_IV_KB_IV = mean_absolute_calibration_error(
    MC_IV_KB_IV[:, 0], y_std_mc_IV_kb_4, labels_arr
)
print(
    f"Calibration error for Monte Carlo 0.3 and knowledge base: {cal_abs_MC_IV_KB_I}"
    f"{cal_abs_MC_IV_KB_II}"
    f"{cal_abs_MC_IV_KB_III}"
    f"{cal_abs_MC_IV_KB_IV}"
)

cal_abs_MC_V_KB_I = mean_absolute_calibration_error(
    MC_V_KB_I[:, 0], y_std_mc_V_kb_1, labels_arr
)
cal_abs_MC_V_KB_II = mean_absolute_calibration_error(
    MC_V_KB_II[:, 0], y_std_mc_V_kb_2, labels_arr
)
cal_abs_MC_V_KB_III = mean_absolute_calibration_error(
    MC_V_KB_III[:, 0], y_std_mc_V_kb_3, labels_arr
)
cal_abs_MC_V_KB_IV = mean_absolute_calibration_error(
    MC_V_KB_IV[:, 0], y_std_mc_V_kb_4, labels_arr
)
print(
    f"Calibration error for Monte Carlo 0.3 and knowledge base: {cal_abs_MC_V_KB_I}"
    f"{cal_abs_MC_V_KB_II}"
    f"{cal_abs_MC_V_KB_III}"
    f"{cal_abs_MC_V_KB_IV}"
)

fig, ax = plt.subplots()
# only these two lines are calibration curves
plt.plot(mcmc_x, mcmc_y, marker="o", linewidth=1, label="MC")
plt.plot(kb_x, kb_y, marker="o", linewidth=1, label="KB")
# plt.plot(cnn_x, cnn_y, marker='o', linewidth=1, label='CNN')

# reference line, legends, and axis labels
line = mlines.Line2D([0, 1], [0, 1], color="black")
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
fig.suptitle("Calibration plot for Fake News detection")
ax.set_xlabel("Predicted probability")
ax.set_ylabel("True probability in each bin")
plt.legend()
plt.show()

# Plotting the distribution of standard deviations
plt.figure(figsize=(12, 6))

# Histogram for Model 1
plt.subplot(1, 2, 1)
plt.hist(y_std_mc, bins=20, color="blue", edgecolor="black", alpha=0.7, label="MC")
plt.hist(y_std_kbmc, bins=20, color="green", edgecolor="black", alpha=0.7, label="KB")
plt.title("Distribution of Standard Deviations")
plt.xlabel("Standard Deviation")
plt.ylabel("Frequency")
plt.legend()

# Box plot for both models
plt.subplot(1, 2, 2)
plt.boxplot(
    [y_std_mc, y_std_kbmc], vert=False, patch_artist=True, tick_labels=["MC", "KB"]
)
plt.title("Box Plot of Standard Deviations")
plt.xlabel("Standard Deviation")

plt.tight_layout()
plt.show()

# print(f"Mean for MCMC: {mean_mcmc}")
# print(f"Mean for KB: {mean_kb}")

# Uncertainty for Monte Carlo: 0.7155846691735436
# Uncertainty for KB: 0.02531613895076692 for depending on the label, 0.0051147184423575055 depending on prediction,
#   for fake KB: 0.1074494955923054 and real KB: 0.03316469670916891
# Pearson Correlation Coefficient for Markov Chain Monte Carlo: 0.5405843098121953
# Pearson Correlation Coefficient for Knowledge Base depending on the results: -0.08490549921492696
# Pearson Correlation Coefficient for Knowledge Base depending on the label: -0.31762634757759195
# Pearson Correlation Coefficient for Knowledge Base real: -0.5163704658195758
# Pearson Correlation Coefficient for Knowledge Base fake: 0.4968736370015538
