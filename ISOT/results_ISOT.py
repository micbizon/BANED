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
# original
MC_V = np.load("fcl_prob_mc_0.5.npy")
KB_I = np.load("fcl_prob_kb_0_1.npy")
MC_IV_KB_IV = np.load("fcl_prob_mc_0.4_kb_0_4.npy")
MC_V_KB_I = np.load("fcl_prob_mc_0.5_kb_0_1.npy")

y_std_mc_V = np.load("fcl_std_mc_0.5.npy").flatten()
y_std_kb = np.load("fcl_std_kb_0_1.npy").flatten()
y_std_mc_IV_kb_IV = np.load("fcl_std_mc_0.4_kb_0_4.npy").flatten()
y_std_mc_V_kb_I = np.load("fcl_std_mc_0.5_kb_0_1.npy").flatten()

KB_I_pred = np.load("fcl_predictions_kb_0_1.npy")
MC_V_pred = np.load("fcl_predictions_mc_0.5.npy")
MC_IV_KB_IV_pred = np.load("fcl_predictions_mc_0.4_kb_0_4.npy")
MC_V_KB_I_pred = np.load("fcl_predictions_mc_0.5_kb_0_1.npy")

# bp and ensembl
BP_V = np.load("fcl_prob_bp_0.5.npy")
EN = np.load("fcl_prob_ensemble.npy")
BP_V_KB_I = np.load("fcl_prob_bp_0.5_kb_0_1.npy")
KB_I_EN = np.load("fcl_prob_kb_0_1_ensemble.npy")
KB_II_EN = np.load("fcl_prob_kb_0_2_ensemble.npy")

y_std_bp_V = np.load("fcl_std_bp_0.5.npy").flatten()
y_std_en = np.load("fcl_std_ensemble.npy").flatten()
y_std_bp_V_kb_I = np.load("fcl_std_bp_0.5_kb_0_1.npy").flatten()
y_std_kb_I_en = np.load("fcl_std_kb_0_1_ensemble.npy").flatten()
y_std_kb_II_en = np.load("fcl_std_kb_0_2_ensemble.npy").flatten()

BP_V_pred = np.load("fcl_predictions_bp_0.5.npy")
EN_pred = np.load("fcl_predictions_ensemble.npy")
BP_V_KB_I_pred = np.load("fcl_predictions_bp_0.5_kb_0_1.npy")
KB_I_EN_pred = np.load("fcl_predictions_kb_0_1_ensemble.npy")
KB_II_EN_pred = np.load("fcl_predictions_kb_0_2_ensemble.npy")

# pair labels with indexes (labels_all does not include the headers!)
# one is real, 0 is fake
labels_all = [1] * (len(pd.read_csv("real_clean.csv")) - 1) + [0] * (
    len(pd.read_csv("fake_clean.csv")) - 1
)

indices = np.load("fcl_indices.npy")

labels_test = []

for number in indices:
    labels_test.append(labels_all[number])
labels_arr = np.array(labels_test)


# sharpness
def sharpness(y_std: np.ndarray) -> float:
    """Return sharpness (a single measure of the overall confidence)."""
    assert y_std.ndim == 1, "Input array must be 1D"
    assert np.all(y_std >= 0), "All standard deviations must be positive"
    sharp_metric = np.sqrt(np.mean(y_std**2))
    return sharp_metric


def HalfScope(array):
    sorted = np.sort(array)
    # low_index = int(len(array)*0.25)
    # up_index = int(len(array)*0.75)
    # middle_half = sorted[low_index:up_index]
    # scope = np.max(middle_half) - np.min(middle_half)
    bottom_half = sorted[: len(sorted) // 2]
    scope = np.max(bottom_half) - np.min(bottom_half)
    return scope


def avg_threshold_scope(array):
    scope = []
    for item in array:
        predictions = np.array(item).flatten()
        sorted = np.sort(predictions)
        low_index = int(len(item) * 0.25)
        up_index = int(len(item) * 0.75)
        middle_half = sorted[low_index:up_index]
        scope.append(np.max(middle_half) - np.min(middle_half))
    avg_scope = sum(scope) / len(scope)
    return avg_scope


# Calculate sharpness for both models
sharp_metric_mc_V = sharpness(y_std_mc_V)
sharp_scope_mc_V = HalfScope(y_std_mc_V)
print(
    f"Sharpness Monte Carlo 0.5 : {sharp_metric_mc_V} and scope {sharp_scope_mc_V} and average scope {avg_threshold_scope(MC_V_pred)}"
)

sharp_metric_kb_I = sharpness(y_std_kb)
sharp_scope_kb_I = HalfScope(y_std_kb)
print(
    f"Sharpness Knowledge Base Monte Carlo: {sharp_metric_kb_I} and scope {sharp_scope_kb_I} and average scope {avg_threshold_scope(KB_I_pred)}"
)

sharp_metric_mc_IV_kb_IV = sharpness(y_std_mc_IV_kb_IV)
sharp_scope_mc_IV_kb_IV = HalfScope(y_std_mc_IV_kb_IV)
print(
    f"Sharpness Monte Carlo 0.4 with Knowledge Base 0.4: {sharp_metric_mc_IV_kb_IV} and scope {sharp_scope_mc_IV_kb_IV} and average scope {avg_threshold_scope(MC_IV_KB_IV_pred)}"
)

sharp_metric_mc_V_kb_I = sharpness(y_std_mc_V_kb_I)
sharp_scope_mc_V_kb_I = HalfScope(y_std_mc_V_kb_I)
print(
    f"Sharpness Monte Carlo 0.5 with Knowledge Base 0.1: {sharp_metric_mc_V_kb_I} and scope {sharp_scope_mc_V_kb_I} and average scope {avg_threshold_scope(MC_V_KB_I_pred)}"
)

# chosen methods
sharp_metric_bp_V = sharpness(y_std_bp_V)
sharp_scope_bp_V = HalfScope(y_std_bp_V)
print(
    f"Sharpness Bootstrap 0.5 : {sharp_metric_bp_V} and scope {sharp_scope_bp_V} and average scope {avg_threshold_scope(BP_V_pred)}"
)

sharp_metric_en = sharpness(y_std_en)
sharp_scope_en = HalfScope(y_std_en)
print(
    f"Sharpness Ensembl : {sharp_metric_en} and scope {sharp_scope_en} and average scope {avg_threshold_scope(EN_pred)}"
)

sharp_metric_bp_V_kb_I = sharpness(y_std_bp_V_kb_I)
sharp_scope_bp_V_kb_I = HalfScope(y_std_bp_V_kb_I)
print(
    f"Sharpness Bootstrap 0.5 Knowledge Base 0.1 : {sharp_metric_bp_V_kb_I} and scope {sharp_scope_bp_V_kb_I} and average scope {avg_threshold_scope(BP_V_KB_I_pred)}"
)

sharp_metric_kb_I_en = sharpness(y_std_kb_I_en)
sharp_scope_kb_I_en = HalfScope(y_std_kb_I_en)
print(
    f"Sharpness Knowledge Base 0.1 Ensembl : {sharp_metric_kb_I_en} and scope {sharp_scope_kb_I_en} and average scope {avg_threshold_scope(KB_I_EN_pred)}"
)

sharp_metric_kb_II_en = sharpness(y_std_kb_II_en)
sharp_scope_kb_II_en = HalfScope(y_std_kb_II_en)
print(
    f"Sharpness Knowledge Base 0.2 Ensembl : {sharp_metric_kb_II_en} and scope {sharp_scope_kb_II_en} and average scope {avg_threshold_scope(KB_II_EN_pred)}"
)


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


def ece_score(py, y_test, n_bins=10):
    py = np.array(py)
    y_test = np.array(y_test)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    py_index = np.argmax(py, axis=1)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.array(py_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)


# Calculate callibration
# original
cal_ece_mc_V = ece_score(MC_V, labels_all, n_bins=12)
cal_abs_mc_V = mean_absolute_calibration_error(MC_V[:, 0], y_std_mc_V, labels_arr)
print(f"Calibration error for Monte Carlo 0.5: {cal_abs_mc_V} & {cal_ece_mc_V}")

cal_ece_KB_I = ece_score(KB_I, labels_all, n_bins=12)
cal_abs_KB_I = mean_absolute_calibration_error(KB_I[:, 0], y_std_kb, labels_arr)
print(f"Calibration error for knowledge base 0.1: {cal_abs_KB_I} & {cal_ece_KB_I}")

cal_ece_MCKB_IV = ece_score(MC_IV_KB_IV, labels_all, n_bins=12)
cal_abs_MC_IV_KB_IV = mean_absolute_calibration_error(
    MC_IV_KB_IV[:, 0], y_std_mc_IV_kb_IV, labels_arr
)
print(
    f"Calibration error for Monte Carlo 0.4 and knowledge base 0.4: {cal_abs_MC_IV_KB_IV}, {cal_ece_MCKB_IV}"
)

cal_ece_MCKB_V = ece_score(MC_V_KB_I, labels_all, n_bins=12)
cal_abs_MC_V_KB_I = mean_absolute_calibration_error(
    MC_V_KB_I[:, 0], y_std_mc_V_kb_I, labels_arr
)
print(
    f"Calibration error for Monte Carlo 0.5 and knowledge base: {cal_abs_MC_V_KB_I}, {cal_ece_MCKB_V}"
)

# chosen
cal_ece_bp_V = ece_score(BP_V, labels_all, n_bins=12)
cal_abs_bp_V = mean_absolute_calibration_error(BP_V[:, 0], y_std_bp_V, labels_arr)
print(f"Calibration error for Bootstrap 0.5: {cal_abs_bp_V} & {cal_ece_bp_V}")

cal_ece_en = ece_score(EN, labels_all, n_bins=12)
cal_abs_en = mean_absolute_calibration_error(EN[:, 0], y_std_en, labels_arr)
print(f"Calibration error for Ensembl: {cal_abs_en} & {cal_ece_en}")

cal_ece_bp_V_kb_I = ece_score(BP_V_KB_I, labels_all, n_bins=12)
cal_abs_bp_V_kb_I = mean_absolute_calibration_error(
    BP_V_KB_I[:, 0], y_std_bp_V_kb_I, labels_arr
)
print(
    f"Calibration error for Bootstrap 0.5 and knowledge base 0.1: {cal_abs_bp_V_kb_I}, {cal_ece_bp_V_kb_I}"
)

cal_ece_kb_I_en = ece_score(KB_I_EN, labels_all, n_bins=12)
cal_abs_kb_I_en = mean_absolute_calibration_error(
    KB_I_EN[:, 0], y_std_kb_I_en, labels_arr
)
print(
    f"Calibration error for Knowledge base 0.1 and Ensembl: {cal_abs_kb_I_en}, {cal_ece_kb_I_en}"
)

cal_ece_kb_II_en = ece_score(KB_II_EN, labels_all, n_bins=12)
cal_abs_kb_II_en = mean_absolute_calibration_error(
    KB_II_EN[:, 0], y_std_kb_II_en, labels_arr
)
print(
    f"Calibration error for Knowledge base 0.2 and Ensembl: {cal_abs_kb_II_en}, {cal_ece_kb_II_en}"
)

# plots
# original
mc5_y, mc5_x = calibration_curve(labels_test, MC_V[:, 0], n_bins=12)
kb1_y, kb1_x = calibration_curve(labels_test, KB_I[:, 0], n_bins=12)

# chosen
bp5_y, bp5_x = calibration_curve(labels_test, BP_V[:, 0], n_bins=12)
en_y, en_x = calibration_curve(labels_test, EN[:, 0], n_bins=12)

fig, ax = plt.subplots()
# only these two lines are calibration curves
plt.plot(mc5_x, mc5_y, marker=".", linewidth=1, color="tomato", label="MC Dropout 0.5")
plt.plot(
    kb1_x, kb1_y, marker=".", linewidth=1, color="goldenrod", label="KB Dropout 0.1"
)
plt.plot(bp5_x, bp5_y, marker=".", linewidth=1, color="green", label="Bootstrap 0.5")
plt.plot(en_x, en_y, marker=".", linewidth=1, color="blueviolet", label="Ensembl")

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

# combined plots
kbmc44_y, kbmc44_x = calibration_curve(labels_test, MC_IV_KB_IV[:, 0], n_bins=12)
kbmc51_y, kbmc51_x = calibration_curve(labels_test, MC_V_KB_I[:, 0], n_bins=12)
# chosen
bpkb51_y, bpkb51_x = calibration_curve(labels_test, BP_V_KB_I[:, 0], n_bins=12)
kb1en_y, kb1en_x = calibration_curve(labels_test, KB_I_EN[:, 0], n_bins=12)
kb2en_y, kb2en_x = calibration_curve(labels_test, KB_II_EN[:, 0], n_bins=12)

fig, ax = plt.subplots()
# only these two lines are calibration curves
plt.plot(
    kbmc44_x,
    kbmc44_y,
    marker=".",
    linewidth=1,
    color="darkorange",
    label="KBMC 0.4 rate, 0.4 support",
)
plt.plot(
    kbmc51_x,
    kbmc51_y,
    marker=".",
    linewidth=1,
    color="chocolate",
    label="KBMC 0.5 rate, 0.1 support",
)
plt.plot(
    bpkb51_x,
    bpkb51_y,
    marker=".",
    linewidth=1,
    color="yellowgreen",
    label="BpKB 0.5 rate, 0.1 support",
)
plt.plot(
    kb1en_x,
    kb1en_y,
    marker=".",
    linewidth=1,
    color="steelblue",
    label="KBEn 0.1 support",
)
plt.plot(
    kb2en_x, kb2en_y, marker=".", linewidth=1, color="dimgrey", label="KBEn 0.2 support"
)

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
# plt.figure(figsize=(12, 6))

# Histogram for Model 1
# plt.subplot(1, 2, 1)
# plt.hist(y_std_mc_V_kb_1, bins=20, color='blue', edgecolor='black', alpha=0.5, label='KBMC 0.5, 0.1')
# plt.hist(y_std_kb_I, bins=20, color='yellow', edgecolor='black', alpha=0.6, label='KB 0.1')
# plt.hist(y_std_mc_V, bins=20, color='purple', edgecolor='black', alpha=0.5, label='MC 0.5')
# plt.hist(y_std_mc_IV_kb_4, bins=20, color='orange', edgecolor='black', alpha=0.5, label='KBMC 0.4, 0.4')

# plt.title('Distribution of Standard Deviations')
# plt.xlabel('Standard Deviation')
# plt.ylabel('Frequency')
# plt.legend()

# Define the number of bins and the range
bins = np.linspace(
    0, 0.5, 20
)  # Adjust the range and number of bins as needed# Calculate histogram data for each dataset
hist_mc_V, _ = np.histogram(y_std_mc_V, bins=bins)
hist_kb_I, _ = np.histogram(y_std_kb, bins=bins)
hist_mc_V_kb_1, _ = np.histogram(y_std_mc_V_kb_I, bins=bins)
hist_mc_IV_kb_4, _ = np.histogram(
    y_std_mc_IV_kb_IV, bins=bins
)  # Create a single figure

hist_bp_V, _ = np.histogram(y_std_bp_V, bins=bins)
hist_en, _ = np.histogram(y_std_en, bins=bins)
hist_bp_V_kb_I, _ = np.histogram(y_std_bp_V_kb_I, bins=bins)
hist_kb_I_en, _ = np.histogram(y_std_kb_I_en, bins=bins)
hist_kb_II_en, _ = np.histogram(y_std_kb_II_en, bins=bins)

plt.figure(figsize=(12, 6))  # Plot each bin individually
for i in range(len(bins) - 1):
    bin_range = (
        bins[i],
        bins[i + 1],
    )  # Get the counts for the current bin for each dataset
    counts = [
        (hist_mc_V[i], "tomato", "Monte Carlo Dropout (0.5 rate)"),
        (hist_kb_I[i], "goldenrod", "Knowledge Base Dropout (0.1 min support)"),
        (hist_bp_V[i], "green", "Bootstrap (0.5 rate)"),
        (hist_en[i], "blueviolet", "Ensembl"),
    ]  # Sort counts by the number of occurrences (descending order) to plot larger values first
    counts.sort(reverse=True)  # Plot the counts in the current bin range
    for count, color, label in counts:
        plt.bar(
            (bins[i] + bins[i + 1]) / 2,
            count,
            width=(bins[i + 1] - bins[i]) * 0.9,  # Adjust the width to avoid overlap
            color=color,
            label=label if i == 0 else "",
            alpha=0.85,
        )  # Set the title and labels
plt.title("Distribution of Standard Deviations")
plt.xlabel("Value")
plt.ylabel("Frequency")  # Show the legend
plt.legend()  # Show the plot
plt.show()

# Box plot for both models
# plt.subplot(1, 2, 2)
# plt.boxplot([y_std_mc, y_std_kbmc], vert=False, patch_artist=True, tick_labels=['MC', 'KB'])
# plt.title('Box Plot of Standard Deviations')
# plt.xlabel('Standard Deviation')

# plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))  # Plot each bin individually
for i in range(len(bins) - 1):
    bin_range = (
        bins[i],
        bins[i + 1],
    )  # Get the counts for the current bin for each dataset
    counts = [
        (hist_mc_V_kb_1[i], "chocolate", "Knowledge Base 0.1 Monte Carlo Dropout 0.5"),
        (
            hist_mc_IV_kb_4[i],
            "darkorange",
            "Knowledge Base 0.4 Monte Carlo Dropout 0.4",
        ),
        (hist_bp_V_kb_I[i], "yellowgreen", "Bootstrap 0.5 rate Knowledge Base 0.1"),
        (hist_kb_I_en[i], "steelblue", "Knowledge Base 0.1 Ensembl"),
        (hist_kb_II_en[i], "dimgrey", "Knowledge Base 0.2 Ensembl"),
    ]  # Sort counts by the number of occurrences (descending order) to plot larger values first
    counts.sort(reverse=True)  # Plot the counts in the current bin range
    for count, color, label in counts:
        plt.bar(
            (bins[i] + bins[i + 1]) / 2,
            count,
            width=(bins[i + 1] - bins[i]) * 0.9,  # Adjust the width to avoid overlap
            color=color,
            label=label if i == 0 else "",
            alpha=0.85,
        )  # Set the title and labels
plt.title("Distribution of Standard Deviations")
plt.xlabel("Value")
plt.ylabel("Frequency")  # Show the legend
plt.legend()  # Show the plot
plt.show()

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


# data comparison
def max_min_calibration_error_difference(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    num_bins: int = 12,
    vectorized: bool = False,
    recal_model: IsotonicRegression = None,
    prop_type: str = "interval",
) -> float:
    # Check that input arrays are flat
    assert_is_flat_same_shape(y_pred, y_std, y_true)
    # Check that prop_type is valid
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

    # Absolute difference between expected and observed proportions
    abs_diff_proportions = np.abs(exp_proportions - obs_proportions)

    # Get the max and min of the calibration error across bins
    max_diff = np.max(abs_diff_proportions)
    min_diff = np.min(abs_diff_proportions)

    # Return the difference between max and min calibration errors
    return max_diff - min_diff


all_data_list = [
    cal_abs_mc_V,
    sharp_metric_mc_V,
    sharp_metric_kb_I,
    cal_abs_KB_I,
    cal_abs_bp_V,
    sharp_metric_bp_V,
    cal_abs_en,
    sharp_metric_en,
    cal_abs_MC_IV_KB_IV,
    sharp_metric_mc_IV_kb_IV,
    cal_abs_MC_V_KB_I,
    sharp_metric_mc_V_kb_I,
    cal_abs_bp_V_kb_I,
    sharp_metric_bp_V_kb_I,
    cal_abs_kb_I_en,
    sharp_metric_kb_I_en,
    cal_abs_kb_II_en,
    sharp_metric_kb_II_en,
]
uq_labels_list = [
    "MC5",
    "KB1",
    "Bp5",
    "En",
    "KBMC44",
    "KBMC15",
    "BpKB51",
    "KB1En",
    "KB2En",
]


# max & min calibration error and sharpness
all_list = [
    KB_I_pred[:, :, 0],
    MC_V_pred[:, :, 0],
    MC_IV_KB_IV_pred[:, :, 0],
    MC_V_KB_I_pred[:, :, 0],
    BP_V_pred[:, :, 0],
    EN_pred[:, :, 0],
    BP_V_KB_I_pred[:, :, 0],
    KB_I_EN_pred[:, :, 0],
    KB_II_EN_pred[:, :, 0],
]
all_std_list = [
    y_std_kb,
    y_std_mc_V,
    y_std_mc_IV_kb_IV,
    y_std_mc_V_kb_I,
    y_std_bp_V,
    y_std_en,
    y_std_bp_V_kb_I,
    y_std_kb_I_en,
    y_std_kb_II_en,
]


pred_diff_list = [
    max(
        max_min_calibration_error_difference(arr[a, :], y_std, labels_arr)
        for arr, y_std in zip(all_list, all_std_list)
    )
    for a in range(0, 99)
]
std_max_list = [max(y_std) - min(y_std) for y_std in all_std_list]

print(pred_diff_list)
print(std_max_list)

# comparison sums
j = 0
for i in uq_labels_list:
    print(f"Sum for {i} is {all_data_list[j] + all_data_list[j + 1]}")
    # print(f"Adjusted sum for {i} is {all_data_list[j] + all_data_list[j + 1]}")
    j = j + 2

# plot

fig, ax = plt.subplots()
# only these two lines are calibration curves
plt.plot(
    sharp_metric_mc_V,
    cal_abs_mc_V,
    marker="o",
    linewidth=1,
    color="tomato",
    label="MC Dropout 0.5",
)
plt.plot(
    sharp_metric_kb_I,
    cal_abs_KB_I,
    marker="o",
    linewidth=1,
    color="goldenrod",
    label="KB Dropout 0.1",
)
plt.plot(
    sharp_metric_bp_V,
    cal_abs_bp_V,
    marker="o",
    linewidth=1,
    color="green",
    label="Bootstrap 0.5",
)
plt.plot(
    sharp_metric_en,
    cal_abs_en,
    marker="o",
    linewidth=1,
    color="blueviolet",
    label="Ensembl",
)
plt.plot(
    sharp_metric_mc_IV_kb_IV,
    cal_abs_MC_IV_KB_IV,
    marker="o",
    linewidth=1,
    color="darkorange",
    label="KBMC 0.4 rate, 0.4 support",
)
plt.plot(
    sharp_metric_mc_V_kb_I,
    cal_abs_MC_V_KB_I,
    marker="o",
    linewidth=1,
    color="chocolate",
    label="KBMC 0.5 rate, 0.1 support",
)
plt.plot(
    sharp_metric_bp_V_kb_I,
    cal_abs_bp_V_kb_I,
    marker="o",
    linewidth=1,
    color="yellowgreen",
    label="BpKB 0.5 rate, 0.1 support",
)
plt.plot(
    sharp_metric_kb_I_en,
    cal_abs_kb_I_en,
    marker="o",
    linewidth=1,
    color="steelblue",
    label="KBEn 0.1 support",
)
plt.plot(
    sharp_metric_kb_II_en,
    cal_abs_kb_II_en,
    marker="o",
    linewidth=1,
    color="dimgrey",
    label="KBEn 0.2 support",
)

# plt.plot(cnn_x, cnn_y, marker='o', linewidth=1, label='CNN')

# reference line, legends, and axis labels
fig.suptitle("Calibration and sharpness plot for UQ for Fake News detection")
ax.set_xlabel("Sharpness")
ax.set_ylabel("Calibration")
plt.legend()
plt.show()
