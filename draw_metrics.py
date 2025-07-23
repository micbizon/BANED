import gc
import logging
import os
import re
import shutil
from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uncertainty_toolbox as uct
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# DATA_SUFFIX = "-emineyetm"
# DATA_SUFFIX = ""
DATA_SUFFIX = "-chosen-ones"
RESULTS_DIR = f"data{DATA_SUFFIX}/results"
INDICES_TEST_PATH = os.path.join(RESULTS_DIR, "fcl_indices.npy")
TRUE_DATA_FILE = os.path.join(f"data{DATA_SUFFIX}", "real_clean.csv")
FAKE_DATA_FILE = os.path.join(f"data{DATA_SUFFIX}", "fake_clean.csv")
DRAWINGS_DIR = f"drawings{DATA_SUFFIX}"
SEED = 42
NUMBER_BINS = 12


def get_files_with_keyword(dir_path: str, keyword: str) -> list[str]:
    return [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f)) and keyword in f
    ]


def get_bin_entries(results_files_groups: dict, nbins: int = 20) -> dict:
    result_filepath = os.path.join(DRAWINGS_DIR, "bin_count.csv")
    logging.info(f"saving bin counts into a {result_filepath} file")
    bin_count_dict = dict()
    for name, files_dict in tqdm(results_files_groups.items()):
        bin_count = [0] * nbins
        probs = np.load(files_dict["prob"])
        for p in probs[:, 0]:
            bin_count[int(p * nbins) if p < 1 else -1] += 1
        bin_count_dict[name] = ",".join([str(i) for i in bin_count])
    with open(result_filepath, "w") as f:
        print(f"model_name,{','.join([str(i) for i in range(nbins)])}", file=f)
        for model_name, bin_count in bin_count_dict.items():
            print(f"{model_name},{bin_count}", file=f)
    return bin_count_dict


def get_grouped_files_with_results(dir_path: str) -> defaultdict[Any, dict]:
    """
    example files group:
        fcl_prob_monte_carlo_dp_0.4_kb_dropout_0_1.npy
        fcl_indices_monte_carlo_dp_0.4_kb_dropout_0_1.npy
        fcl_mean_monte_carlo_dp_0.4_kb_dropout_0_1.npy
        fcl_std_monte_carlo_dp_0.4_kb_dropout_0_1.npy
    """
    starters = ["prob", "indices", "mean", "std"]
    experiment_groups = defaultdict(dict)
    for f in os.listdir(dir_path):
        fstrip = re.sub("fcl_", "", f)
        fstrip = re.sub("\.npy", "", fstrip)
        for starter in starters:
            if fstrip.startswith(starter):
                name = re.sub(f"{starter}_", "", fstrip)
                name = name if name not in starters else "base"
                experiment_groups[name][starter] = os.path.join(dir_path, f)
    logging.info(
        f"got grouped files for {len(experiment_groups)} different experiments"
    )
    return experiment_groups


def get_test_labels(labels_file_path: str) -> list[int]:
    indices = np.load(labels_file_path)
    true_data = pd.read_csv(TRUE_DATA_FILE, engine="python", delimiter=",").dropna()
    rexts = true_data["text"].tolist()
    fake_data = pd.read_csv(FAKE_DATA_FILE, engine="python", delimiter=",").dropna()
    fexts = fake_data["text"].tolist()
    labels = ["real"] * len(rexts) + ["fake"] * len(fexts)
    label_encoder = LabelEncoder()
    labels_all = label_encoder.fit_transform(labels)
    logging.info("successfully loaded true test labels")
    return [labels_all[i] for i in indices]


def get_all_accuracy(results_files_groups: dict, labels: list[int]) -> dict:
    result_filepath = os.path.join(DRAWINGS_DIR, "accuracy_score.csv")
    logging.info(f"saving accuracy scores into a {result_filepath} file")
    accuracy_scores_list, accuracy_scores_dict = list(), dict()
    for name, files_dict in tqdm(results_files_groups.items()):
        probs = np.load(files_dict["prob"])
        pred_binary = probs.argmin(axis=1)
        accuracy = accuracy_score(labels, pred_binary)
        accuracy_scores_list.append((name, accuracy))
    accuracy_scores_list.sort(key=lambda x: x[1], reverse=True)
    with open(result_filepath, "w") as f:
        print("model_name,accuracy_score", file=f)
        for model_name, score in accuracy_scores_list:
            accuracy_scores_dict[model_name] = score
            print(f"{model_name},{score:.5f}", file=f)
    return accuracy_scores_dict


def get_all_precision(results_files_groups: dict, labels: list[int]) -> dict:
    result_filepath = os.path.join(DRAWINGS_DIR, "precision_score.csv")
    logging.info(f"saving precision scores into a {result_filepath} file")
    precision_scores_list, precision_scores_dict = list(), dict()
    for name, files_dict in tqdm(results_files_groups.items()):
        probs = np.load(files_dict["prob"])
        pred_binary = probs.argmax(axis=1)
        precision = precision_score(labels, pred_binary)
        precision_scores_list.append((name, precision))
    precision_scores_list.sort(key=lambda x: x[1], reverse=True)
    with open(result_filepath, "w") as f:
        print("model_name,precision_score", file=f)
        for model_name, score in precision_scores_list:
            precision_scores_dict[model_name] = score
            print(f"{model_name},{score:.5f}", file=f)
    return precision_scores_dict


def plot_calibration_curves(
    results_files_groups: dict,
    labels: list[int],
    nbins: int = 20,
    min_bin: int = 0,
    max_bin: int = -1,
) -> None:
    logging.info(f"plotting calibration curves into a {DRAWINGS_DIR} directory")
    for name, files_dict in tqdm(results_files_groups.items()):
        plt.clf()
        fig, ax = plt.subplots()
        max_bin = max_bin if max_bin > 0 else nbins
        probs = np.load(files_dict["prob"])
        y, x = calibration_curve(labels, probs[:, 0], n_bins=nbins)
        x_lim = x[min_bin:max_bin]
        y_lim = y[min_bin:max_bin]
        plt.plot(x_lim, y_lim, marker="o", linewidth=1, label=name)
        ax.axline((0, 0), slope=1, c="black")
        fig.suptitle("Calibration plot for Fake News detection")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("True probability in each bin")
        # plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        if len(x_lim) > 0 and len(y_lim) > 0:
            x_min_lim, x_max_lim = (
                np.min(x[min_bin:max_bin]),
                np.max(x[min_bin:max_bin]),
            )
            y_min_lim, y_max_lim = (
                np.min(y[min_bin:max_bin]),
                np.max(y[min_bin:max_bin]),
            )
            y_min_lim = y_min_lim if y_min_lim < x_min_lim else x_min_lim
            y_max_lim = y_max_lim if y_max_lim > x_max_lim else x_max_lim
            plt.xlim(x_min_lim, x_max_lim)
            plt.ylim(y_min_lim, y_max_lim)
        plt.tight_layout()
        plt.savefig(
            os.path.join(DRAWINGS_DIR, f"{name}_calibration_curves.png"), dpi=800
        )
        plt.close()
        gc.collect()


def plot_sharpness(results_files_groups: dict, nbins: int = 20) -> None:
    logging.info(f"plotting sharpness into a {DRAWINGS_DIR} directory")
    plt.clf()
    for name, files_dict in tqdm(results_files_groups.items()):
        std = np.load(files_dict["std"]).flatten()
        plt.hist(std, bins=nbins, alpha=0.7, label=name)
    plt.title("Distribution of Standard Deviations")
    plt.xlabel("Standard Deviation")
    plt.ylabel("Frequency")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(DRAWINGS_DIR, "sharpness.png"), dpi=800)
    plt.close()
    gc.collect()


def get_sharpness(std: np.ndarray) -> float:
    assert std.ndim == 1, "Input array must be 1D"
    assert np.all(std >= 0), "All standard deviations must be positive"
    return np.sqrt(np.mean(std**2))


def get_all_sharpness(results_files_groups: dict) -> dict:
    result_filepath = os.path.join(DRAWINGS_DIR, "sharpness_score.csv")
    logging.info(f"saving sharpness scores into a {result_filepath} file")
    sharpness_scores, sharpness_scores_dict = list(), dict()
    for name, files_dict in tqdm(results_files_groups.items()):
        sharpness_scores.append(
            (name, get_sharpness(np.load(files_dict["std"]).flatten()))
        )
    sharpness_scores.sort(key=lambda x: x[1])
    with open(result_filepath, "w") as f:
        print("model_name,sharpness_score", file=f)
        for model_name, score in sharpness_scores:
            sharpness_scores_dict[model_name] = score
            print(f"{model_name},{score:.5f}", file=f)
    return sharpness_scores_dict


def get_ece_score(py, y_test, n_bins=10):
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


def get_all_threshold_ece(
    results_files_groups: dict,
    true_labels: np.ndarray,
    threshold: float,
    nbins: int = 20,
) -> dict:
    result_filepath = os.path.join(DRAWINGS_DIR, f"ece_score_{threshold}_threshold.csv")
    logging.info(f"saving ece scores into a {result_filepath} file")
    ece_scores, ece_scores_dict = list(), dict()
    for name, files_dict in tqdm(results_files_groups.items()):
        probs = np.load(files_dict["prob"])
        probs_t, true_labels_t = list(), list()
        for p, tl in zip(probs, true_labels):
            if p[0] > threshold:
                probs_t.append(p)
                true_labels_t.append(tl)
        nbins_t = int((1.0 - threshold) / (1.0 / float(nbins)))
        probs_t, true_labels_t = np.array(probs_t), np.array(true_labels_t)
        if len(probs_t) > 0:
            ece = get_ece_score(probs_t, true_labels_t, n_bins=nbins_t)
        else:
            ece = 999
        ece_scores.append((name, ece))
    ece_scores.sort(key=lambda x: x[1])
    with open(result_filepath, "w") as f:
        print("model_name,ece_score", file=f)
        for model_name, score in ece_scores:
            ece_scores_dict[model_name] = score
            print(f"{model_name},{score:.5f}", file=f)
    return ece_scores_dict


def get_all_mace(
    results_files_groups: dict, true_labels: np.ndarray, nbins: int = 20
) -> dict:
    result_filepath = os.path.join(DRAWINGS_DIR, "mean_absolute_calibration_error.csv")
    logging.info(
        f"saving mean absolute calibration error into a {result_filepath} file"
    )
    mace_scores, mace_scores_dict = list(), dict()
    for name, files_dict in tqdm(results_files_groups.items()):
        probs = np.load(files_dict["prob"])[:, 0]
        std = np.load(files_dict["std"]).flatten()
        std[std == 0] = 0.000000001
        mace = uct.mean_absolute_calibration_error(
            probs, std, true_labels, num_bins=nbins
        )
        mace_scores.append((name, mace))
    mace_scores.sort(key=lambda x: x[1])
    with open(result_filepath, "w") as f:
        print("model_name,mace", file=f)
        for model_name, score in mace_scores:
            mace_scores_dict[model_name] = score
            print(f"{model_name},{score:.5f}", file=f)
    return mace_scores_dict


def get_all_threshold_mace(
    results_files_groups: dict,
    true_labels: np.ndarray,
    threshold: float,
    nbins: int = 20,
) -> dict:
    result_filepath = os.path.join(
        DRAWINGS_DIR, f"mean_absolute_calibration_error_{threshold}_threshold.csv"
    )
    logging.info(
        f"saving mean absolute calibration error above a {threshold} threshold into a {result_filepath} file"
    )
    mace_scores, mace_scores_dict = list(), dict()
    for name, files_dict in tqdm(results_files_groups.items()):
        probs = np.load(files_dict["prob"])[:, 0]
        std = np.load(files_dict["std"]).flatten()
        std[std == 0] = 0.000000001
        probs_t, std_t, true_labels_t = list(), list(), list()
        for p, s, tl in zip(probs, std, true_labels):
            if p > threshold:
                probs_t.append(p)
                std_t.append(s)
                true_labels_t.append(tl)
        nbins_t = int((1.0 - threshold) / (1.0 / float(nbins)))
        probs_t, std_t, true_labels_t = (
            np.array(probs_t),
            np.array(std_t),
            np.array(true_labels_t),
        )
        if len(probs_t) > 0:
            mace = uct.mean_absolute_calibration_error(
                probs_t, std_t, true_labels_t, num_bins=nbins_t
            )
        else:
            mace = 999
        mace_scores.append((name, mace))
    mace_scores.sort(key=lambda x: x[1])
    with open(result_filepath, "w") as f:
        print(f"model_name,mace", file=f)
        for model_name, score in mace_scores:
            mace_scores_dict[model_name] = score
            print(f"{model_name},{score:.5f}", file=f)
    return mace_scores_dict


def ensure_dir_exists(dir_path: str) -> None:
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)


def config_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


if __name__ == "__main__":
    config_logger()
    labels_test = get_test_labels(INDICES_TEST_PATH)
    grouped_filepaths = get_grouped_files_with_results(RESULTS_DIR)
    ensure_dir_exists(DRAWINGS_DIR)
    plot_calibration_curves(
        grouped_filepaths, labels_test, min_bin=0, max_bin=999, nbins=NUMBER_BINS
    )
    plot_sharpness(grouped_filepaths, nbins=NUMBER_BINS)
    get_all_sharpness(grouped_filepaths)
    get_all_mace(grouped_filepaths, np.array(labels_test), nbins=NUMBER_BINS)
    for t in [0.5, 0.7, 0.75, 0.9]:
        get_all_threshold_mace(
            grouped_filepaths, np.array(labels_test), t, nbins=NUMBER_BINS
        )
        get_all_threshold_ece(
            grouped_filepaths, np.array(labels_test), t, nbins=NUMBER_BINS
        )
    get_all_accuracy(grouped_filepaths, labels_test)
    get_all_precision(grouped_filepaths, labels_test)
    get_bin_entries(grouped_filepaths, nbins=NUMBER_BINS)
