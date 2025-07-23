import os

import matplotlib.pyplot as plt
import numpy as np

DATADIR = "data/results"


def main() -> None:
    # Sample data (replace these with your actual data)
    y_std_mc_V = np.load(
        os.path.join(DATADIR, "cnn_test_pred_std_monte_carlo_dp_0.5.npy")
    ).flatten()
    y_std_kb_I = np.load(
        os.path.join(DATADIR, "cnn_test_pred_std_kb_dropout_0_1.npy")
    ).flatten()
    y_std_mc_V_kb_1 = np.load(
        os.path.join(DATADIR, "cnn_test_pred_std_monte_carlo_dp_0.5_kb_dropout_0_1.npy")
    ).flatten()
    y_std_mc_IV_kb_4 = np.load(
        os.path.join(DATADIR, "cnn_test_pred_std_monte_carlo_dp_0.5_kb_dropout_0_4.npy")
    ).flatten()
    print(np.max(y_std_mc_V))
    print(np.max(y_std_kb_I))
    print(np.max(y_std_mc_V_kb_1))
    print(np.max(y_std_mc_IV_kb_4))
    # Define the number of bins and the range
    bins = np.linspace(0, 0.5, 20)  # Adjust the range and number of bins as needed
    # Calculate histogram data for each dataset
    hist_mc_V, _ = np.histogram(y_std_mc_V, bins=bins)
    hist_kb_I, _ = np.histogram(y_std_kb_I, bins=bins)
    hist_mc_V_kb_1, _ = np.histogram(y_std_mc_V_kb_1, bins=bins)
    hist_mc_IV_kb_4, _ = np.histogram(y_std_mc_IV_kb_4, bins=bins)
    # Create a single figure
    plt.figure(figsize=(12, 6))
    # Plot each bin individually
    for i in range(len(bins) - 1):
        bin_range = (bins[i], bins[i + 1])
        # Get the counts for the current bin for each dataset
        counts = [
            (hist_mc_V[i], "darksalmon", "Monte Carlo Dropout (0.5 rate)"),
            (hist_kb_I[i], "green", "Knowledge Base Dropout (0.1 min support)"),
            (
                hist_mc_V_kb_1[i],
                "darkblue",
                "Knowledge Base 0.1 Monte Carlo Dropout 0.5",
            ),
            (
                hist_mc_IV_kb_4[i],
                "goldenrod",
                "Knowledge Base 0.4 Monte Carlo Dropout 0.4",
            ),
        ]
        # Sort counts by the number of occurrences (descending order) to plot larger values first
        counts.sort(reverse=True)
        # Plot the counts in the current bin range
        for count, color, label in counts:
            plt.bar(
                (bins[i] + bins[i + 1]) / 2,
                count,
                width=(bins[i + 1] - bins[i])
                * 0.9,  # Adjust the width to avoid overlap
                color=color,
                label=label if i == 0 else "",
                alpha=0.9,
            )
    # Set the title and labels
    plt.title("Distribution of Standard Deviations")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    # Show the legend
    plt.legend()
    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
