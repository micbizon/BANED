#! /usr/bin/env python3

import argparse
import ast
import logging
import statistics
from typing import Sequence

import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split


def calculate_support(
    text: str,
    support: pd.DataFrame,
    probability: float,
) -> float:
    words = set(text.split())

    sum = 0
    n = 0

    for _, row in support.iterrows():
        itemset = set(ast.literal_eval(row["itemsets"]))
        support_val = row["support"]

        if itemset.issubset(words):
            sum += support_val
            n += 1

    if n == 0:
        return 0

    return sum / n * probability


def calculate_support_new(
    text: str,
    support: pd.DataFrame,
    probability: float,
) -> float:
    words = set(text.split())

    highest_support = 0
    n = 0

    for _, row in support.iterrows():
        itemset = set(ast.literal_eval(row["itemsets"]))
        support_val = row["support"]

        if itemset.issubset(words):
            highest_support = max(highest_support, support_val)
            n += 1

    if n == 0:
        return 0

    return (1 + highest_support * n) * probability


def match(
    df: pd.DataFrame,
    probabilites: np.ndarray,
    support: pd.DataFrame,
    limit: int | None = None,
    show_progress: bool = False,
) -> Sequence[float]:
    assert df.shape[0] == probabilites.shape[0], (
        f"number of articles ({df.shape[0]}) must match the number of probabilities ({probabilites.shape[0]})"
    )

    results = []

    if limit is not None:
        df = df.head(limit)
        probabilites = probabilites[:limit]

    for (_, row), prob in tqdm.tqdm(
        zip(df.iterrows(), probabilites), total=df.shape[0], disable=not show_progress
    ):
        res = calculate_support_new(row["text"], support, prob)

        results.append(res)

    return results


def get_test(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    indices = np.arange(0, df.shape[0])

    _, test_indices = train_test_split(indices, test_size=0.2, random_state=seed)

    return df.iloc[test_indices]


def main(
    probabilities: str,
    fake_support: str,
    real_support: str,
    input: str,
    out: str,
    out_mean: str,
    limit: int | None = None,
):
    probabilities_arr = np.load(probabilities)

    fake_support_df = pd.read_csv(fake_support)
    real_support_df = pd.read_csv(real_support)

    df = pd.read_csv(input).dropna()

    df = get_test(df, 42)

    assert probabilities_arr.shape[1] == 2, "Probabilities must have 2 columns"

    # here we assume that 0th columns is the probability of being fake
    # and 1st column is the probability of being real

    logging.info("Calculating fake support")
    results_fake = match(df, probabilities_arr[:, 0], fake_support_df, limit, True)

    logging.info("Calculating real support")
    results_real = match(df, probabilities_arr[:, 1], real_support_df, limit, True)

    df = df.head(len(results_fake))

    df["results_fake"] = results_fake
    df["results_real"] = results_real

    df.to_csv(out, index=False)

    with open(out_mean, "w") as f:
        f.write(f"Average results for real and fake support\n")
        f.write(f"fake: {statistics.mean(results_fake)}\n")
        f.write(f"real: {statistics.mean(results_real)}\n")


def config_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s -  %(levelname)s - %(message)s",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Apriori_algo")

    parser.add_argument(
        "--probabilities", type=str, required=True, help="path to probabilities file"
    )
    parser.add_argument(
        "--fake_support", type=str, required=True, help="path to fake support file"
    )
    parser.add_argument(
        "--real_support", type=str, required=True, help="path to real support file"
    )

    parser.add_argument(
        "--limit", type=int, help="limit the number of articles to process"
    )

    parser.add_argument("input", type=str, help="input csv file with the articles")

    parser.add_argument(
        "--out", type=str, help="where to save the csv results", default="results.csv"
    )

    parser.add_argument(
        "--out_mean",
        type=str,
        help="where to save mean results for given supports",
        default="means.txt",
    )

    args = parser.parse_args()

    config_logger()

    main(
        args.probabilities,
        args.fake_support,
        args.real_support,
        args.input,
        args.out,
        args.out_mean,
        args.limit,
    )
