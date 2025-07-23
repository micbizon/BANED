import argparse
import os
import time

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

TEST_MODE = False


def apriori_algo(basket_sets, min_sup):
    st = time.time()

    frequent_itemsets = apriori(basket_sets, min_support=min_sup, use_colnames=True)
    frequent_itemsets.sort_values("support", ascending=False, inplace=True)

    et = time.time()
    elapsed_time = et - st
    return frequent_itemsets, elapsed_time, len(frequent_itemsets)


def transaction_encoder(df):
    te = TransactionEncoder()
    df = df.astype(str)
    transactions = df["text"].str.split(" ")
    te_ary = te.fit(transactions).transform(transactions)
    return pd.DataFrame(te_ary, columns=te.columns_)


def save_df(dataframe, support, data_file):
    # Convert frozensets to lists for better readability in CSV
    dataframe["itemsets"] = dataframe["itemsets"].apply(lambda x: list(x))

    directory = os.path.join(os.path.dirname(data_file), "results")
    os.makedirs(directory, exist_ok=True)
    file_name, extension = os.path.basename(data_file).split(".")
    save_path = os.path.join(
        directory, f"{file_name}_apriori_sup_{str(support)}.{extension}"
    )
    dataframe.to_csv(save_path, index=False)
    print(f"results for support: {support} saved in: {save_path}")


def draw_frequent_itemset(input_file):
    df = pd.read_csv(input_file)

    # to check if it's working just get a slice of data
    if TEST_MODE:
        df = df.head(100)

    basket_sets = transaction_encoder(df)

    runtime = []
    nb_frequent_itemset = []
    support = []
    for i in range(1, 11):
        min_sup = i / 10
        support.append(min_sup)
        frequent_itemsets, elapsed_time, frequent_itemsets_size = apriori_algo(
            basket_sets, min_sup
        )
        runtime.append(elapsed_time)
        nb_frequent_itemset.append(frequent_itemsets_size)
        print(
            f"******************************Support = {min_sup} ******************************"
        )
        print(f"frequent_itemsets : {frequent_itemsets}")
        print(f"elapsed_time : {elapsed_time}")
        print(f"len(frequent_itemsets) : {frequent_itemsets_size}")
        print(frequent_itemsets)
        print(f"{len(frequent_itemsets) = }")
        print(f"{type(frequent_itemsets) = }")

        save_df(frequent_itemsets, min_sup, input_file)

    print(f"runtime =  {runtime}")
    print(f"nb_frequent_itemset =  {nb_frequent_itemset}")
    print(f"support =  {support}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Apriori_algo")
    parser.add_argument("-i", "--input", help="input csv file with preprocessed data")
    args = parser.parse_args()

    draw_frequent_itemset(args.input)
