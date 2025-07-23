import numpy as np
import pandas as pd
import scipy.stats as stats

KB_fake = pd.read_csv("fnn_fake_clean_apriori_sup_0.1.csv")
KB_real = pd.read_csv("fnn_real_clean_apriori_sup_0.1.csv")

frequency_real = KB_real["support"].tolist()
frequency_fake = KB_fake["support"].tolist()

item_real = KB_real["itemsets"].tolist()
item_fake = KB_fake["itemsets"].tolist()

indexes_both = []

m = 0

for element in item_fake:
    n = 0
    while n < len(item_real):
        index = []
        if element == item_real[n]:
            index = [m] + [n]
            indexes_both.append(index)
        n = n + 1
    m = m + 1

# chi-square
test_results = []
for pair in indexes_both:
    p = frequency_fake[pair[0]] * len(frequency_fake)
    r = frequency_real[pair[1]] * len(frequency_real)
    table = [[p, r], [len(frequency_fake) - p, len(frequency_real) - r]]
    chi2, p, dof, expected = stats.chi2_contingency(table)
    if p <= 0.05:
        test_results.append([pair, {chi2}, {p}])
print(len(test_results))  # 1236
print(len(indexes_both))  # 1961
