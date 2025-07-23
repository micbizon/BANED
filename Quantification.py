import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

probabilities_cnn = np.load("fnn_all_clean_cnn_prob.npy")
texts_cnn = pd.read_csv("fnn_all_clean.csv", nrows=20).dropna()
frequencies_fake = pd.read_csv("fnn_fake_clean_apriori_sup_0.1.csv", nrows=20)
frequencies_real = pd.read_csv("fnn_real_clean_apriori_sup_0.1.csv", nrows=20)
labels = ["real"] * len(frequencies_real) + ["fake"] * len(frequencies_fake)

# mixing X to regain the articles from the model
X = np.arange(0, len(texts_cnn))
y = [0] * len(texts_cnn)

X_a, X_e, y_a, y_e = train_test_split(X, y, test_size=0.2, random_state=42)
# Extract numbers and texts
numbers = X_e.tolist()
texts = texts_cnn.values.tolist()

# getting back the test texts
mapped_texts = []
mapped_labels = []
for number in numbers:
    mapped_texts.append(texts[number])
    mapped_labels.append(labels[number])

# removing indexes
frequencies_real.reset_index(inplace=True)
frequencies_real.drop("index", axis=1, inplace=True)
frequencies_fake.reset_index(inplace=True)
frequencies_fake.drop("index", axis=1, inplace=True)

# separating frequencies into frequency and word sets
KnowledgeBase_real = frequencies_real["itemsets"].tolist()
KnowledgeBase_real = [sub.replace("[", "") for sub in KnowledgeBase_real]
KnowledgeBase_real = [sub.replace("]", "") for sub in KnowledgeBase_real]
KnowledgeBase_real = [sub.replace("'", "") for sub in KnowledgeBase_real]
KnowledgeBase_real = [txt.split(",") for txt in KnowledgeBase_real]

FrequenciesIndexed_real = frequencies_real["support"].tolist()

# KnowledgeBase_fake
KnowledgeBase_fake = frequencies_fake["itemsets"].tolist()
KnowledgeBase_fake = [sub.replace("[", "") for sub in KnowledgeBase_fake]
KnowledgeBase_fake = [sub.replace("]", "") for sub in KnowledgeBase_fake]
KnowledgeBase_fake = [sub.replace("'", "") for sub in KnowledgeBase_fake]
KnowledgeBase_fake = [txt.split(",") for txt in KnowledgeBase_fake]

FrequenciesIndexed_fake = frequencies_fake["support"].tolist()


# find common elements between texts and knowledge base and collect frequencies of matched elements
# common_items = []

# for i in range (0, len(probabilities_cnn)):
#   if mapped_labels[i]=='fake':
#      common_items = common_items + list(set(mapped_texts[i]) & set(KnowledgeBase_fake))
# else:
#    common_items = common_items + list(set(mapped_texts[i]) & set(KnowledgeBase_real))


def check_whole_match(kb_el_list, text):
    for kb_el in kb_el_list:
        if kb_el not in text:
            return False
    return True


def get_similar_items(text, knowledge_base):
    items = dict()
    for i, kb_element in enumerate(knowledge_base):
        if check_whole_match(kb_element, text):
            items["index"] = i
            items["word"] = kb_element
    return items


def extract_most_similar_item(mapped_texts, KnowledgeBase):
    similar_items = []
    # you need to account for entries from the knowledge base with multiple words
    for element_mapped_text in mapped_texts:
        similar_items.append(get_similar_items(element_mapped_text, KnowledgeBase))
        # ncw = []
        # element_common_index = []
        # for element_text in element_mapped_text:
        #     for kb_entry in KnowledgeBase:
        #         ncw_row = 0
        #         i = 0
        #         for word in kb_entry:
        #             if word in element_text:
        #                 ncw_row = ncw_row + 1
        #                 element_common_index.append(i)
        #         ncw.append(ncw_row)
        #         i = i + 1
        # max_index_element_text, max_value_element_text = max(enumerate(ncw), key=lambda x: x[1])
        # similar_items.append(
        #     {"index": max_index_element_text, "value": max_value_element_text, "word": element_common_index})
    return similar_items


similar_items_real = extract_most_similar_item(mapped_texts, KnowledgeBase_real)
similar_items_fake = extract_most_similar_item(mapped_texts, KnowledgeBase_fake)
print(similar_items_real)
print(similar_items_fake)
# average probability*frequency of matched element for each text
# average all --> uncertainty
