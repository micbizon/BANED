import argparse
import re

import contractions
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def preprocessing_data(text):
    stemmer = PorterStemmer()
    text_before = contractions.fix(text)
    text = re.sub(r"[^a-zA-Z\s]", "", text_before).strip()
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_words = [word.lower() for word in words if word.lower() not in stop_words]
    filtered_words = [stemmer.stem(word) for word in filtered_words]
    unique_words = set(filtered_words)
    result_words = " ".join(list(unique_words))
    return result_words


def prepare_data(input_file, output_file):
    df = pd.read_csv(input_file)
    df = df.astype(str)
    df.loc[df["title"] == "nan", "title"] = df.loc[df["title"] == "nan", "text"].apply(
        lambda t: t.split(".")[0]
    )
    df.loc[df["text"] == "nan", "text"] = df.loc[df["text"] == "nan", "title"]

    df["title"] = df["title"].apply(preprocessing_data)
    df["text"] = df["text"].apply(preprocessing_data)

    df.to_csv(output_file, index=False)
    print(
        f"Data from a {input_file} file, has been preprocessed and saved in a {output_file} file"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Prep_data", description="Preprocess data for training"
    )
    parser.add_argument("-i", "--input", help="input csv file")
    parser.add_argument("-o", "--output", help="output cleaned csv file")
    args = parser.parse_args()

    nltk.download("punkt")
    nltk.download("stopwords")

    prepare_data(args.input, args.output)
