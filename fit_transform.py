import argparse
import ast
import gc
import logging
import os
import pickle
import re
import sys
import time

import fastrand
import numpy as np
import pandas as pd
import psutil
import tensorflow
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

SEED = 42
BATCH_SIZE = 1024
EPOCHS = 20
N = 100
TEST_MODE = False


def save_model(model, save_file: str) -> None:
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    with open(save_file, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info(f"model saved in: {save_file}")


def save_results_and_model(
    data_file,
    prob,
    indices,
    dropout_probability,
    bootstrapping_probability,
    kb_dropout,
    mean,
    std,
    model_type,
    ensemble,
    predictions,
    model,
):
    directory = os.path.join(os.path.dirname(data_file), "results")
    os.makedirs(directory, exist_ok=True)
    suffix = f"_mc_{dropout_probability}" if dropout_probability > 0 else ""
    suffix += (
        f"_bp_{bootstrapping_probability}" if bootstrapping_probability > 0 else ""
    )
    if kb_dropout is not None:
        support_value = re.findall(r"[-+]?(?:\d*\.*\d+)", kb_dropout)[0].replace(
            ".", "_"
        )
        suffix += f"_kb_{support_value}"
    suffix += "_ensemble" if ensemble else ""
    prob_file = os.path.join(directory, f"{model_type}_prob{suffix}")
    indices_file = os.path.join(directory, f"{model_type}_indices{suffix}")
    mean_file = os.path.join(directory, f"{model_type}_mean{suffix}")
    std_file = os.path.join(directory, f"{model_type}_std{suffix}")
    predictions_file = os.path.join(directory, f"{model_type}_predictions{suffix}")
    model_file = os.path.join(directory, f"{model_type}_keras_model{suffix}")
    np.save(prob_file, prob)
    np.save(indices_file, indices)
    np.save(mean_file, mean)
    np.save(std_file, std)
    np.save(predictions_file, predictions)
    save_model(model=model, save_file=model_file)
    logging.info(f"probabilities for test data saved into: {prob_file}")
    logging.info(f"corresponding indices for test data saved in: {indices_file}")


def predict_prob(number):
    return [number[0], 1 - number[0]]


def get_preformated_input(x_test, vectorizer, txts, indices, dropout_txts):
    if dropout_txts is None:
        return x_test.toarray()
    output = []
    for idx in indices:
        words = set(txts[idx].split())
        words_list = dropout_txts[idx]
        dropout_words = set()
        for itemset, support_val in words_list:
            if support_val > fastrand.pcg32() / (2**32 - 1):
                dropout_words.update(itemset)
        new_words = words.difference(dropout_words)
        output.append(" ".join(new_words))
    return vectorizer.transform(output).toarray()


def get_texts_with_dropout_prob(txts, indices, kb_df):
    dropout_txts = {}
    itemset_list = []
    for _, row in kb_df.iterrows():
        itemset = set(ast.literal_eval(row["itemsets"]))
        support_val = row["support"]
        itemset_list.append((itemset, support_val))
    for idx in indices:
        words = set(txts[idx].split())
        dropout_words = []
        for itemset, support_val in itemset_list:
            if itemset.issubset(words):
                dropout_words.append((itemset, support_val))
        dropout_txts[idx] = dropout_words
    return dropout_txts


def main(
    true_data_file,
    fake_data_file,
    true_test_data_file,
    fake_test_data_file,
    dropout_probability,
    knowledge_base,
    model_type,
    bootstrapping_probability,
    ensemble,
) -> float:
    proc = psutil.Process(os.getpid())
    true_data = pd.read_csv(true_data_file, engine="python", delimiter=",").dropna()
    rexts = true_data["text"].tolist()
    fake_data = pd.read_csv(fake_data_file, engine="python", delimiter=",").dropna()
    fexts = fake_data["text"].tolist()
    kb_df = pd.read_csv(knowledge_base) if knowledge_base is not None else None
    del true_data, fake_data

    # to check if it's working just get a slice of data
    if TEST_MODE:
        rexts = rexts[:50]
        fexts = fexts[:50]

    texts = rexts + fexts
    labels = ["real"] * len(rexts) + ["fake"] * len(fexts)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    logging.info(f"{label_encoder.inverse_transform([0, 1]) = }")
    # label_encoder.inverse_transform([0, 1]) = array(['fake', 'real'], dtype='<U4')
    indices = list(range(X.shape[0]))
    del rexts, fexts

    if true_test_data_file and fake_test_data_file:
        true_test_data = pd.read_csv(
            true_test_data_file, engine="python", delimiter=","
        ).dropna()
        rexts_test = true_test_data["text"].tolist()
        fake_test_data = pd.read_csv(
            fake_test_data_file, engine="python", delimiter=","
        ).dropna()
        fexts_test = fake_test_data["text"].tolist()
        del true_test_data, fake_test_data

        if TEST_MODE:
            rexts_test = rexts_test[:50]
            fexts_test = fexts_test[:50]

        texts_test = rexts_test + fexts_test
        texts = texts_test
        X_train, y_train, indices_train = X, y, indices
        labels_test = ["real"] * len(rexts_test) + ["fake"] * len(fexts_test)
        X_test = vectorizer.transform(texts_test)
        y_test = label_encoder.fit_transform(labels_test)
        indices_test = list(range(X_test.shape[0]))
        del X, y, indices, rexts_test, fexts_test, texts_test
    else:
        X_train, X_test, y_train, y_test, indices_train, indices_test = (
            train_test_split(X, y, indices, test_size=0.2, random_state=SEED)
        )
        del X, y, indices

    if model_type == "fcl":
        model = fcl_model(
            X_train.shape[1],
            dropout_prob=dropout_probability,
            bootstrap=bootstrapping_probability,
        )
    elif model_type == "cnn":
        model = cnn_model(
            X_train.shape[1],
            dropout_prob=dropout_probability,
            bootstrap=bootstrapping_probability,
        )
    else:
        raise ValueError(f"Undupported model type: {model_type}.")

    n = (
        N
        if dropout_probability > 0
        or knowledge_base is not None
        or bootstrapping_probability > 0
        else 1
    )
    dropout_txts = (
        get_texts_with_dropout_prob(texts, indices_test, kb_df)
        if knowledge_base is not None
        else None
    )
    # callback = tensorflow.keras.callbacks.EarlyStopping(patience=5)

    if ensemble:
        y_pred_list = []
        X_train_array = X_train.toarray()

        elapsed_time_inference = 0
        for i in range(N):
            new_X_train, new_y_train = resample(
                X_train_array,
                y_train,
                n_samples=int(len(X_train_array) * 0.5),
                replace=True,
                random_state=i,
            )
            x_t, x_v, y_t, y_v = train_test_split(
                new_X_train, new_y_train, test_size=0.1, random_state=SEED
            )
            train_gen = DataGenerator(x_t, y_t, BATCH_SIZE)
            validation_gen = DataGenerator(x_v, y_v, BATCH_SIZE)
            model.fit(
                train_gen,
                epochs=EPOCHS,
                validation_data=validation_gen,
            )
            del train_gen, validation_gen, x_t, x_v, y_t, y_v, new_X_train, new_y_train

            start_time_inference = time.time()
            preformated_input = get_preformated_input(
                X_test, vectorizer, texts, indices_test, dropout_txts
            )
            # y_pred_list.append(model.predict(preformated_input, batch_size=BATCH_SIZE))
            test_gen = DataGenerator(preformated_input, None, BATCH_SIZE)
            y_pred_list.append(model.predict(test_gen))
            clear_memory()
            elapsed_time_inference += time.time() - start_time_inference
    else:
        x_t, x_v, y_t, y_v = train_test_split(
            X_train.toarray(), y_train, test_size=0.1, random_state=SEED
        )
        train_gen = DataGenerator(x_t, y_t, BATCH_SIZE)
        validation_gen = DataGenerator(x_v, y_v, BATCH_SIZE)
        model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=validation_gen,
        )
        del train_gen, validation_gen, x_t, x_v, y_t, y_v, X_train, y_train

        y_pred_list = []
        start_time_inference = time.time()
        for _ in range(n):
            preformated_input = get_preformated_input(
                X_test, vectorizer, texts, indices_test, dropout_txts
            )
            test_gen = DataGenerator(preformated_input, None, BATCH_SIZE)
            y_pred_list.append(model.predict(test_gen))
            print(f"## {human_readable_sizeof_format(proc.memory_info().rss) = }")
            del preformated_input, test_gen
            clear_memory()
        elapsed_time_inference = time.time() - start_time_inference

    y_pred = np.stack(y_pred_list)
    y_pred_mean = y_pred.mean(axis=0)
    y_pred_std = y_pred.std(axis=0)
    y_pred_binary = np.round(y_pred_mean).flatten()
    accuracy = accuracy_score(y_test, y_pred_binary)
    logging.info(f"accuracy: {accuracy}")

    y_prob = np.array(list(map(predict_prob, y_pred_mean)))
    logging.info(f"probability: {y_prob}")
    save_results_and_model(
        data_file=true_data_file,
        prob=y_prob,
        indices=indices_test,
        dropout_probability=dropout_probability,
        bootstrapping_probability=bootstrapping_probability,
        kb_dropout=knowledge_base,
        mean=y_pred_mean,
        std=y_pred_std,
        model_type=model_type,
        ensemble=ensemble,
        predictions=y_pred,
        model=model,
    )
    return elapsed_time_inference


def range_0_1_float_type(arg):
    min_val, max_val = 0, 1
    try:
        f = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f < min_val or f > max_val:
        raise argparse.ArgumentTypeError(
            f"Argument must be in range [{min_val}; {max_val}]"
        )
    return f


def clear_memory() -> None:
    tensorflow.keras.backend.clear_session()
    gc.collect()


def human_readable_sizeof_format(num, suffix="B"):
    """by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified"""
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0


def print_memory_usage(variables: dict) -> None:
    for name, size in sorted(
        ((name, sys.getsizeof(value)) for name, value in list(variables.items())),
        key=lambda x: -x[1],
    )[:5]:
        print("{:>30}: {:>8}".format(name, human_readable_sizeof_format(size)))


def config_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


if __name__ == "__main__":
    tensorflow.keras.utils.set_random_seed(SEED)
    # tensorflow.config.run_functions_eagerly(True)

    parser = argparse.ArgumentParser("FIT_TRANSFORM")
    parser.add_argument("-r", "--real", help="real data csv file")
    parser.add_argument("-f", "--fake", help="fake data csv file")
    parser.add_argument("-rt", "--real-test", help="real data only for test csv file")
    parser.add_argument("-ft", "--fake-test", help="fake data only for test csv file")
    parser.add_argument(
        "-dp",
        "--dropout-probability",
        help="dropout probability",
        type=range_0_1_float_type,
        default=0.0,
    )
    parser.add_argument(
        "-kb", "--knowledge-base", help="clean knowledge base csv file", default=None
    )
    parser.add_argument(
        "-t", "--time-results-file", help="csv file to write elapsed time", default=None
    )
    parser.add_argument(
        "-m", "--model-type", help="neural network model to use", choices=["fcl", "cnn"]
    )
    parser.add_argument(
        "-bp",
        "--bootstrapping-probability",
        help="bootstrapping probability",
        type=range_0_1_float_type,
        default=0.0,
    )
    parser.add_argument(
        "-e", "--ensemble", help="if run with ensemble method", action="store_true"
    )
    parser.add_argument("--no-training-dropout", action="store_true")
    args = parser.parse_args()

    config_logger()

    if args.no_training_dropout:
        from models_no_training_dropout import DataGenerator, cnn_model, fcl_model
    else:
        from models import DataGenerator, cnn_model, fcl_model

    start_time = time.time()
    # with tensorflow.device('/GPU:0'):
    elapsed_time_inference = main(
        true_data_file=args.real,
        fake_data_file=args.fake,
        true_test_data_file=args.real_test,
        fake_test_data_file=args.fake_test,
        dropout_probability=args.dropout_probability,
        knowledge_base=args.knowledge_base,
        model_type=args.model_type,
        bootstrapping_probability=args.bootstrapping_probability,
        ensemble=args.ensemble,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    if args.time_results_file:
        with open(args.time_results_file, "a") as f:
            print(f"{args},{elapsed_time},{elapsed_time_inference}", file=f)
    logging.info(args)
    logging.info(f"elapsed time: {elapsed_time}")
