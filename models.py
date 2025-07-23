import numpy as np
import tensorflow
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Conv1D, Dense, GlobalMaxPooling1D
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    def __init__(
        self, x_set: np.ndarray, y_set: np.ndarray | None, batch_size: int, **kwargs
    ):
        super().__init__(**kwargs)
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self) -> int:
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        if self.y is None:
            return (batch_x,)
        else:
            batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
            return (batch_x, batch_y)


class CustomDropout(tensorflow.keras.layers.Layer):
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=False):
        # if training:
        #     return inputs
        return tensorflow.keras.random.dropout(inputs, rate=self.rate)


def fcl_model(
    input_shape: int, dropout_prob: float = 0.0, bootstrap: float = 0.0
) -> Sequential:
    model = Sequential(
        [
            Input(shape=(input_shape,)),
            CustomDropout(rate=bootstrap),
            Dense(64, activation="relu"),
            CustomDropout(rate=dropout_prob),
            Dense(32, activation="relu"),
            CustomDropout(rate=dropout_prob),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def cnn_model(
    input_shape: int,
    filters: int = 256,
    kernel_size: int = 3,
    dropout_prob=0.2,
    bootstrap: float = 0.0,
) -> Sequential:
    model = Sequential(
        [
            Input(
                shape=(
                    input_shape,
                    1,
                )
            ),
            CustomDropout(rate=bootstrap),
            Conv1D(filters, kernel_size, activation="relu"),
            GlobalMaxPooling1D(),
            # CustomDropout(rate=dropout_prob),
            Dense(filters, activation="relu"),
            CustomDropout(rate=dropout_prob),
            Dense(32, activation="relu"),
            CustomDropout(rate=dropout_prob),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
