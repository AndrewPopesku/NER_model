import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, TimeDistributed, BatchNormalization, Attention, Input

def create_model_w1(input_dim, output_dim, input_length):
    model = Sequential()
    model.add(Input(shape=(input_length,), dtype='int32'))
    model.add(Embedding(input_dim=input_dim, output_dim=50))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(output_dim, activation="softmax")))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def create_model_w2(input_dim, output_dim, input_length):
    model = Sequential()
    model.add(Input(shape=(input_length,), dtype='int32'))
    model.add(Embedding(input_dim=input_dim, output_dim=50))
    model.add(Bidirectional(LSTM(units=50, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(output_dim, activation="softmax")))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def create_model_w3(input_dim, output_dim, input_length):
    model = Sequential()
    model.add(Input(shape=(input_length,), dtype='int32'))
    model.add(Embedding(input_dim=input_dim, output_dim=200))
    model.add(Bidirectional(LSTM(units=50, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(output_dim, activation="softmax")))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model

def create_model_w4(input_dim, output_dim, input_length):
    model = Sequential()
    model.add(Input(shape=(input_length,), dtype='int32'))
    model.add(Embedding(input_dim=input_dim, output_dim=200))
    model.add(Bidirectional(LSTM(units=200, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(output_dim, activation="softmax")))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model

def create_model_w5(input_dim, output_dim, input_length):
    model = Sequential()
    model.add(Input(shape=(input_length,), dtype='int32'))
    model.add(Embedding(input_dim=input_dim, output_dim=50))
    model.add(Bidirectional(LSTM(units=200, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(output_dim, activation="softmax")))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model

def create_model_w6(input_dim, output_dim, input_length):
    # Ініціалізація покращеної моделі
    model = Sequential()
    model.add(Input(shape=(input_length,), dtype='int32'))
    model.add(Embedding(input_dim=input_dim, output_dim=50))
    model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Attention Layer
    # Використання стандартного шару Attention з TensorFlow
    attention = Attention()([model.output, model.output])
    attention = tf.keras.layers.GlobalAveragePooling1D()(attention)

    model.add(TimeDistributed(Dense(100, activation="relu")))
    model.add(TimeDistributed(Dense(output_dim, activation="softmax")))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model

def create_model_w7(input_dim, output_dim, input_length):
    # Ініціалізація покращеної моделі
    model = Sequential()
    model.add(Input(shape=(input_length,), dtype='int32'))
    model.add(Embedding(input_dim=input_dim, output_dim=50))
    model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(output_dim, activation="softmax")))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model

def create_model_w8(input_dim, output_dim, input_length):
    # Ініціалізація покращеної моделі
    model = Sequential()
    model.add(Input(shape=(input_length,), dtype='int32'))
    model.add(Embedding(input_dim=input_dim, output_dim=100))
    model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(output_dim, activation="softmax")))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model
