import os
import importlib
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, TimeDistributed, Dense, SpatialDropout1D, BatchNormalization, Dropout, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from datasets import load_dataset
import matplotlib.pyplot as plt
import pickle

# Завантаження даних
dataset = load_dataset("conll2003", trust_remote_code=True)

ner_feature = dataset['train'].features['ner_tags']
tag_id_to_name = {i: name for i, name in enumerate(ner_feature.feature.names)}
tag2idx = {name: i for i, name in tag_id_to_name.items()}

words = list(set([word for sentence in dataset['train']['tokens'] for word in sentence]))
word2idx = {w: i + 2 for i, w in enumerate(words)}
word2idx['UNK'] = 1
word2idx['PAD'] = 0

def encode_sentence(sentence, word2idx):
    return [word2idx.get(word, 1) for word in sentence]

def encode_tags(tags):
    return [tag for tag in tags]

X_train = [encode_sentence(sentence, word2idx) for sentence in dataset['train']['tokens']]
y_train = [encode_tags(tags) for tags in dataset['train']['ner_tags']]
X_val = [encode_sentence(sentence, word2idx) for sentence in dataset['validation']['tokens']]
y_val = [encode_tags(tags) for tags in dataset['validation']['ner_tags']]
X_test = [encode_sentence(sentence, word2idx) for sentence in dataset['test']['tokens']]
y_test = [encode_tags(tags) for tags in dataset['test']['ner_tags']]

X_train = pad_sequences(X_train, padding='post')
y_train = pad_sequences(y_train, padding='post')
X_val = pad_sequences(X_val, padding='post')
y_val = pad_sequences(y_val, padding='post')
X_test = pad_sequences(X_test, padding='post')
y_test = pad_sequences(y_test, padding='post')

y_train = np.array([to_categorical(i, num_classes=len(tag2idx)) for i in y_train], dtype=np.float32)
y_val = np.array([to_categorical(i, num_classes=len(tag2idx)) for i in y_val], dtype=np.float32)
y_test = np.array([to_categorical(i, num_classes=len(tag2idx)) for i in y_test], dtype=np.float32)

# Збереження словників
with open('word2idx.pkl', 'wb') as f:
    pickle.dump(word2idx, f)

with open('tag_id_to_name.pkl', 'wb') as f:
    pickle.dump(tag_id_to_name, f)

# Ініціалізація спрощеної моделі без використання Word2Vec
input_dim = len(word2idx)
output_dim = len(tag2idx)
input_length = X_train.shape[1]

model = Sequential()
model.add(Input(shape=(input_length,), dtype='int32'))
model.add(Embedding(input_dim=input_dim, output_dim=50, input_length=input_length))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(output_dim, activation="softmax")))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Калбеки для тренування
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Тренування моделі
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=1, verbose=1, callbacks=[early_stopping])

# Оцінка моделі
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)

# Оцінка моделі
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Візуалізація результатів тренування
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.title('Accuracy')

plt.show()

# Передбачення на тестовому наборі
predictions = model.predict(X_test)

# Функція для обчислення точності для кожного тегу
def compute_tag_accuracy(y_true, y_pred, tag_id_to_name):
    y_true_flat = np.argmax(y_true, axis=-1).flatten()
    y_pred_flat = np.argmax(y_pred, axis=-1).flatten()

    labels = list(tag_id_to_name.keys())
    target_names = list(tag_id_to_name.values())

    report = classification_report(y_true_flat, y_pred_flat, labels=labels, target_names=target_names, zero_division=0)
    return report

# Обчислення точності для кожного тегу
tag_accuracy_report = compute_tag_accuracy(y_test, predictions, tag_id_to_name)
print(tag_accuracy_report)
