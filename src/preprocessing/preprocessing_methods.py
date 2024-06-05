import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def preprocess_data_w1(dataset, tag2idx):
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

    # Встановлення максимальної довжини послідовностей
    max_length = 100  # Ви можете змінити це значення на бажану максимальну довжину

    X_train = pad_sequences(X_train, maxlen=max_length, padding='post')
    y_train = pad_sequences(y_train, maxlen=max_length, padding='post')
    X_val = pad_sequences(X_val, maxlen=max_length, padding='post')
    y_val = pad_sequences(y_val, maxlen=max_length, padding='post')
    X_test = pad_sequences(X_test, maxlen=max_length, padding='post')
    y_test = pad_sequences(y_test, maxlen=max_length, padding='post')

    y_train = np.array([to_categorical(i, num_classes=len(tag2idx)) for i in y_train], dtype=np.float32)
    y_val = np.array([to_categorical(i, num_classes=len(tag2idx)) for i in y_val], dtype=np.float32)
    y_test = np.array([to_categorical(i, num_classes=len(tag2idx)) for i in y_test], dtype=np.float32)

    return X_train, X_val, X_test, y_train, y_val, y_test, word2idx, tag2idx

def preprocess_data_w2(dataset, tag2idx):

    def filter_sentences(example):
        # Перевірка, чи всі теги в реченні є "O"
        return not all(tag == 0 for tag in example['ner_tags'])
    
    filtered_dataset = {}
    for split in dataset:
        filtered_dataset[split] = dataset[split].filter(filter_sentences)
    
    return preprocess_data_w1(filtered_dataset, tag2idx)
