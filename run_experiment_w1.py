import os
import pickle

import tensorflow as tf
import numpy as np
from src.utils import load_function, get_tag_mapping, set_device
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Налаштування
device_name = "gpu"  # 'cpu' або 'gpu'
model_version = 'create_model_w7'
preprocess_version = 'preprocess_data_w2'
epochs = 50
batch_size = 32
model_dir = './models/saved_models'
model_filename = 'best_model.keras'


# Виконання налаштування пристрою
set_device(device_name)

# Завантаження та обробка даних
dataset, tag_id_to_name, tag2idx = get_tag_mapping()

preprocess = load_function('src.preprocessing.preprocessing_methods', preprocess_version)
X_train, X_val, X_test, y_train, y_val, y_test, word2idx, tag2idx = preprocess(dataset, tag2idx)

# Збереження словників
with open('data/processed/word2idx.pkl', 'wb') as f:
    pickle.dump(word2idx, f)

with open('data/processed/tag_id_to_name.pkl', 'wb') as f:
    pickle.dump(tag_id_to_name, f)

# Ініціалізація моделі
create_model = load_function('src.models.model_definitions', model_version)
input_dim = len(word2idx)
output_dim = len(tag2idx)
input_length = X_train.shape[1]

model = create_model(input_dim, output_dim, input_length)
model.summary()

# Переконайтеся, що директорія для збереження моделей існує
os.makedirs(model_dir, exist_ok=True)
model_checkpoint_path = os.path.join(model_dir, model_filename)

# Калбеки для тренування
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(model_checkpoint_path, save_best_only=True)

# Тренування моделі
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[early_stopping, model_checkpoint])


# Оцінка моделі
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)

evaluate_model = load_function('src.evaluation.evaluation_methods', 'evaluate_model')
evaluate_model(model, X_test, y_test)

# Візуалізація результатів тренування
plot_history = load_function('src.visualization.visualization_methods', 'plot_training_history')
plot_history(history)

# Декодування передбачень
predictions = model.predict(X_test)
compute_tag_accuracy = load_function('src.evaluation.evaluation_methods', 'compute_tag_accuracy')
tag_accuracy_report = compute_tag_accuracy(y_test, predictions, tag_id_to_name)
print(tag_accuracy_report)
