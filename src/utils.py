import os
import importlib
import tensorflow as tf
from datasets import load_dataset

def get_tag_mapping():
    dataset = load_dataset("conll2003", trust_remote_code=True)
    ner_feature = dataset['train'].features['ner_tags']
    tag_id_to_name = {i: name for i, name in enumerate(ner_feature.feature.names)}
    tag2idx = {name: i for i, name in tag_id_to_name.items()}
    return dataset, tag_id_to_name, tag2idx

def load_function(module_name, function_name):
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    return function

def set_device(device_name):

    try:
        if device_name == "cpu":
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Використання лише CPU
        elif device_name == "gpu":
            physical_devices = tf.config.list_physical_devices('GPU')
            if len(physical_devices) > 0:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)  # Використання GPU
            else:
                print("GPU не знайдено, використовується CPU")
        else:
            raise ValueError("Невідомий device_name. Виберіть 'cpu' або 'gpu'.")
    except Exception as e:
        print(f"Виникла помилка: {e}")

