import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Завантаження збережених словників
with open('../data/processed/word2idx.pkl', 'rb') as f:
    word2idx = pickle.load(f)

with open('../data/processed/tag_id_to_name.pkl', 'rb') as f:
    tag_id_to_name = pickle.load(f)

# Завантаження моделі
model = load_model('../models/saved_models/best_model.keras')

# Власні дані (20 різних речень)
own_sentences = [
    ["This", "is", "a", "test", "sentence"],
    ["Another", "sentence", "for", "testing"],
    ["We", "are", "testing", "NER", "model"],
    ["John", "Smith", "is", "a", "software", "engineer"],
    ["Google", "is", "a", "tech", "company"],
    ["Barack", "Obama", "was", "the", "44th", "President", "of", "the", "United", "States"],
    ["Mount", "Everest", "is", "the", "highest", "mountain", "in", "the", "world"],
    ["Paris", "is", "the", "capital", "of", "France"],
    ["Python", "is", "a", "popular", "programming", "language"],
    ["The", "COVID-19", "pandemic", "has", "affected", "many", "countries"],
    ["NASA", "plans", "to", "send", "astronauts", "to", "Mars"],
    ["Tesla", "is", "known", "for", "its", "electric", "vehicles"],
    ["Amazon", "was", "founded", "by", "Jeff", "Bezos"],
    ["Elon", "Musk", "is", "the", "CEO", "of", "SpaceX"],
    ["The", "Great", "Wall", "of", "China", "is", "a", "historic", "site"],
    ["Cristiano", "Ronaldo", "is", "a", "famous", "football", "player"],
    ["The", "Eiffel", "Tower", "is", "a", "famous", "landmark", "in", "Paris"],
    ["Machine", "learning", "is", "a", "subset", "of", "artificial", "intelligence"],
    ["The", "Amazon", "rainforest", "is", "located", "in", "South", "America"],
    ["COVID-19", "vaccines", "have", "been", "distributed", "worldwide"]
]

# Правильні відповіді (мітки)
true_labels = [
    ["O", "O", "O", "O", "O"],
    ["O", "O", "O", "O"],
    ["O", "O", "O", "B-MISC", "O"],
    ["B-PER", "I-PER", "O", "O", "O", "O"],
    ["B-ORG", "O", "O", "O", "O", "O"],
    ["B-PER", "I-PER", "O", "O", "O", "O", "B-ORG", "O", "O", "B-LOC", "I-LOC"],
    ["B-LOC", "I-LOC", "O", "O", "O", "O", "O", "O"],
    ["B-LOC", "O", "O", "O", "B-LOC"],
    ["B-MISC", "O", "O", "O", "O", "O"],
    ["O", "B-MISC", "O", "O", "O", "O"],
    ["B-ORG", "O", "O", "O", "O", "O", "B-LOC"],
    ["B-ORG", "O", "O", "O", "O", "O", "O"],
    ["B-ORG", "O", "O", "O", "B-PER", "I-PER"],
    ["B-PER", "I-PER", "O", "O", "O", "B-ORG"],
    ["O", "B-LOC", "I-LOC", "O", "B-LOC", "O", "O"],
    ["B-PER", "I-PER", "O", "O", "O", "O", "O"],
    ["O", "B-LOC", "I-LOC", "O", "O", "O", "B-LOC"],
    ["B-MISC", "O", "O", "O", "O", "O", "B-MISC", "I-MISC"],
    ["O", "B-LOC", "O", "O", "B-LOC", "I-LOC"],
    ["B-MISC", "O", "O", "O", "O", "O"]
]

# Перетворення речень у послідовності індексів
def encode_sentence(sentence, word2idx):
    return [word2idx.get(word, 1) for word in sentence]

X_own = [encode_sentence(sentence, word2idx) for sentence in own_sentences]
X_own = pad_sequences(X_own, padding='post')

# Передбачення
predictions = model.predict(X_own)

# Декодування передбачень
def decode_predictions(predictions, tag_id_to_name):
    decoded_predictions = []
    for sentence in predictions:
        sentence_tags = []
        for word in sentence:
            tag_idx = np.argmax(word)
            tag_name = tag_id_to_name[tag_idx]
            sentence_tags.append(tag_name)
        decoded_predictions.append(sentence_tags)
    return decoded_predictions

decoded_predictions = decode_predictions(predictions, tag_id_to_name)

# Вивід у зручному форматі для порівняння
def print_comparison(own_sentences, true_labels, decoded_predictions):
    for sentence, true_sentence, pred_sentence in zip(own_sentences, true_labels, decoded_predictions):
        print(f"{'Word':<15}{'True Label':<15}{'Predicted Label'}")
        print("="*45)
        for word, true_label, pred_label in zip(sentence, true_sentence, pred_sentence):
            print(f"{word:<15}{true_label:<15}{pred_label}")
        print("\n")

print_comparison(own_sentences, true_labels, decoded_predictions)

# Порівняння з правильними відповідями
def compare_predictions(true_labels, decoded_predictions):
    correct = 0
    total = 0
    for true_sentence, pred_sentence in zip(true_labels, decoded_predictions):
        for true_label, pred_label in zip(true_sentence, pred_sentence):
            if true_label == pred_label:
                correct += 1
            total += 1
    accuracy = correct / total
    return accuracy

accuracy = compare_predictions(true_labels, decoded_predictions)

print("Точність передбачень:", accuracy)