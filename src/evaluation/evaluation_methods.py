import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test):
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_accuracy}')
    return test_loss, test_accuracy


# Функція для обчислення точності для кожного тегу
def compute_tag_accuracy(y_true, y_pred, tag_id_to_name):
    y_true_flat = np.argmax(y_true, axis=-1).flatten()
    y_pred_flat = np.argmax(y_pred, axis=-1).flatten()

    labels = list(tag_id_to_name.keys())
    target_names = list(tag_id_to_name.values())

    report = classification_report(y_true_flat, y_pred_flat, labels=labels, target_names=target_names, zero_division=0)
    return report

'''def decode_predictions(predictions, tag_id_to_name):
    predicted_tags = []
    for sentence in predictions:
        sentence_tags = []
        for word in sentence:
            tag_idx = np.argmax(word)  # Знаходження індексу з найбільшою ймовірністю
            tag_name = tag_id_to_name[tag_idx]  # Перетворення індексу на назву тегу
            sentence_tags.append(tag_name)
        predicted_tags.append(sentence_tags)
    return predicted_tags

def compute_metrics(y_true, y_pred, tag_id_to_name):
    correct_preds, total_correct, total_preds = 0., 0., 0.
    for i in range(len(y_true)):
        true_tags = np.argmax(y_true[i], axis=-1)
        pred_tags = np.argmax(y_pred[i], axis=-1)
        for t, p in zip(true_tags, pred_tags):
            if t != 0:  # Ігнорування тегів "O"
                total_correct += 1
                if t == p:
                    correct_preds += 1
            if p != 0:
                total_preds += 1

    precision = correct_preds / total_preds if total_preds > 0 else 0
    recall = correct_preds / total_correct if total_correct > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    return precision, recall, f1'''
