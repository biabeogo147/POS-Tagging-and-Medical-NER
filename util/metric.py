import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred, ignore_label):
    predictions, labels = eval_pred
    mask = labels != ignore_label
    predictions = np.argmax(predictions, axis=-1)
    return accuracy.compute(predictions=predictions[mask], references=labels[mask])