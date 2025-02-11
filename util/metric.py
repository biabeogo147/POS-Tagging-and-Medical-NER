import numpy as np

def compute_metrics(eval_pred, ignore_label, accuracy):
    predictions, labels = eval_pred
    mask = labels != ignore_label
    predictions = np.argmax(predictions, axis=-1)
    return accuracy.compute(predictions=predictions[mask], references=labels[mask])