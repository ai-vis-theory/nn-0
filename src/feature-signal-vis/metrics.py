from collections import defaultdict
import numpy as np
import torch

class MulticlassMetrics:
    """
    This class buffers the metrics for multiclass classification.
    Also, estimates the average of the metrics whose values are stored in the buffer.
    """
    def __init__(self):
        self.reset_state() # Emptying the buffer

    def reset_state(self):
        self.metrics = defaultdict(list) # Initializing the map {metric->buffer(empty list)}

    def update_state(self, **kwargs):
        """
        Named arguments are expected to be in form of {metric:score}
        """
        for key in kwargs:
            self.metrics[key].append(kwargs[key]) # Appending metric wise scores

    def result(self):
        """
        Returns the average of the metrics whose values are stored in the buffer.
        """
        return {
             k: np.mean(self.metrics[k]) for k in self.metrics.keys()
        }

def calculate_f1_precision_recall_accuracy(y_true_oh, y_pred, average='macro'):
    """
    Inputs:
    - y_true_oh: (batch_size, num_classes) / Tensor[float]
    - y_pred: (batch_size, num_classes) / Tensor[float]
    - average: 'macro', 'micro', or 'weighted'

    Outputs:
    - f1_score : float tensor
    - precision : float tensor
    - recall : float tensor
    - acc : float tensor
    """
    y_pred = torch.argmax(y_pred, dim=1)
    y_true = torch.argmax(y_true_oh, dim=1)

    num_classes = y_true_oh.shape[1]

    # Initialize
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)
    support = torch.zeros(num_classes)  # number of true instances per class

    for i in range(num_classes):
        tp[i] = ((y_pred == i) & (y_true == i)).sum().item()
        fp[i] = ((y_pred == i) & (y_true != i)).sum().item()
        fn[i] = ((y_pred != i) & (y_true == i)).sum().item()
        support[i] = (y_true == i).sum().item()

    precision_per_class = tp / (tp + fp + 1e-8)
    recall_per_class = tp / (tp + fn + 1e-8)
    f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + 1e-8)

    total_support = support.sum()

    if average == 'macro':
        precision = precision_per_class.mean()
        recall = recall_per_class.mean()
        f1_score = f1_per_class.mean()
    elif average == 'weighted':
        precision = (precision_per_class * support).sum() / (total_support + 1e-8)
        recall = (recall_per_class * support).sum() / (total_support + 1e-8)
        f1_score = (f1_per_class * support).sum() / (total_support + 1e-8)
    elif average == 'micro':
        # Micro = global TP/FP/FN
        tp_total = tp.sum()
        fp_total = fp.sum()
        fn_total = fn.sum()

        precision = tp_total / (tp_total + fp_total + 1e-8)
        recall = tp_total / (tp_total + fn_total + 1e-8)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    else:
        raise ValueError(f"Unsupported average type: {average}")

    acc = (tp.sum() + 1e-8) / (total_support + 1e-8)

    return f1_score, precision, recall, acc

def calculate_class_wise_accuracy(y_true_oh, y_pred):
    """
    Inputs:
    Shape of y_true_oh : (batch_size, num_classes) / Type: Tensor[Float]
    Shape of y_pred : (batch_size, num_classes) / Type: Tensor[Float]

    Output:
    class_wise_accuracy : dict
    """

    y_pred = torch.argmax(y_pred, dim=1)  # Predicted class indices
    y_true = torch.argmax(y_true_oh, dim=1)  # True class indices

    class_wise_accuracy = {}

    for i in range(y_true_oh.shape[1]):
        true_class_mask = (y_true == i)            # Samples where true label is class i
        total_true = true_class_mask.sum().item()   # Total number of samples of class i

        if total_true == 0:
            accuracy = 0.0
        else:
            correct_preds = (y_pred[true_class_mask] == i).sum().item()  # Correct predictions for class i
            accuracy = correct_preds / total_true

        class_wise_accuracy[f"class_{i}_accuracy_[skip]"] = accuracy

    return class_wise_accuracy
