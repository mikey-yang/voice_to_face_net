import torch
import numpy as np
from sklearn.metrics import roc_auc_score 

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def auc(y_true, y_score):
    y_true_2d = np.zeros((len(y_true), 1251))
    y_score_2d = np.zeros((len(y_true), 1251))

    for y in range(len(y_true)):
        y_true_2d[y][y_true[y]] = 1
    for y in range(len(y_score)):
        y_score_2d[y][y_score[y]] = 1
    
    return roc_auc_score(y_true_2d, y_score_2d, multi_class='ovo')


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
