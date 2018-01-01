
import numpy as np
from scipy.optimize import golden

EPS = np.finfo(float).eps


def get_k_predictions(test, pred, k=None):
    pred_k = np.zeros(pred.shape, dtype=bool)
    for i, truth in enumerate(test):
        kk = int(np.sum(truth)) if k is None else k
        rank = np.argsort(pred[i])[::-1]
        pred_k[i, rank[:kk]] = 1
    return pred_k


def diff_label_cardinality(t, pred, test):
    lc_pred = np.sum(pred >= t) / len(pred)
    lc_test = np.sum(test) / len(test)
    return np.abs(lc_test - lc_pred)


def get_tau_per_class(test, pred):
    taus = np.empty(test.shape[1])
    for l in range(test.shape[1]):
        tau = golden(diff_label_cardinality, args=(pred[:, l], test[:, l]), brack=(0.001, 0.9))
        taus[l] = tau
    return taus


def diff_label_cardinality_fast(t, pred, lc_test):
    lc_pred = np.sum(pred >= t) / len(pred)
    return np.abs(lc_test - lc_pred)


def get_tau_global(test, pred):
    lc_test = np.sum(test) / len(test)
    tau = golden(diff_label_cardinality_fast, args=(pred, lc_test), brack=(0.001, 0.9))
    return tau


def get_tau_predictions(test, pred, tau=None):
    if tau == -1:
        t = get_tau_per_class(test, pred)
    elif tau is None:
        t = get_tau_global(test, pred)
    else:
        t = tau
    print(t)
    return pred >= t


def instance_precision(y, yhat):
    a = np.sum(yhat & y, axis=1)
    b = np.sum(yhat, axis=1) + EPS
    return (1.0 / len(a)) * np.sum(a / b)


def instance_recall(y, yhat):
    a = np.sum(yhat & y, axis=1)
    b = np.sum(y, axis=1) + EPS
    return (1.0 / len(a)) * np.sum(a / b)


def instance_F1(y, yhat):
    a = np.sum(yhat & y, axis=1)
    b = np.sum(yhat, axis=1) + EPS
    c = np.sum(y, axis=1)
    return (1.0 / len(a)) * np.sum(2.0 * a / (b + c + EPS))


def macro_precision(y, yhat):
    a = np.sum(yhat & y, axis=0)
    b = np.sum(yhat, axis=0) + EPS
    return (1.0 / len(a)) * np.sum(a / b)


def macro_recall(y, yhat):
    a = np.sum(yhat & y, axis=0)
    b = np.sum(y, axis=0) + EPS
    return (1.0 / len(a)) * np.sum(a / b)


def macro_F1(y, yhat):
    a = np.sum(yhat & y, axis=0)
    b = np.sum(yhat, axis=0) + EPS
    c = np.sum(y, axis=0)
    return (1.0 / len(a)) * np.sum(2.0 * a / (b + c + EPS))


def micro_precision(y, yhat):
    a = np.sum(yhat & y)
    b = np.sum(yhat)
    return a / b


def micro_recall(y, yhat):
    a = np.sum(yhat & y)
    b = np.sum(y)
    return a / b


def micro_F1(y, yhat):
    a = np.sum(yhat & y)
    b = np.sum(yhat)
    c = np.sum(y)
    return 2.0 * a / (b + c + EPS)


def get_precision_recall_fscore(test, pred, mmi):
    if mmi == 'instance':
        prec = instance_precision(test, pred)
        recall = instance_recall(test, pred)
        f1 = instance_F1(test, pred)
    elif mmi == 'macro':
        prec = macro_precision(test, pred)
        recall = macro_recall(test, pred)
        f1 = macro_F1(test, pred)
    elif mmi == 'micro':
        prec = micro_precision(test, pred)
        recall = micro_recall(test, pred)
        f1 = micro_F1(test, pred)
    else:
        prec = 0
        recall = 0
        f1 = 0
    return prec, recall, f1


def print_precision_recall_fscore(test, pred):
    print(instance_precision(test, pred), instance_recall(test, pred), instance_F1(test, pred), 'instance')
    print(macro_precision(test, pred), macro_recall(test, pred), macro_F1(test, pred), 'macro')
    print(micro_precision(test, pred), micro_recall(test, pred), micro_F1(test, pred), 'micro')


def AP(labels, probs):
    avg_precs = []
    for lbls, prbs in zip(labels, probs):
        sindices = np.argsort(prbs)[::-1]
        n_labels = np.sum(lbls)
        n_precs = 0
        precs = 0
        for i, sidx in enumerate(sindices):
            if lbls[sidx]:
                n_precs += 1
                precs += n_precs / (i + 1)
            if n_precs == n_labels:
                avg_precs.append(precs / n_labels)
                break
    return np.array(avg_precs)


def mAP(labels, probs):
    return np.mean(AP(labels, probs))
