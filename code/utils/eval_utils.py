def prec_recall_f1(pred, ref):
    n_hit = 0
    for item in pred:
        if item in ref:
            n_hit += 1
    return _prec_recall_f1(len(pred), len(ref), n_hit)

def _prec_recall_f1(n_pred, n_ref, n_hit):
    if n_pred == 0:
        n_pred = 1
    if n_ref == 0:
        n_ref = 1
    prec = n_hit / n_pred
    recall = n_hit / n_ref
    if prec > 0 and recall > 0:
        f1 = 2/(1/prec + 1/recall)
    else:
        f1 = 0

    return prec, recall, f1

def format_prec_recall_f1(prec, recall, f1):
    prec = '{:.3f}'.format(prec)
    recall = '{:.3f}'.format(recall)
    f1 = '{:.3f}'.format(f1)
    return prec, recall, 

def average(lis):
    return sum(lis) / len(lis)