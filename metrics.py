# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np

class runningScore(object):

    def __init__(self, n_classes):
        # initial the confusion metric
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        """
        the maske shape is the flattened shape

        :param label_true:
        :param label_pred:
        :param n_class:
        :return:
        """
        mask = (label_true >= 0) & (label_true < n_class)  # contains the bool value


        # Count number of occurrences of each value in array of non-negative ints
        hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask],
                           minlength=n_class**2).reshape(n_class, n_class)

        return hist


    def update(self, label_trues, label_preds):
        # 对单张image进行处理：
        for lt, lp in zip(label_trues, label_preds):
            # lt, lp is [(1),512,512]
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
            - precision + recall --> F1 score
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        precision = np. diag(hist) / hist.sum(axis=0)
        recall = np.diag(hist) / hist.sum(axis=1)
        F1_score = 2*(precision*recall / (precision + recall))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {'Overall Acc: \t': acc,
                'Mean Acc : \t': acc_cls,
                'FreqW Acc : \t': fwavacc,
                'Mean IoU : \t': mean_iu}, cls_iu, F1_score

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
