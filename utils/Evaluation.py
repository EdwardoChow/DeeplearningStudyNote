# -*- coding: utf-8 -*-
from sklearn.datasets import make_classification
from sklearn import metrics

class Evaluation:
    def __init__(self, pred, true, prob):
        self.y_pred = pred
        self.y_true = true
        self.y_prob = prob

    def Get_Accuracy(self):  # Accuracy 准确率：分类器正确分类的样本数与总样本数之比
        accuracy = metrics.accuracy_score(self.y_true, self.y_pred)
        print("SVM Accuracy_Score = %f" % accuracy)
        return accuracy

    def Get_Precision_score(self):  # Precision：精准率 正确被预测的正样本(TP)占所有被预测为正样本(TP+FP)的比例.
        precision = metrics.precision_score(self.y_true, self.y_pred)
        print("SVM Precision = %f" % precision)
        return precision

    def Get_Recall(self):  # Recall 召回率 正确被预测的正样本(TP)占所有真正 正样本(TP+FN)的比例.
        Recall = metrics.recall_score(self.y_true, self.y_pred)
        print("SVM Recall = %f" % Recall)
        return Recall

    def Get_f1_score(self):  # F1-score: 精确率(precision)和召回率(Recall)的调和平均数
        f1_score = metrics.f1_score(self.y_true, self.y_pred)
        print("SVM F1-Score  = %f" % f1_score)
        return f1_score

    def Get_Auc_value(self):
        # fpr, tpr, thresholds = metrics.roc_curve(samples_test_y, proba_pred_y, pos_label=2)
        auc = metrics.roc_auc_score(self.y_true, self.y_prob)
        print("SVM AUC value: AUC = %f" % auc)
        return auc

# 调用方式：
"""
from util.Evaluation import Evaluation

evaluation = Evaluation(y_predict, y_test_ran, proba_pred_y)  # 输入为，y预测的标签，y的真实标签，y的预测可能性值
evaluation.Get_Accuracy()
evaluation.Get_Precision_score()
evaluation.Get_Recall()
evaluation.Get_f1_score()
evaluation.Get_Auc_value()
"""
