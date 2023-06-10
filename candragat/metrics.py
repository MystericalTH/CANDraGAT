import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics.functional.classification import binary_auroc, binary_precision_recall_curve, binary_f1_score, binary_matthews_corrcoef, binary_cohen_kappa
from torchmetrics.functional.classification import binary_precision, binary_recall, binary_accuracy, binary_specificity
from torchmetrics.functional.regression import pearson_corrcoef, mean_absolute_error, mean_squared_error, spearman_corrcoef, r2_score
from imblearn.metrics import geometric_mean_score

# # –––––––––––– Metrics for Classification –––––––––––––

class BasicCriterion(object):

    def __init__(self):
        super().__init__()
        self.name = None

    def compute(self, answer, label):
        raise NotImplementedError("Base class")
    
    def __call__(self,answer, label, requires_backward = False):
        if requires_backward:
            return self.compute(answer, label)
        return round(self.compute(answer, label).item(), 4)
    
    def __len__(self):
        return 1

class GMeans(BasicCriterion):
    def __init__(self):
        super().__init__()
        self.name = 'GMeans'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        label = np.array(label)
        answer = np.array(answer)
        answer = np.array([1 if i >= 0.5 else 0 for i in answer])
        gmeans = geometric_mean_score(answer, label)
        return gmeans
    def __call__(self,answer, label):
        return round(self.compute(answer, label), 4)
class AUCPR(BasicCriterion):
    def __init__(self):
        super(AUCPR, self).__init__()
        self.name = 'AUCPR'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        label = torch.LongTensor(label)
        answer = torch.Tensor(answer)
        precision,recall,_, = binary_precision_recall_curve(answer, label)
        aucpr_all = -np.trapz(precision,recall)
        return aucpr_all

class Accuracy(BasicCriterion):                      
    def __init__(self):
        super(Accuracy, self).__init__()
        self.name = 'Accuracy'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        label = torch.LongTensor(label)
        answer = torch.Tensor(answer)
        accuracy = binary_accuracy(answer, label)
        return accuracy

class Balanced_Accuracy(BasicCriterion):                      
    def __init__(self):
        super(Balanced_Accuracy, self).__init__()
        self.name = 'Balanced_Accuracy'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        label = torch.LongTensor(label)
        answer = torch.Tensor(answer)
        recall = binary_recall(answer, label)
        specificity = binary_specificity(answer, label)
        bal_accuracy = (recall+specificity)/2
        return bal_accuracy

class AUROC(BasicCriterion):
    def __init__(self):
        super(AUROC, self).__init__()
        self.name = 'AUROC'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        label = torch.LongTensor(label)
        answer = torch.Tensor(answer)
        auroc = binary_auroc(answer, label)
        return auroc
    
class MCC(BasicCriterion):
    def __init__(self):
        super(MCC, self).__init__()
        self.name = 'MCC'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        label = torch.LongTensor(label)
        answer = torch.Tensor(answer)
        mcc = binary_matthews_corrcoef(answer, label)
        return mcc

class Kappa(BasicCriterion):
    def __init__(self):
        super(Kappa, self).__init__()
        self.name = 'Kappa'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        label = torch.LongTensor(label)
        answer = torch.Tensor(answer)
        kappa = binary_cohen_kappa(answer, label)
        return kappa

class BCE(BasicCriterion):
    def __init__(self):
        super(BCE, self).__init__()
        self.name = 'BCE'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        label = torch.Tensor(label)
        answer = torch.Tensor(answer)
        bce = F.binary_cross_entropy(answer, label, reduction='mean')
        return bce

class F1(BasicCriterion):
    def __init__(self):
        super(F1, self).__init__()
        self.name = 'F1'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        label = torch.LongTensor(label)
        answer = torch.Tensor(answer)
        f1 = binary_f1_score(answer, label)
        return f1

class Precision(BasicCriterion):
    def __init__(self):
        super(Precision, self).__init__()
        self.name = 'Precision'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        label = torch.LongTensor(label)
        answer = torch.Tensor(answer)
        precision = binary_precision(answer,label)
        return precision

class Recall(BasicCriterion):
    def __init__(self):
        super(Recall, self).__init__()
        self.name = 'Recall'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        label = torch.LongTensor(label)
        answer = torch.Tensor(answer)
        recall = binary_recall(answer, label)
        return recall
    
class Specificity(BasicCriterion):
    def __init__(self):
        super(Specificity, self).__init__()
        self.name = 'Specificity'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        label = torch.LongTensor(label)
        answer = torch.Tensor(answer)
        specificity = binary_specificity(answer, label)
        return specificity

# import numpy as np
# import torch
# import torch.nn.functional as F
# from scipy.stats import spearmanr
# from sklearn.metrics import r2_score, roc_auc_score,precision_recall_curve
# from sklearn.metrics import f1_score,precision_score,recall_score




# class AUC(BasicCriterion):
#     def __init__(self):
#         super(AUC, self).__init__()
#         self.name = 'AUC'

#     def compute(self, answer, label):
#         assert len(answer) == len(label)

#         answer = np.array(answer)
#         label = np.array(label)

#         result = roc_auc_score(y_true=label, y_score=answer)
#         return result

# class AUC_PR(object):
#     def __init__(self):
#         super(AUC_PR, self).__init__()
#         self.name = 'AUC_PR'

#     def compute(self, answer, label):
#         assert len(answer) == len(label)

#         answer = np.array(answer)
#         label = np.array(label)

#         precision,recall,_, = precision_recall_curve(label,answer)
#         auPR_all = -np.trapz(precision,recall)
#         return auPR_all,4

# class BCE(BasicCriterion):
#     def __init__(self):
#         super(BCE, self).__init__()
#         self.name = 'BCE'

#     def compute(self, answer, label):
#         assert len(answer) == len(label)

#         answer = torch.Tensor(answer)
#         label = torch.Tensor(label)

#         BCE = F.binary_cross_entropy(answer, label, reduction='mean')
#         return BCE

# class F1(BasicCriterion):
#     def __init__(self):
#         super(F1, self).__init__()
#         self.name = 'F1'

#     def compute(self, answer, label):
#         assert len(answer) == len(label)

#         answer = np.array(answer)
#         label = np.array(label)

#         result = f1_score(y_true=label, y_pred=answer)
#         return result


# class Precision(BasicCriterion):
#     def __init__(self):
#         super(Precision, self).__init__()
#         self.name = 'Precision'

#     def compute(self, answer, label):
#         assert len(answer) == len(label)

#         answer = np.array(answer)
#         label = np.array(label)

#         result = precision_score(y_true=label, y_pred=answer)
#         return result

# class Recall(BasicCriterion):
#     def __init__(self):
#         super(Recall, self).__init__()
#         self.name = 'Recall'

#     def compute(self, answer, label):
#         assert len(answer) == len(label)

#         answer = np.array(answer)
#         label = np.array(label)

#         result = recall_score(y_true=label, y_pred=answer)
#         return result

class RMSE(BasicCriterion):
    def __init__(self):
        super(RMSE, self).__init__()
        self.name = 'RMSE'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = torch.Tensor(answer).squeeze(-1)
        label = torch.Tensor(label)
        RMSE = mean_squared_error(answer, label).sqrt()
        return RMSE


class MAE(BasicCriterion):
    def __init__(self):
        super(MAE, self).__init__()
        self.name = 'MAE'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = torch.Tensor(answer)
        label = torch.Tensor(label)
        #print("Size for MAE")
        MAE = mean_absolute_error(answer, label)
        return MAE


class MSE(BasicCriterion):
    def __init__(self):
        super(MSE, self).__init__()
        self.name = 'MSE'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = torch.Tensor(answer)
        label = torch.Tensor(label)
        #print("Size for MAE")
        MSE = mean_squared_error(answer, label)
        return MSE


class PCC(BasicCriterion):
    def __init__(self):
        super(PCC, self).__init__()
        self.name = 'PCC'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = torch.Tensor(answer)
        label = torch.Tensor(label)
        #print("Size for MAE")
        pcc = pearson_corrcoef(answer, label)
        return pcc


class R2(BasicCriterion):
    def __init__(self):
        super(R2, self).__init__()
        self.name = 'R2'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = torch.Tensor(answer)
        label = torch.Tensor(label)
        #print("Size for MAE")
        r_squared = r2_score(answer, label)
        return r_squared


class SRCC(BasicCriterion):
    def __init__(self):
        super(SRCC, self).__init__()
        self.name = 'Spearman Rank Cor. Coef.'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        #print("Size for MAE")
        answer = torch.Tensor(answer)
        label = torch.Tensor(label)
        srcc = spearman_corrcoef(answer, label)
        return srcc