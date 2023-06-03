import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, roc_auc_score,precision_recall_curve,f1_score,precision_score,recall_score


class AUC_PR(object):
    def __init__(self):
        super(AUC_PR, self).__init__()
        self.name = 'AUC_PR'

    def compute(self, answer, label):
        assert len(answer) == len(label)

        answer = np.array(answer)
        label = np.array(label)

        precision,recall,_, = precision_recall_curve(label,answer)
        auPR_all = -np.trapz(precision,recall)
        return round(auPR_all,4)

#not yet completed
'''
class ACC(object):                      
    def __init__(self):
        super(ACC, self).__init__()
        self.name = 'ACC'

    def compute(self, answer, label):
        assert len(answer) == len(label)

        total = len(answer)

        answer = torch.Tensor(answer)
        label = torch.Tensor(label)

        pred = torch.argmax(answer)
        correct = sum(pred == label).float()
        acc = correct / total

        return round(acc.item(),4)
'''


class AUC(object):
    def __init__(self):
        super(AUC, self).__init__()
        self.name = 'AUC'

    def compute(self, answer, label):
        assert len(answer) == len(label)

        answer = np.array(answer)
        label = np.array(label)

        result = roc_auc_score(y_true=label, y_score=answer)
        return round(result,4)


class BCE(object):
    def __init__(self):
        super(BCE, self).__init__()
        self.name = 'BCE'

    def compute(self, answer, label):
        assert len(answer) == len(label)

        answer = torch.Tensor(answer)
        label = torch.Tensor(label)

        BCE = F.binary_cross_entropy(answer, label, reduction='mean')
        return round(BCE.item(),4)

class F1(object):
    def __init__(self):
        super(F1, self).__init__()
        self.name = 'F1'

    def compute(self, answer, label):
        assert len(answer) == len(label)

        answer = np.array(answer)
        label = np.array(label)

        result = f1_score(y_true=label, y_pred=answer)
        return round(result,4)


class Precision(object):
    def __init__(self):
        super(Precision, self).__init__()
        self.name = 'Precision'

    def compute(self, answer, label):
        assert len(answer) == len(label)

        answer = np.array(answer)
        label = np.array(label)

        result = precision_score(y_true=label, y_pred=answer)
        return round(result,4)

class Recall(object):
    def __init__(self):
        super(Recall, self).__init__()
        self.name = 'Recall'

    def compute(self, answer, label):
        assert len(answer) == len(label)

        answer = np.array(answer)
        label = np.array(label)

        result = recall_score(y_true=label, y_pred=answer)
        return round(result,4)

class RMSE(object):
    def __init__(self):
        super(RMSE, self).__init__()
        self.name = 'RMSE'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = torch.Tensor(answer).squeeze(-1)
        label = torch.Tensor(label)
        #print("Size for RMSE")
        #print("Answer size: ", answer.size())
        #print("Label size: ", label.size())
        RMSE = F.mse_loss(answer, label, reduction='mean').sqrt()
        #SE = F.mse_loss(answer, label, reduction='none')
        #print("SE: ", SE)
        #MSE = SE.mean()
        #print("MSE: ", MSE)
        #RMSE = MSE.sqrt()
        return round(RMSE.item(),4)


class MAE(object):
    def __init__(self):
        super(MAE, self).__init__()
        self.name = 'MAE'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = torch.Tensor(answer)
        label = torch.Tensor(label)
        #print("Size for MAE")
        MAE = F.l1_loss(answer, label, reduction='mean')
        return MAE.item()


class MSE(object):
    def __init__(self):
        super(MSE, self).__init__()
        self.name = 'MSE'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = torch.Tensor(answer)
        label = torch.Tensor(label)
        #print("Size for MAE")
        MSE = F.mse_loss(answer, label, reduction='mean')
        return MSE.item()


class PCC(object):
    def __init__(self):
        super(PCC, self).__init__()
        self.name = 'PCC'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = np.array(answer)
        label = np.array(label)
        #print("Size for MAE")
        pcc = np.corrcoef(answer, label)
        return pcc[0][1]

class R2(object):
    def __init__(self):
        super(R2, self).__init__()
        self.name = 'R2'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = np.array(answer)
        label = np.array(label)
        #print("Size for MAE")
        r_squared = r2_score(answer, label)
        return r_squared


class SRCC(object):
    def __init__(self):
        super(SRCC, self).__init__()
        self.name = 'Spearman Rank Cor. Coef.'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        #print("Size for MAE")
        srcc = spearmanr(answer, label)
        return srcc[0]
