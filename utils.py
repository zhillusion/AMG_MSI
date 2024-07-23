from torch import nn
import torch
from omegaconf import OmegaConf, DictConfig
from typing import Optional, List, Any, Dict, Tuple, Union
from sklearn import metrics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def init_weights(module: nn.Module):
    """
    Initialize one module. It uses xavier_norm to initialize nn.Embedding
    and xavier_uniform to initialize nn.Linear's weight.

    Parameters
    ----------
    module
        A Pytorch nn.Module.
    """
    if isinstance(module, nn.Embedding):
        nn.init.xavier_normal_(module.weight)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def ema_model_parameter_ini(model,ema_model):
    for param_main, param_ema in zip(model.parameters(), ema_model.parameters()):
        param_ema.data.copy_(param_main.data)
        param_ema.requires_grad = False
    # for param_ema in ema_model.parameters():
    #     param_ema.detach_()
    # ema_model.eval()
    return ema_model

def ema_model_parameter_update(model,ema_model,theta):
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data.mul_(1-theta).add_(param.data, alpha = theta)
    return ema_model



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def calculate_precision_recall_f1(logits, targets):
    # 将 logits 转换为概率
    probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()
    thresholds = np.unique(probabilities)
    best_threshold = 0
    best_j_index = -1

    # 寻找最佳阈值
    for threshold in thresholds:
        preds_binary = (probabilities >= threshold).astype(int)
        tp = ((preds_binary == 1) & (targets == 1)).sum()
        tn = ((preds_binary == 0) & (targets == 0)).sum()
        fp = ((preds_binary == 1) & (targets == 0)).sum()
        fn = ((preds_binary == 0) & (targets == 1)).sum()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        j_index = sensitivity + specificity
        if j_index > best_j_index:
            best_j_index = j_index
            best_threshold = threshold

    # 使用最佳阈值计算指标
    pred_binary = (probabilities >= best_threshold).astype(int)
    precision = metrics.precision_score(targets, pred_binary, zero_division=0)
    recall = metrics.recall_score(targets, pred_binary)
    f1 = metrics.f1_score(targets, pred_binary)

    return precision, recall, f1


def calculate_accuracy_from_metrics(precision, recall, n_positive, n_negative):
    # Calculate true positives (TP) and false positives (FP)
    tp = precision * n_positive
    fp = tp / precision - tp

    # Calculate false negatives (FN) and true negatives (TN)
    fn = (tp / recall) - tp
    tn = n_negative - fp

    # Calculate accuracy
    accuracy = (tp + tn) / (n_positive + n_negative)
    return accuracy



class AUCRecorder(object):
    def __init__(self):
        self.prediction = []
        self.target = []
        
    def update(self, prediction, target):
        self.prediction = self.prediction + prediction.tolist()
        self.target = self.target + target.tolist()

    def save_to_file(self, filepath):
        data = np.column_stack((self.prediction, self.target))
        np.savetxt(filepath, data, delimiter=",", header="prediction,target", comments="")


    @property
    def auc(self):
        prediction = np.array(self.prediction)
        target = np.array(self.target)
        fpr, tpr, thresholds = metrics.roc_curve(target, prediction, pos_label=1)
        auc = metrics.auc(fpr, tpr)  
        return auc

    def draw_roc(self, path):
        prediction = np.array(self.prediction)
        target = np.array(self.target)
        fpr, tpr, thresholds = metrics.roc_curve(target, prediction, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        plt.figure(figsize=(6, 6), dpi=300)  # 调整图形大小和分辨率
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')  # 加粗线条并指定颜色
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # 修改对角线样式
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)  # 增加字体大小
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('Receiver Operating Characteristic', fontsize=16)  # 增加标题字体大小
        plt.legend(loc="lower right", fontsize=12)  # 调整图例大小
        plt.grid(True)  # 添加网格
        plt.savefig(path, bbox_inches='tight')  # 保存图形时包括全部绘制区域


