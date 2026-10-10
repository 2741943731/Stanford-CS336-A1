import torch
from cs336_basics.nn_utils import softmax

def CrossEntropy(y, targets):
    """
    y (batch, vocab) TLM输出的logits值
    tagets (batch,) 真实标签序号的数组
    """
    batch_size = y.shape[0]
    y_max = torch.max(y, keepdim=True, dim=-1)[0]
    y = y - y_max
    y_exp = torch.exp(y)
    log_y = y - torch.log(y_exp.sum(dim=-1, keepdim=True))
    p = log_y[range(batch_size), targets]
    return -torch.mean(p)
    
