import torch
from cs336_basics.nn_utils import softmax

class CrossEntropy:
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        self.batch_size, self.vocab_size = inputs.shape

    def forward(self):
        y_pred = softmax(self.inputs, dim=-1)
        p = y_pred[range(self.batch_size, self.targets)]
        return -torch.sum(torch.log(p))