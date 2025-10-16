import torch
from torch.optim.lr_scheduler import _LRScheduler
import math

class WarmUpCosineDecayScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super(WarmUpCosineDecayScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            return [base_lr * step / self.warmup_steps for base_lr in self.base_lrs]
        else:
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
            decayed = (1 - self.min_lr) * cosine_decay + self.min_lr
            return [base_lr * decayed for base_lr in self.base_lrs]