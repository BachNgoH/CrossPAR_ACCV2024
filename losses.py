import torch
import torch.nn as nn
import torch.nn.functional as F

def ratio2weight(targets, ratio):
    ratio = torch.from_numpy(ratio).type_as(targets)

    # --------------------- dangwei li TIP20 ---------------------
    pos_weights = targets * (1 - ratio)
    neg_weights = (1 - targets) * ratio
    weights = torch.exp(neg_weights + pos_weights)


    # --------------------- AAAI ---------------------
    # pos_weights = torch.sqrt(1 / (2 * ratio.sqrt())) * targets
    # neg_weights = torch.sqrt(1 / (2 * (1 - ratio.sqrt()))) * (1 - targets)
    # weights = pos_weights + neg_weights

    # for RAP dataloader, targets element may be 2, with or without smooth, some element must great than 1
    weights[targets > 1] = 0.0

    return weights

class BCELoss(nn.Module):

    def __init__(self, sample_weight=None, size_sum=True, scale=None, tb_writer=None):
        super(BCELoss, self).__init__()

        self.sample_weight = sample_weight
        self.size_sum = size_sum
        self.hyper = 0.8
        self.smoothing = None

    def forward(self, logits, targets):
        #logits = logits[0]

        if self.smoothing is not None:
            targets = (1 - self.smoothing) * targets + self.smoothing * (1 - targets)

        loss_m = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        targets_mask = torch.where(targets.detach().cpu() > 0.5, torch.ones(1), torch.zeros(1))
        if self.sample_weight is not None:
            sample_weight = ratio2weight(targets_mask, self.sample_weight)

            loss_m = (loss_m * sample_weight.cuda())

        # losses = loss_m.sum(1).mean() if self.size_sum else loss_m.mean()
        loss = loss_m.sum(1).mean() if self.size_sum else loss_m.sum()

        return [loss], [loss_m]


def build_loss(loss_type="bce_loss", sample_weight=None, scale=None, size_sum=True):
    if loss_type == "bce_loss":
        return BCELoss(sample_weight=sample_weight, size_sum=size_sum, scale=scale)
    if loss_type == "ce":
        return nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

