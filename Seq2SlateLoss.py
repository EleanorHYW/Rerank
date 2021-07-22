import torch
import torch.nn as nn
from torch import Tensor
import math

class Seq2SlateLoss(nn.Module):

    def __init__(self, max_length) -> None:
        super(Seq2SlateLoss, self).__init__()
        self.weight = torch.Tensor([1 / math.log(i + 1) for i in range(1, max_length + 1)])

    def forward(self, logits: Tensor, label: Tensor) -> Tensor:
        """
        Loss for seq2slate:
        input: 
        logits: [seq_len, seq_len], for j_th logit in logits, logit[i](logits[j][i]) represents the probability of choosing item i in position j
        label: [seq_len, 1] for every item
        """
        # import pdb; pdb.set_trace()
        batch_size = len(logits)
        label = label.to(logits[0].device)
        loss_sum = 0
        for i in range(0, batch_size):
            logit = logits[i]
            inf_mask = logit == 0
            labe = label[i]
            length = logit.size(1)
            w1 = self.weight[:length].view(1, -1, 1)
            weight = w1.expand(*logit.size()).to(logit.device)
            y = label[i][:length].expand(*logit.size())
            iy = torch.mul(y, weight)
            loss = - torch.mul(torch.log(logit[~inf_mask]), iy[~inf_mask].float()).sum()
            seq_loss = loss / (y[~inf_mask].sum() / length + 1/math.exp(1))
            loss_sum = loss_sum + seq_loss
        return loss_sum / batch_size

