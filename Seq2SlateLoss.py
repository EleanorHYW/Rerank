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
        k = logits[0].size(0)
        label = label.to(logits[0].device)
        loss_sum = 0
        # logits[i] [sample_num, 1, seq_len, seq_len]
        for i in range(0, batch_size):
            # seq_len, k, seq_len
            x = logits[i].squeeze(1).transpose(0, 1)
            length = x.size(0)
            w1 = self.weight[:length].view(-1, 1, 1)
            weight = w1.expand(*x.size()).to(x.device)
            y = label[i][:length].expand(*x.size())
            iy = torch.mul(y, weight)
            mask = x == 0
            loss = - torch.mul(torch.log(x[~mask]), iy[~mask].float()).sum()
            seq_loss = (loss / (y[~mask].sum() / k + 1/math.exp(1))) / k
            loss_sum = loss_sum + seq_loss
        loss_average = loss_sum / batch_size 
        return loss_average

