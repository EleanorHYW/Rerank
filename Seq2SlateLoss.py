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
        import pdb; pdb.set_trace()
        batch_size = len(logits)
        label = label.to(logits[0].device)
        loss_sum = 0
        for i in range(0, batch_size):
        # for i in range(1):
            inf_mask = logits[i] == 0
            logit = torch.log(logits[i])
            # logit = torch.clamp(logit, min=-10, max=10)
            logit = torch.log(logit+0.0000001)
            labe = label[i]
            length = logit.size(1)
            seq_loss = 0
            for j in range(length):
            # for j in range(1):
                x = logit.squeeze(0)[j]
                mask = inf_mask[0][j]
                y = labe[:length]
                loss = - torch.matmul(x[~mask].unsqueeze(0), y[~mask].float().unsqueeze(1))
                # loss = - torch.matmul(x.unsqueeze(0), y.float().unsqueeze(1))
                # take care of loss
                loss = loss / (y.sum() + 1/math.exp(1))
                seq_loss = seq_loss + loss * self.weight[j]
            loss_sum = loss_sum + seq_loss
        loss_average = loss_sum / batch_size 
        return loss_average

