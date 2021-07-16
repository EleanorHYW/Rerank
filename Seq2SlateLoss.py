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
            inf_mask = logits[i] == 0
            logit = logits[i]
            # logit = torch.log(logits[i] + 0.00001)
            # logit = torch.clamp(logit, min=-10, max=10)
            labe = label[i]
            length = logit.size(1)
            seq_loss = 0
            for j in range(length):
                x = logit.squeeze(0)[j]
                mask = inf_mask[0][j]
                y = labe[:length]
                loss = - torch.mul(torch.log(x[~mask]), y[~mask].float()).sum()
                # c = torch.where(torch.isinf(x), torch.full_like(x, 0), x)
                # loss2 = - torch.mul(c, y.float()).sum()
                # loss = - torch.mul(c, y.float()).sum()
                # if loss != loss2:
                #     import pdb; pdb.set_trace()
                #     print("Different loss calculated. loss={}, loss2={}".format(loss, loss2))
                #     # assert loss == loss2, "Different loss calculated. loss={}, loss2={}".format(loss, loss2)
                loss = loss / (y[~mask].sum() + 1/math.exp(1))
                seq_loss = seq_loss + loss * self.weight[j]
            loss_sum = loss_sum + seq_loss
        loss_average = loss_sum / batch_size 
        return loss_average

