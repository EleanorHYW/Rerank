import torch
import torch.nn as nn
from torch import Tensor
import math

class Seq2SlateLoss(nn.Module):

    def __init__(self) -> None:
        super(Seq2SlateLoss, self).__init__()
        self.weight = torch.Tensor([1 / math.log(i + 1) for i in range(1, 101)]).view(-1, 1)

    def forward(self, logits: Tensor, label: Tensor) -> Tensor:
        """
        Loss for seq2slate:
        input: 
        logits: [seq_len, seq_len], for j_th logit in logits, logit[i] represents the probability of choosing item i in position j
        label: [seq_len, 1] for every item
        """
        # import pdb; pdb.set_trace()
        seq_len = logits.size(1)
        batch_size = logits.size(0)

        # step_loss: for every step, [seq_len, 1]
        step_loss = torch.bmm(logits, label.float()).sum(dim=2)
        step_loss_norm = torch.div(step_loss, label.sum(dim=1)).view(step_loss.size(0), step_loss.size(1), -1)
        # seq_loss is weighted sum of step_loss
        seq_loss = torch.matmul(step_loss_norm.transpose(1, 2), self.weight[:seq_len, :]).view(batch_size, -1)
        return seq_loss.sum()