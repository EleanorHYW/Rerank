import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import tanh, sigmoid
from torch.nn.functional import softmax


class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout, bidir):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim // 2 if bidir else hidden_dim
        self.n_layers = n_layers * 2 if bidir else n_layers
        self.num_direction = 2 if bidir else 1
        self.bidir = bidir
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, bidirectional=bidir)
        state_size = (self.n_layers * self.num_direction, 1, self.hidden_dim)
        self.init_h = nn.Parameter(torch.zeros(state_size))
        self.init_c = nn.Parameter(torch.zeros(state_size))

    def forward(self, inputs, h0=None, c0=None):
        batch_size = inputs.size(0)
        inputs = inputs.transpose(1, 0)
        if h0 is None and c0 is None:
            state_size = (self.n_layers * self.num_direction, batch_size, self.hidden_dim)
            h0 = self.init_h.expand(*state_size).contiguous()
            c0 = self.init_c.expand(*state_size).contiguous()
        outputs, hn, cn = self.lstm(inputs, h0, c0)
        # LSTM outputs and (h, c)
        return outputs.transpose(1, 0), hn, cn


class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Attention, self).__init__()
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.context_linear = nn.Linear(input_dim, hidden_dim)
        self.v = Parameter(torch.FloatTensor(hidden_dim), requires_grad=False)
        self._inf = Parameter(torch.FloatTensor([float('inf')]), requires_grad=False)

    def forward(self, input, context, mask):
        inf = self._inf.unsqueeze(1).expand(mask.size())

        # context bsz, seq_len, dim
        seq_len = context.size(1)
        bsz = context.size(0)

        # inp bsz, dim
        # input bsz, dim
        inp = self.input_linear(input).unsqueeze(2)
        # inp bsz, dim, 1
        inp = inp.expand(-1, -1, seq_len)
        # inp bsz, dim, seq_len

        ctx = self.context_linear(context).transpose(1, 2)
        # ctx bsz, dim, seq_len
        # v bsz, 1, dim
        v = self.v.unsqueeze(0).expand(bsz, -1).unsqueeze(1)

        attn = torch.matmul(v, tanh(inp + ctx))
        attn = attn.squeeze(1)
        # attn bsz, seq_len

        if mask.any():
            attn[mask] = inf[mask]
        # bsz, seq_len
        attn_softmax = softmax(attn, dim=-1)

        hidden = torch.matmul(ctx, attn_softmax.unsqeeze(2))
        # hidden bsz, dim, 1
        hidden = hidden.squeeze(2)
        # hidden bsz, dim

        return hidden, attn_softmax

    # Todo
    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)


class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.input_to_hidden = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn = Attention(hidden_dim, hidden_dim)

        self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.runner = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, encoder_inputs, decoder_input, hidden, context):
        bsz = encoder_inputs.size(0)
        seq_len = encoder_inputs.size(1)
        mask = self.mask.repeat(seq_len).unsqueeze(0).repeat(bsz, 1)
        runner = self.runner.repeat(seq_len)
        for i in range(seq_len):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(bsz, -1).long()

        # self.attn.init_int(mask.size())

        def step(x, hidden):
            h, c = hidden
            gates = self.input_to_hidden(x), self.hidden_to_hidden(h)
            input, forget, cell, out = gates.chunk(4, 1)
            input = sigmoid(input)
            forget = sigmoid(forget)
            cell = tanh(cell)
            out = sigmoid(out)
            c_t = (forget * c) + (input * cell)
            h_t = out * tanh(c_t)
            hidden_t, output = self.attn(h_t, context, torch.eq(mask, 0))
            hidden_t = tanh(self.hidden_out(torch.cat(hidden_t, h_t), 1))
            return hidden_t, c_t, output
