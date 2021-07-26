import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch import tanh, sigmoid
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence

class Encoder(nn.Module):
    """
    Encoder class for Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim,
                 n_layers,
                 dropout,
                 bidir):
        """
        Initiate Encoder

        :param Tensor embedding_dim: Number of embbeding channels
        :param int hidden_dim: Number of hidden units for the LSTM
        :param int n_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim//2 if bidir else hidden_dim
        self.n_layers = n_layers*2 if bidir else n_layers
        self.bidir = bidir
        self.lstm = nn.LSTM(embedding_dim,
                            self.hidden_dim,
                            n_layers,
                            dropout=dropout,
                            bidirectional=bidir)

        # Used for propagating .cuda() command
        self.h0 = Parameter(torch.zeros(1), requires_grad=False)
        self.c0 = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs, masks,
                hidden):
        """
        Encoder - Forward-pass

        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor hidden: Initiated hidden units for the LSTMs (h, c)
        :return: LSTMs outputs and hidden units (h, c)
        """
        # bsz * seq_len * embedding_dim
        # embedded_inputs = embedded_inputs.permute(1, 0, 2)

        outputs, hidden = self.lstm(embedded_inputs, hidden)
        outputs = pad_packed_sequence(outputs, batch_first=True)
        return outputs[0], hidden, outputs[1]

    def init_hidden(self, embedded_inputs):
        """
        Initiate hidden units

        :param Tensor embedded_inputs: The embedded input of Pointer-NEt
        :return: Initiated hidden units for the LSTMs (h, c)
        """

        batch_size = embedded_inputs.batch_sizes[0]

        # Reshaping (Expanding)
        h0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)
        c0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)
        nn.init.uniform_(h0, -1, 1)
        nn.init.uniform_(c0, -1, 1)
        return h0, c0


class Attention(nn.Module):
    """
    Attention model for Pointer-Net
    """

    def __init__(self, input_dim,
                 hidden_dim):
        """
        Initiate Attention

        :param int input_dim: Input's diamention
        :param int hidden_dim: Number of hidden units in the attention
        """

        super(Attention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self._inf = Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)
        self.tanh = tanh
        self.softmax = nn.Softmax(dim=-1)

        # Initialize vector V
        nn.init.uniform_(self.V, -1, 1)

    def forward(self, input,
                context,
                mask):
        """
        Attention - Forward-pass

        :param Tensor input: Hidden state h
        :param Tensor context: Attention context
        :param ByteTensor mask: Selection mask
        :return: tuple of - (Attentioned hidden state, Alphas)
        """

        # (batch, hidden_dim, seq_len)
        inp = self.input_linear(input).unsqueeze(2).expand(-1, -1, context.size(1))
        seq_len = inp.size(2)

        # (batch, hidden_dim, seq_len)
        context = context.permute(0, 2, 1)
        ctx = self.context_linear(context)

        # (batch, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

        # (batch, seq_len)
        att = torch.bmm(V, self.tanh(inp + ctx)).squeeze(1)
        self.inf = self._inf.expand(*mask.size())
        attn = self.softmax(att)
        if len(att[mask]) > 0:
            att[mask] = self.inf[mask]
        
        alpha = self.softmax(att)
        # smooth
        # alpha = torch.where(torch.isnan(alpha), torch.full_like(alpha, 0), alpha)

        hidden_state = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)

        return hidden_state, alpha, attn

    def init_inf(self, mask_size):
        self.inf = self._inf.expand(mask_size[1])


class Decoder(nn.Module):
    """
    Decoder model for Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim):
        """
        Initiate Decoder

        :param int embedding_dim: Number of embeddings in Pointer-Net
        :param int hidden_dim: Number of hidden units for the decoder's RNN
        """

        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.input_to_hidden = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.att = Attention(hidden_dim, hidden_dim)

        # Used for propagating .cuda() command
        # self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.runner = Parameter(torch.arange(500), requires_grad=False)

    def forward(self, 
                embedded_inputs,
                decoder_input,
                hidden,
                context,
                masks,
                sampling=False):
        """
        Decoder - Forward-pass

        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor decoder_input: First decoder's input
        :param Tensor hidden: First decoder's hidden states
        :param Tensor context: Encoder's outputs
        :return: (Output probabilities, Pointers indices), last hidden state
        """
        batch_size = embedded_inputs.batch_sizes[0]
        seq_lens = masks.sum(dim=1)
        seq_idx = torch.Tensor([seq_lens[:i].sum().item() for i in range(batch_size + 1)]).to(seq_lens.device)
        input_length = max(seq_lens)

        # # (batch, seq_len)
        # mask = self.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1)
        self.att.init_inf(masks.size())

        # Generating arange(input_length + 1), broadcasted across batch_size
        runner = self.runner[:input_length].unsqueeze(0).long()
        # runner = runner.unsqueeze(0).expand(batch_size, -1).long()

        def step(x, hidden, cxt, msk):
            """
            Recurrence step function

            :param Tensor x: Input at time t
            :param tuple(Tensor, Tensor) hidden: Hidden states at time t-1
            :return: Hidden states at time t (h, c), Attention probabilities (Alpha)
            """

            # Regular LSTM
            h, c = hidden

            gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)
            input, forget, cell, out = gates.chunk(4, 1)

            input = sigmoid(input)
            forget = sigmoid(forget)
            cell = tanh(cell)
            out = sigmoid(out)

            c_t = (forget * c) + (input * cell)
            h_t = out * tanh(c_t)

            # Attention section
            hidden_t, output, attn = self.att(h_t, cxt, torch.eq(msk, 0))
            # output is softmaxed attn score
            hidden_t = tanh(self.hidden_out(torch.cat((hidden_t, h_t), 1)))

            return hidden_t, c_t, output, attn

        # Recurrence loop
        outputs = []
        pointers = []
        atts = []
        for idx in range(batch_size):
            output = []
            pointer = []
            att = []
            inp = decoder_input[idx].unsqueeze(0)
            hid = (hidden[0][idx].unsqueeze(0), hidden[1][idx].unsqueeze(0))
            length = seq_lens[idx].item()
            cxt = context[idx][:length].unsqueeze(0)
            msk = masks[idx][:length].unsqueeze(0)
            # import pdb; pdb.set_trace()
            emb = embedded_inputs.data[int(seq_idx[idx].item()) : int(seq_idx[idx + 1].item())].unsqueeze(0)
            for _ in range(seq_lens[idx]):
                h_t, c_t, outs, attn = step(inp, hid, cxt, msk)
                hid = (h_t, c_t)
                # Masking selected inputs
                masked_outs = outs * msk
                # max_probs, indices = masked_outs.max(1)
                # random sampling according to probability instead of choosing max
                if sampling:
                    indices = torch.multinomial(masked_outs, 1).view(-1)
                else:
                    _, indices = masked_outs.max(1)
                # max_probs = masked_outs[0][indices]
                one_hot_pointers = (runner[0][:length] == indices.unsqueeze(1).expand(-1, outs.size()[1])).float()
                # Update mask to ignore seen indices
                msk = msk * (1 - one_hot_pointers)

                # Get embedded inputs by max indices
                embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1, self.embedding_dim).byte()
                inp = emb[embedding_mask.bool()].unsqueeze(0)
                # warning: may cause problem because outs is the total softmax rather than softmax over unseened indices
                output.append(outs)
                att.append(attn)
                pointer.append(indices)
            outputs.append(torch.stack(output).permute(1, 0, 2))
            atts.append(torch.stack(att).permute(1, 0, 2))
            pointers.append(torch.stack(pointer).transpose(0, 1))
        assert len(outputs) == batch_size
        # add another assert
        return (outputs, pointers), atts, hidden


class PointerNet(nn.Module):
    """
    Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim,
                 lstm_layers,
                 dropout,
                 bidir=False):
        """
        Initiate Pointer-Net

        :param int embedding_dim: Number of embbeding channels
        :param int hidden_dim: Encoders hidden units
        :param int lstm_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(PointerNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.bidir = bidir
        # self.embedding = nn.Linear(2, embedding_dim)
        self.encoder = Encoder(embedding_dim,
                               hidden_dim,
                               lstm_layers,
                               dropout,
                               bidir)
        self.dropout = nn.Dropout(p=dropout)
        self.decoder = Decoder(embedding_dim, hidden_dim)

        # change decoder_input into history item embedding

        # decoder_input0 <go>
        # requires_grad = True or False ? 
        self.decoder_input0 = Parameter(torch.FloatTensor(embedding_dim), requires_grad=True)

        # Initialize decoder_input0
        nn.init.uniform_(self.decoder_input0, -1, 1)

    def forward(self, inputs, masks, sampling=False):
        """
        PointerNet - Forward-pass

        :param Tensor inputs: Input sequence
        # :param Tensor history: History sequence
        :return: Pointers probabilities and indices
        """
        batch_size = inputs.batch_sizes[0]

        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)
        # decoder_input0 = torch.mean(history, dim=1)

        embedded_inputs = inputs.to(decoder_input0.device)
        # embedded_inputs = self.embedding(inputs).view(batch_size, input_length, -1)
        masks = masks.to(decoder_input0.device)

        encoder_hidden0 = self.encoder.init_hidden(embedded_inputs)
        encoder_outputs, encoder_hidden, seq_lens = self.encoder(embedded_inputs, masks, encoder_hidden0)
        if self.bidir:
            decoder_hidden0 = (torch.cat([_ for _ in encoder_hidden[0][-2:]], dim=-1),
                               torch.cat([_ for _ in encoder_hidden[1][-2:]], dim=-1))
        else:
            decoder_hidden0 = (encoder_hidden[0][-1],
                               encoder_hidden[1][-1])
        encoder_outputs = self.dropout(encoder_outputs)
        (outputs, pointers), atts, decoder_hidden = self.decoder(embedded_inputs,
                                                           decoder_input0,
                                                           decoder_hidden0,
                                                           encoder_outputs,
                                                           masks,
                                                           sampling=False)

        return  outputs, pointers, atts