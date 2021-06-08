import torch

class ptNetEncoder(torch.nn.Module):
    """ Creation of a class to output only the last hidden state from the encoder """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, embedded_inputs, encoder_h0=None, encoder_c0=None):
        return self.encoder(embedded_inputs, h0=encoder_h0, c0=encoder_c0)


class ptNetDecoder(torch.nn.Module):
    """ Creation of a class to output the decoder outputs and pointers """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, embedded_inputs, decoder_input0, decoder_h0, decoder_c0, encoder_outputs):
        (outputs, pointers), decoder_hn, deocder_cn = self.decoder(embedded_inputs,
                                                           decoder_input0,
                                                           decoder_h0,
                                                           decoder_c0,
                                                           encoder_outputs)

        return outputs, pointers, decoder_hn, deocder_cn