import math
import torch
from torch import nn
from torch.autograd import Variable
from utils import get_threshold
import torch.nn.functional as F

class LayerNorm(nn.Module):
    '''
    layer normalization
    '''
    '''
    https://github.com/seba-1511/lstms.pth/blob/master/lstms/normalize.py
    '''
    def __init__(self, input_size, eps=1e-6):
        '''
        :param features: input
        :param eps:  prevent from dividing by zero
        '''
        super(LayerNorm, self).__init__()
        self.input_size = input_size
        self.eps = eps
        self.gamma = nn.Parameter(torch.Tensor(input_size))
        self.beta = nn.Parameter(torch.Tensor(input_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = get_threshold(self.input_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        mean = input.mean(-1, keepdim=True)   ## mean
        std = input.std(-1, keepdim=True)     ## standard-deviation
        return self.gamma * (input - mean) / (std + self.eps) + self.beta

class GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True,
                 dropout=0., layer_normalization=True):
        super(GRUCell,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.layer_normalization=layer_normalization
        self.dropout = nn.Dropout(dropout)

        gate_size = 3 * hidden_size

        self.w_ih = nn.Parameter(torch.Tensor(gate_size, input_size))
        self.w_hh = nn.Parameter(torch.Tensor(gate_size, hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(gate_size))
        self.b_hh = nn.Parameter(torch.Tensor(gate_size))

        if layer_normalization:
            self.ln_ih = LayerNorm(2 * hidden_size)
            self.ln_ih3 = LayerNorm(hidden_size)
            self.ln_hh = LayerNorm(2 * hidden_size)
            self.ln_hh3 = LayerNorm(hidden_size)

        self.reset_parameters()

    def forward(self, input):
        '''
        :param input: [max_len, batch, hidden_size]
        :return:
        '''
        max_len, batch, hidden_size = input.size()
        hidden = Variable(torch.zeros(batch, hidden_size)).cuda()
        encoder_output = Variable(torch.zeros(max_len, batch, hidden_size)).cuda()
        for t in range(max_len):
            x = input[t]
            w_ih, w_hh, b_ih, b_hh = self.dropout(self.w_ih), self.dropout(self.w_hh), self.dropout(self.b_ih), self.dropout(self.b_hh)

            out = F.sigmoid(self.ln_ih(F.linear(x, w_ih[:2*self.hidden_size], b_ih[:2*self.hidden_size])) + self.ln_hh(F.linear(hidden, w_hh[:2*self.hidden_size], b_hh[:2*self.hidden_size])))
            z, r = out.chunk(2, 1)
            h_ = F.tanh(self.ln_ih3(F.linear(x, w_ih[2*self.hidden_size:], b_ih[2*self.hidden_size])) +
                        self.ln_hh3(F.linear(r * hidden, w_hh[2*self.hidden_size:], b_hh[2*self.hidden_size:])))

            hidden = (1. - z) * hidden + z * h_
            encoder_output[t] = hidden

        return encoder_output, hidden

    def reset_parameters(self):
        for weight in self.parameters():
            stdv = get_threshold(weight.size(0))
            if weight.dim() == 2:
                stdv = get_threshold(weight.size(0), weight.size(1))

            weight.data.uniform_(-stdv, stdv)

class BiGRU(nn.Module):

    def __init__(self, forward_gru, backward_gru):
        super(BiGRU, self).__init__()
        self.forward_gru = forward_gru
        self.backward_gru = backward_gru

    def forward(self, input, hidden=None):
        '''
        :param input: [max_len, batch, hidden_size]
        :return:
        '''
        forward_encode_output, forward_hidden = self.forward_gru(input)
        reversed_indexes = [i for i in range(input.size(0)-1, -1, -1)]
        backward_encode_output, backward_hidden = self.backward_gru(input[reversed_indexes])

        encode_output = torch.cat([forward_encode_output, backward_encode_output], -1)
        hidden = Variable(torch.zeros(2, forward_hidden.size(0), forward_hidden.size(1))).cuda()
        hidden[0] = forward_hidden
        hidden[1] = backward_hidden

        return encode_output, hidden


