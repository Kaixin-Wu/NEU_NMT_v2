import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from beam_search import Beam
from utils import get_threshold
from layer import GRUCell, BiGRU

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.2):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(input_size, embed_size)
        ## stdv = 1. / math.sqrt(embed_size)
        ## self.embed.weight.data.normal_(0, stdv)
        stdv = get_threshold(input_size, embed_size)
        self.embed.weight.data.uniform_(-stdv, stdv)
        
        # forward_gru = GRUCell(embed_size, hidden_size, dropout=dropout)
        # backward_gru = GRUCell(embed_size, hidden_size, dropout=dropout)
        # self.gru = BiGRU(forward_gru, backward_gru)

        self.gru = nn.GRU(embed_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)

    def forward(self, src, hidden=None):
        embedded = self.embed(src)
        ## self.gru.flatten_parameters()   ## Edit by Wu Kaixin 2018/1/9
        outputs, hidden = self.gru(embedded, hidden)

        '''
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        '''
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 3, hidden_size)
        stdv = get_threshold(hidden_size * 3, hidden_size)
        self.attn.weight.data.uniform_(-stdv, stdv)
        self.attn.bias.data.zero_()

        self.v = nn.Parameter(torch.rand(hidden_size))
        # stdv = 1. / math.sqrt(self.v.size(0))
        # self.v.data.uniform_(-stdv, stdv)
        stdv = get_threshold(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*2H]
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = self.attn(torch.cat([hidden, encoder_outputs], 2))
        energy = F.tanh(energy.transpose(1, 2))  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_size, embed_size)
        stdv = get_threshold(output_size, embed_size)
        self.embed.weight.data.uniform_(-stdv, stdv)
        ## stdv = 1. / math.sqrt(embed_size)
        ## self.embed.weight.data.normal_(0, stdv)

        ### self.dropout = nn.Dropout(dropout, inplace=True)
        # self.init_state = nn.Linear(hidden_size, hidden_size) ## Edit by Wu Kaixin 1/14
        # stdv = get_threshold(hidden_size, hidden_size)
        # self.init_state.weight.data.uniform_(-stdv, stdv)
        #vself.init_state.bias.data.zero_()

        self.attention = Attention(hidden_size)
        # self.gru = nn.GRU(2 * hidden_size + embed_size, hidden_size,
        #                   n_layers, dropout=dropout)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size,
                          n_layers, dropout=dropout)        

        self.att_hidden_state = nn.Linear(3 * hidden_size, hidden_size)
        stdv = get_threshold(3 * hidden_size, hidden_size)
        self.att_hidden_state.weight.data.uniform_(-stdv, stdv)
        self.att_hidden_state.bias.data.zero_()

        # self.out = nn.Linear(hidden_size * 3, output_size)
        # stdv = get_threshold(hidden_size * 3, output_size)

        self.out = nn.Linear(hidden_size, output_size)
        stdv = get_threshold(hidden_size, output_size)
        self.out.weight.data.uniform_(-stdv, stdv)
        self.out.bias.data.zero_()
        ## self.out.weight.data.uniform_(-0.01, 0.01)

    def forward(self, input, last_hidden, encoder_outputs, input_feeding=None):
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
        if input_feeding is None:
            input_feeding = Variable(torch.zeros(embedded.size(1), self.hidden_size)).cuda()
        input_feeding = input_feeding.unsqueeze(0)
        
        # print(embedded.size())
        # print(input_feeding.size())        
        rnn_input = torch.cat([embedded, input_feeding], 2)
        output, last_hidden = self.gru(rnn_input, last_hidden)
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        context = context.squeeze(1) # [batch, 2hidden_size]
        output = output.squeeze(0)   # [batch, hidden_size]
        output = torch.cat([output, context], 1) # [batch, 3hidden_size]
        output = F.tanh(self.att_hidden_state(output)) # [batch, hidden_size]

        return output, last_hidden, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, trg):
        '''
        :param src:  [src_max_len, batch]
        :param trg:  [trg_max_len, batch]
        :return:
        '''
        encoder_output, hidden = self.encoder(src)
        '''
            ## src: [src_max_len, batch]
            ## encoder_output: [src_max_len, batch, hidden_size]
            ## hidden: (num_layers * num_directions, batch, hidden_size) -> [2, batch, hidden_size]
        '''
        hidden = hidden[:self.decoder.n_layers]
        # hidden = F.tanh(self.decoder.init_state(hidden))

        batch_size = src.size(1)
        max_len = trg.size(0)
        hidden_size = self.decoder.hidden_size
        ## vocab_size = self.decoder.output_size
        # outputs = Variable(torch.zeros(max_len-1, batch_size, 3 * hidden_size)).cuda()
        outputs = Variable(torch.zeros(max_len-1, batch_size, hidden_size)).cuda()

        output = Variable(trg.data[0, :]) # sos [batch]
        input_feeding = None
        for t in range(1, max_len):
            # output: [batch] -> [batch, 3hidden_size]
            output, hidden, attn_weights = self.decoder(output, hidden, encoder_output, input_feeding)

            outputs[t-1] = output
            input_feeding = output
            output =Variable(trg.data[t]).cuda()

        transform_output = self.decoder.out(outputs) # [max_len-1, batch, output_size]
        softmax_output = F.log_softmax(transform_output, dim=2)  # [max_len-1, batch, output_size]

        return softmax_output
    
    '''
    def forward(self, src, trg, teacher_forcing_ratio=1.0):
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len-1, batch_size, vocab_size)).cuda()

        encoder_output, hidden = self.encoder(src)
        hidden = hidden[:self.decoder.n_layers]
        output = Variable(trg.data[0, :])  # sos
        for t in range(1, max_len):
            self.decoder.flatten_parameters()  ## Edit by Wu Kaixin 2018/1/9
            output, hidden, attn_weights = self.decoder(
                    output, hidden, encoder_output)
            outputs[t-1] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            output = Variable(trg.data[t] if is_teacher else top1).cuda()
        return outputs # [max_len-1, batch, vocab_size]
    '''

    def translate(self, src, trg, beam_size, Lang2):
        ''' beam search decoding. '''
        '''
        :param src:   [src_max_len, batch]    ## batch = 1
        :param trg:   [trg_max_len, batch]    ## batch = 1
        :param sentence:  [sentence_len]
        :return: best translate candidate
        '''
        max_len = trg.size(0)
        encoder_output, hidden = self.encoder(src)
        '''
            ## src: [src_max_len, batch]
            ## encoder_output: [src_max_len, batch, hidden_size]
            ## hidden: (num_layers * num_directions, batch, hidden_size) -> [2, batch, hidden_size]
        '''
        hidden = hidden[:self.decoder.n_layers]  # [n_layers, batch, hidden_size]
        # trg: [trg_max_len, batch]
        output = Variable(trg.data[0, :])  # sos  [batch]

        beam = Beam(beam_size, Lang2.vocab.stoi, True)
        input_feeding = None
        for t in range(1, max_len):
            # output:  [batch] -> [batch, output_size]
            output, hidden, attn_weights = self.decoder(
                    output, hidden, encoder_output, input_feeding)
            
            input_feeding = output
            output = self.decoder.out(output) 
            output = F.log_softmax(output, dim=1)

            workd_lk = output
            if output.size(0) == 1:

                output_prob = output.squeeze(0) ## [output_size]
                workd_lk = output_prob.expand(beam_size, output_prob.size(0))  ## [beam_size, output_size]

                # [n_layers, batch, hidden_size]
                hidden = hidden.squeeze(1)  # [n_layers, hidden_size]
                hidden = hidden.expand(beam_size, hidden.size(0), hidden.size(1)) # [beam_size, n_layers, hidden_size]
                hidden = hidden.transpose(0, 1) # [n_layers, beam_size, hidden_size]
                
                # [src_max_len, batch, hidden_size]
                encoder_output = encoder_output.squeeze(1) ## [src_max_len, hidden_size]
                encoder_output = encoder_output.expand(beam_size, encoder_output.size(0), encoder_output.size(1)) ## [beam_size, src_max_len, hidden_size]
                encoder_output = encoder_output.transpose(0, 1)  ## [src_max_len, beam_size, hidden_size]
                input_feeding = input_feeding.squeeze(0)
                input_feeding = input_feeding.expand(beam_size, input_feeding.size(0))

            flag = beam.advance(workd_lk)
            if flag:
                break

            nextInputs = beam.get_current_state()
            # print("[nextInputs]:", nextInputs)
            output = nextInputs
            # output = Variable(nextInputs).cuda()

            originState = beam.get_current_origin()
            ## print("[origin_state]:", originState)
            hidden = hidden[:, originState]
            input_feeding = input_feeding[originState]

        xx, yy = beam.get_best()
        zz = beam.get_final()
        return xx, yy, zz
