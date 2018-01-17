##  Author          :   Wu Kaixin 
##  Date            :   2018/1/3
##  Email           :   wukaixin_neu@163.com
##  Last Modified in:   NEU NLP Lab., shenyang   

"""
    Beam search implementation in PyTorch.
"""

import torch
from torch.autograd import Variable

class Beam(object):
    """
        Ordered beam of candidate outputs.
    """

    def __init__(self, beam_size, vocab, cuda=False):
        """Initialize params."""

        self.beam_size = beam_size
        self.done = False
        self.pad = vocab['<pad>']
        self.sos = vocab['<sos>']
        self.eos = vocab['<eos>']
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(beam_size).zero_()

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(beam_size).fill_(self.pad)]

        # As a sign of termination, if we search best candidate from the back to the front.
        self.nextYs[0][0] = self.sos

        # The attentions (matrix) for each time.
        self.attn = []

    # Get the outputs for the current timestep.
    def get_current_state(self):
        """Get state of beam."""
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
        """Get the backpointer to the beam at this step."""
        return self.prevKs[-1]

    #  Given prob over words for every last beam `wordLk` and attention
    #   `attnOut`: Compute and update the beam search.
    #
    # Parameters:
    #
    #     * `wordLk`- probs of advancing from the last step (K x words)
    #     * `attnOut`- attention at the last step
    #
    # Returns: True if beam search is complete.

    def advance(self, workd_lk):  ## workd_lk[beam_size, output_size]
        """Advance the beam."""
        num_words = workd_lk.size(1) ## output_size

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beam_lk = workd_lk + self.scores.unsqueeze(1).expand_as(workd_lk)
        else:
            beam_lk = workd_lk[0] ## if the first advance operation

        flat_beam_lk = beam_lk.view(-1)  ## flat

        bestScores, bestScoresId = flat_beam_lk.topk(self.beam_size, 0, True, True)
        self.scores = bestScores

        prev_k = bestScoresId / num_words
        self.prevKs.append(prev_k)  ## record beam id
        self.nextYs.append(bestScoresId - prev_k * num_words) ## record word id

        # End condition is when top-of-beam is EOS.
        # (Top-of-beam is the best candidate in current in every step.)
        if int(self.nextYs[-1][0]) == self.eos:
            self.done = True

        return self.done

    def sort_best(self):
        """Sort the beam."""
        return torch.sort(self.scores, 0, True)

    # Get the score of the best in the beam.
    def get_best(self):
        """Get the most likely candidate."""
        scores, ids = self.sort_best()
        return scores[0], ids[0]

    def get_final(self):

        steps = len(self.prevKs)-1
        index = int(self.prevKs[steps][0])

        tokens = []
        # From the back to the front
        while int(self.nextYs[steps][index]) != self.sos:
            token = int(self.nextYs[steps][index])
            tokens.append(token)

            steps -= 1
            index = int(self.prevKs[steps][index])

        return tokens


    # Walk back to construct the full hypothesis.
    #
    # Parameters.
    #
    #     * `k` - the position in the beam to construct.
    #
    # Returns.
    #
    #     1. The hypothesis
    #     2. The attention at each time step.
    def get_hyp(self, k):
        """Get hypotheses."""
        hyp = []
        # print(len(self.prevKs), len(self.nextYs), len(self.attn))
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            k = self.prevKs[j][k]

        return hyp[::-1]
