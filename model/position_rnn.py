import torch.nn as nn
from articulate.utils.torch import RNN
import torch
from torch.nn.functional import relu
from torch.nn.utils.rnn import *


class PositionRNN(nn.Module):

    def __init__(self, ckpt_path=None):
        super(PositionRNN, self).__init__()
        self.rnn = RNN(input_size=6 * 3 + 6 * 9 + 33 * 3 + 23 * 3,
                        output_size=3,
                        hidden_size=1024,
                        num_rnn_layer=2,
                        dropout=0.4)
        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path))

    def load_state_dict(self, state_dict, strict: bool = True):
        print("Loading PositionRNN state dict")
        self.rnn.load_state_dict(state_dict, strict)

    def forward(self, x):
        return self.rnn(x)
