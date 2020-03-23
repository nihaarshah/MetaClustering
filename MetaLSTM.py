import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor, optim


class MetaLSTM(nn.Module):

    # TODO: Make this residual LSTM as in Jibo et al.
    def __init__(self, input_size, hidden_size, output_size, n_layers, batch_size):
        super(MetaLSTM, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.initialize()
        self.lstm_cells = nn.ModuleList()
        for i in range(self.n_layers):
            self.lstm_cells.append(nn.LSTMCell(
                input_size=self.input_size if i == 0 else self.hidden_size,
                hidden_size=self.hidden_size
            ))
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def initialize(self):
        self.h_ = [torch.zeros(
            (self.batch_size, self.hidden_size), dtype=torch.float)
            for _ in range(self.n_layers)]
        self.c_ = [torch.zeros(
            (self.batch_size, self.hidden_size),
            dtype=torch.float) for _ in range(self.n_layers)]

    def forward(self, inputs):

        # h_list = []
        # c_list = []
        # for i, lstm_cell in enumerate(self.lstm_cells):
        #     h_, c_ = lstm_cell(inputs, (self.h_[i], self.c_[i]))
        #     h_list.append(h_.detach())
        #     c_list.append(c_.detach())
        #     inputs = h_

        # self.h_ = torch.stack(h_list)
        # self.c_ = torch.stack(c_list)
        # output = self.linear(self.h_[-1])

        for i, lstm_cell in enumerate(self.lstm_cells):
            h_, c_ = lstm_cell(
                inputs, (self.h_[i], self.c_[i]))
            # have to detach because it tries to pass grad all the way back
            self.h_[i], self.c_[i] = tensor(h_.detach()), tensor(c_.detach())
            inputs = self.h_[i]
        output = self.linear(self.h_[-1])
        return output


if __name__ == "__main__":
    m = MetaLSTM(input_size=2, hidden_size=32,
                 output_size=5, n_layers=2, batch_size=1)
    #inputs = torch.tensor([1, 2], dtype=torch.float).reshape(1, 2)
    # m.forward(inputs)
    # h_ = torch.zeros((1, 32))
    # c_ = torch.zeros((1, 32))
    # lstm = nn.LSTMCell(input_size=2, hidden_size=32)
    # lstm(inputs, (h_, c_))
