import torch.nn as nn


class MetaLSTM(nn.Module):

    # TODO: Make this residual LSTM as in Jibo et al.
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            self.input_size, self.hidden_size, batch_first=True
        )

        self.linear = nn.Linear(hidden_size, output_size)

    # TODO: Add initialization of cell/hidden states (call this when we clear over epochs)

    def forward(self, inputs):
        # break up input into: inputs and hidden/cell states
        inputs, (h_, c_) = inputs
        output, (h_, c_) = self.lstm(inputs, (h_, c_))
        output = self.linear(output)
        return output, (h_, c_)


if __name__ == "__main__":
    pass
