import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# LSTM 모델
class PaddingLstmodel(nn.Module):
    def __init__(self, input_size, hidden_size, layer_size, output_size, bidirectional=True, dropout=0.25):
        super(PaddingLstmodel, self).__init__()

        # 설정 값
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.output_size = output_size
        self.bidirectional = bidirectional

        # LSTM모델 선언
        self.lstm = nn.LSTM(input_size, hidden_size, layer_size, batch_first=True, bidirectional=bidirectional, dropout=dropout)

        # 양방향 체크
        if bidirectional:
            # output linear 모델 선언 (양방향 o)
            self.layer = nn.Linear(hidden_size*2, output_size)

            self.linear_layer = nn.Sequential(
                nn.Linear(hidden_size*2, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU(),
                # nn.Dropout(0.2),
                nn.Linear(hidden_size//2, output_size),
            )
        else:
            # output linear 모델 선언 (양방향 x)
            self.layer = nn.Linear(hidden_size, output_size)
            self.linear_layer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU(),
                # nn.Dropout(0.2),
                nn.Linear(hidden_size//2, hidden_size//4),
                nn.ReLU(),
                # nn.Dropout(0.2),
                nn.Linear(hidden_size//4, output_size)
            )

    def forward(self, padded_tensor, lengths, prints=False):
        if prints:
            print('images shape:', padded_tensor.shape)

        # 초기 세팅
        if self.bidirectional:
            # Hidden state:
            hidden_state = torch.zeros(self.layer_size*2, padded_tensor.size(0), self.hidden_size)
            # Cell state:
            cell_state = torch.zeros(self.layer_size*2, padded_tensor.size(0), self.hidden_size)
        else:
            # Hidden state:
            hidden_state = torch.zeros(self.layer_size, padded_tensor.size(0), self.hidden_size)
            # Cell state:
            cell_state = torch.zeros(self.layer_size, padded_tensor.size(0), self.hidden_size)

        if prints:
            print(
                'hidden_state t0 shape:', hidden_state.shape, '\n' +
                'cell_state t0 shape:', cell_state.shape)

        # padding data packing
        packed_output = pack_padded_sequence(padded_tensor, lengths, batch_first=True, enforce_sorted=False)

        # lstm layer output
        output, (last_hidden_state, last_cell_state) = self.lstm(packed_output, (hidden_state, cell_state))

        if prints:
            print(
                'LSTM: output shape:', output[0].shape, '\n' +
                'LSTM: last_hidden_state shape:', last_hidden_state.shape, '\n' +
                'LSTM: last_cell_state shape:', last_cell_state.shape)

        # get hidden state of final non-padded sequence element:
        padded_output, lengths = pad_packed_sequence(output, batch_first=True)
        h_n = []
        for seq, length in zip(padded_output, lengths):
            h_n.append(seq[length - 1, :])

        lstm_out = torch.stack(h_n)
        if prints:
            print('output reshape:', lstm_out.shape)

        # output linear layers
        out = self.layer(lstm_out)
        # out = self.linear_layer(lstm_out)
        if prints:
            print('FNN: Final output shape:', out.shape)

        return out
