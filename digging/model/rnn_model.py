import torch
import torch.nn as nn
from typing import Optional, Tuple


class LSTMSeqModel(nn.Module):

    def __init__(
            self,
            input_size: int,
            hidden_dim: int,
            sequence_len: int,
            num_layers: int = 1,
    ) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim
        self._seq_len = sequence_len
        self._num_layers = num_layers

        self._rnn = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True)

    def _init_hidden_state(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = inputs.shape[1]
        batch_size = inputs.shape[0]
        zeros = torch.zeros(self._num_layers, batch_size, self._hidden_dim, dtype=inputs.dtype, device=inputs.device)
        return zeros, zeros

    def forward(self, inputs: torch.Tensor, prev_state: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        if len(inputs.shape) == 2:
            inputs = inputs.unsqueeze(1)
        if prev_state is None:
            prev_state = self._init_hidden_state(inputs)
        output = []
        x = inputs
        h = prev_state
        for _ in range(self._seq_len):
            x, h = self._rnn(x, h)
            output.append(x)
        output = torch.cat(output, dim=1)
        return output, h


class GRUSeqModel(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_dim: int,
        sequence_len: int,
        num_layers: int = 1,
    ):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._seq_len = sequence_len
        self._num_layers = num_layers

        self._rnn = nn.GRU(input_size, hidden_dim, num_layers)

    def _init_hidden_state(self, inputs: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size = inputs.shape[:2]
        zeros = torch.zeros(self._num_layers, batch_size, self._hidden_dim, dtype=inputs.dtype, device=inputs.device)
        return zeros

    def forward(self,
                inputs: torch.Tensor,
                prev_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if prev_state is None:
            prev_state = self._init_hidden_state(inputs)
        output = []
        x = inputs
        h = prev_state
        for _ in range(self._seq_len):
            x, h = self._rnn(x, h)
            output.append(x)
        output = torch.cat(output, dim=1)
        return output, h
