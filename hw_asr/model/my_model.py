from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel
import torch


class CustomReLu(nn.Module):
    def __init__(self, min_, max_):
        super().__init__()
        self.min_ = min_
        self.max_ = max_

    def forward(self, x):
        return torch.clamp(x, self.min_, self.max_)


class CustomRecurrentLayer(nn.Module):
    def __init__(self, features, gru_hidden_size):
        super().__init__()
        self.norm = nn.BatchNorm1d(gru_hidden_size * 2)
        self.gru = nn.GRU(features, gru_hidden_size, batch_first=True, dropout=0.2, bidirectional=True)

    def forward(self, x):
        outp, _ = self.gru(x)
        outp = self.norm(outp.transpose(1, 2)).transpose(1, 2)
        return outp


class MyModel(BaseModel):
    def __init__(self, n_feats, n_class, recurrent_layers, gru_hidden_size, **batch):
        super().__init__(n_feats, n_class, **batch)
        self.convolutions = Sequential(
            nn.Conv2d(1, 32, (41, 11), (2, 2)),
            CustomReLu(0, 20),
            nn.Conv2d(32, 32, (21, 11), (2, 1)),
            CustomReLu(0, 20), 
        )
        self.n_feats = n_feats
        self.bn_before_recurrent = nn.BatchNorm1d(self.convolved_features)
        
        assert recurrent_layers >= 2, f"Make recurrent_layers >= 2. Got {recurrent_layers}"
        rucurrent_list = [CustomRecurrentLayer(self.convolved_features, gru_hidden_size)] 
        for i in range(recurrent_layers - 1):
            rucurrent_list.append(CustomRecurrentLayer(gru_hidden_size * 2, gru_hidden_size))
        self.recurrent = Sequential(*rucurrent_list)
        self.fc_head = nn.Linear(in_features=gru_hidden_size * 2, out_features=n_class)

    @property
    def convolved_features(self):
        f = (self.n_feats - 41) // 2 + 1
        f = (f - 21) // 2 + 1
        return 32 * f

    def forward(self, spectrogram, **batch):
        spectrogram = spectrogram.unsqueeze(1)  # batch_size, 1, freq, time
        # print('spectrogram shape:', spectrogram.shape)
        convoluted = self.convolutions(spectrogram)  # batch_size, channels, freq, time
        # print('convoluted shape:', convoluted.shape)
        flattened = convoluted.transpose(1, 3).flatten(2)  # batch_size, time, convolved_features
        # print("flattened shape:", flattened.shape)
        flattened = self.bn_before_recurrent(flattened.transpose(1, 2)).transpose(1, 2)
        outp = self.recurrent(flattened)
        # print("after gru", outp.shape)
        return {"logits": self.fc_head(outp)}

    def transform_input_lengths(self, input_lengths):
        input_lengths = (input_lengths - 11) // 2 + 1
        input_lengths = (input_lengths - 11) + 1
        return input_lengths
