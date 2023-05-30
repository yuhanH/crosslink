import torch
import torch.nn as nn
import numpy as np
import copy

class ConvBlock(nn.Module):
    def __init__(self, size, stride = 2, hidden_in = 64, hidden = 64):
        super(ConvBlock, self).__init__()
        pad_len = int(size / 2)
        self.scale = nn.Sequential(
                        nn.Conv1d(hidden_in, hidden, size, stride, pad_len, dilation=2),
                        nn.BatchNorm1d(hidden),
                        nn.ReLU(),
                        )
        self.res = nn.Sequential(
                        nn.Conv1d(hidden, hidden, size, padding = pad_len),
                        nn.BatchNorm1d(hidden),
                        nn.ReLU(),
                        nn.Conv1d(hidden, hidden, size, padding = pad_len),
                        nn.BatchNorm1d(hidden),
                        )
        self.relu = nn.ReLU()

    def forward(self, x):
        scaled = self.scale(x)
        identity = scaled
        res_out = self.res(scaled)
        out = self.relu(res_out + identity)
        return out

class ESMEncoder(nn.Module):
    def __init__(self, in_channel, output_size = 512, filter_size = 3, num_blocks = 5):
        super(ESMEncoder, self).__init__()
        self.filter_size = filter_size
        self.conv_start = nn.Sequential(
                                    nn.Conv1d(in_channel, 1024, 1, 1, 1),
                                    nn.BatchNorm1d(1024),
                                    nn.ReLU(),
                                    )
        hidden_ins = [1024, 512, 512, 512]
        hiddens =          [512, 512, 512, 512]
        self.res_blocks = self.get_res_blocks(num_blocks, hidden_ins, hiddens)
        self.conv_end = nn.Conv1d(512, output_size, 1)

    def forward(self, x):
        x = self.conv_start(x)
        x = self.res_blocks(x)
        out = self.conv_end(x)
        out_mean = torch.mean(out, dim = 2)
        return out_mean

    def get_res_blocks(self, n, his, hs):
        blocks = []
        for i, h, hi in zip(range(n), hs, his):
            blocks.append(ConvBlock(self.filter_size, hidden_in = hi, hidden = h))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks

class SeqEncoder(ESMEncoder):

    def __init__(self, in_channel, output_size = 512, filter_size = 3, num_blocks = 5):
        super(ESMEncoder, self).__init__()
        self.filter_size = filter_size
        self.conv_start = nn.Sequential(
                                    nn.Conv1d(in_channel, 16, 1, 1, 1),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU(),
                                    )
        hidden_ins = [16, 32, 64, 128]
        hiddens =        [32, 64, 128, 256]
        self.res_blocks = self.get_res_blocks(num_blocks, hidden_ins, hiddens)
        self.conv_end = nn.Conv1d(256, output_size, 1)

class TFBindingModel(nn.Module):

    def __init__(self):
        super(TFBindingModel, self).__init__()
        self.seq_encoder = SeqEncoder(5)
        self.esm_encoder = ESMEncoder(2560)
        self.header = nn.Sequential(
                nn.Linear(512 + 512, 512),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 1))

    def forward(self, seq, esm_emb):
        seq_emb = self.seq_encoder(seq)
        esm_emb = self.esm_encoder(esm_emb)
        combined = torch.cat((seq_emb, esm_emb), dim = 1)
        out = self.header(combined)
        return out

if __name__ == '__main__':
    esm = torch.randn(32, 2560, 60)
    dna = torch.randn(32, 5, 60)
    model = TFBindingModel()
    out = model(dna, esm)
    print(out.shape)
