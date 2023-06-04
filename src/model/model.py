import torch
import torch.nn as nn
import numpy as np
import copy

from attention import JointCrossAttentionBlock


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
        out = self.conv_end(x).permute(0,2,1)
        # out_mean = torch.mean(out, dim = 2)
        return out

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
        combined = torch.cat((seq_emb.squeeze(0), esm_emb.squeeze(0)), dim = 1)
        out = self.header(combined)
        return out
    

class TFBindingCrossAttentionModel(nn.Module):

    def __init__(self, args):
        super(TFBindingCrossAttentionModel, self).__init__()
        self.seq_encoder = SeqEncoder(5)
        self.esm_encoder = ESMEncoder(2560)

        # joint cross attn
        joint_cross_attn_depth = args.joint_cross_attn_depth
        self.joint_cross_attns = nn.ModuleList([])
        for _ in range(joint_cross_attn_depth):
            attn = JointCrossAttentionBlock(
                dim = 512,
                context_dim = 512,
                dropout = 0.5
            )
            self.joint_cross_attns.append(attn)

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
        # joint cross attention
        for cross_attn in self.joint_cross_attns:
            seq_emb, esm_emb = cross_attn(
                seq_emb,
                context = esm_emb,
                context_mask = None
            )
        seq_emb = torch.mean(seq_emb, dim = 1)
        esm_emb = torch.mean(esm_emb, dim = 1)
        combined = torch.cat((seq_emb, esm_emb), dim = 1)
        out = self.header(combined)
        return out

if __name__ == '__main__':
    esm = torch.randn(32, 2560, 60)
    dna = torch.randn(32, 5, 60)
    model = TFBindingModel()
    out = model(dna, esm)
    print(out.shape)
