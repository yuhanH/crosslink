from torch import nn
from bidirectional_cross_attention import BidirectionalCrossAttention

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )


class JointCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        context_dim = None,
        ff_mult = 4,
        dropout = 0.,
        **kwargs
    ):
        super().__init__()
        context_dim = default(context_dim, dim)

        self.attn = BidirectionalCrossAttention(dim = dim, context_dim = context_dim, dropout = dropout, prenorm = True, **kwargs)
        self.ff = FeedForward(dim, mult = ff_mult, dropout = dropout)
        self.context_ff = FeedForward(context_dim, mult = ff_mult, dropout = dropout)

    def forward(
        self,
        x,
        context,
        mask = None,
        context_mask = None
    ):
        attn_out, context_attn_out = self.attn(x, context, mask = mask, context_mask = context_mask)

        x = x + attn_out
        context = context + context_attn_out

        x = self.ff(x) + x
        context = self.context_ff(context) + context

        return x, context
