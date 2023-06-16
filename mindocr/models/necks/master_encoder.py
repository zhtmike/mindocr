from typing import List

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from ..utils import MultiHeadAttention, PositionalEncoding, PositionwiseFeedForward


class MasterEncoder(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        with_encoder: bool = False,
        multi_heads_count: int = 8,
        stacks: int = 3,
        dropout: float = 0.2,
        feed_forward_size: int = 2048,
        share_parameter: bool = False,
    ) -> None:
        super(MasterEncoder, self).__init__()
        self.out_channels = in_channels
        self.share_parameter = share_parameter
        self.attention = nn.CellList(
            [
                MultiHeadAttention(multi_heads_count, in_channels, dropout)
                for _ in range(1 if share_parameter else stacks)
            ]
        )
        self.position_feed_forward = nn.CellList(
            [
                PositionwiseFeedForward(in_channels, feed_forward_size, dropout)
                for _ in range(1 if share_parameter else stacks)
            ]
        )
        self.position = PositionalEncoding(in_channels, dropout)
        self.layer_norm = nn.LayerNorm([in_channels], epsilon=1e-6)
        self.stacks = stacks
        self.dropout = nn.Dropout(keep_prob=1 - dropout)
        self.with_encoder = with_encoder

    def construct(self, features: List[Tensor]) -> Tensor:
        # convert input from N,C,H,W to N,H*W,C
        x = features[0]
        N, C, _, _ = x.shape
        x = x.reshape(N, C, -1)
        x = ops.transpose(x, (0, 2, 1))

        output = self.position(x)
        if self.with_encoder:
            for i in range(self.stacks):
                if self.share_parameter:
                    actual_i = 0
                else:
                    actual_i = i
                normed_output = self.layer_norm(output)
                output = output + self.dropout(
                    self.attention[actual_i](
                        normed_output, normed_output, normed_output
                    )
                )
                normed_output = self.layer_norm(output)
                output = output + self.dropout(
                    self.position_feed_forward[actual_i](normed_output)
                )
            output = self.layer_norm(output)
        return output
