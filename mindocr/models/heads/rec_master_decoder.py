from typing import Optional, Tuple

import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from ..utils.attention_cells import MultiHeadAttention, PositionalEncoding, PositionwiseFeedForward

__all__ = ["MasterDecoder"]


class MasterDecoder(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        batch_max_length: int = 25,
        multi_heads_count: int = 8,
        stacks: int = 3,
        dropout: float = 0.0,
        feed_forward_size: int = 2048,
        padding_symbol: int = 2,
        share_parameter: bool = False,
    ) -> None:
        super().__init__()
        self.share_parameter = share_parameter
        self.batch_max_length = batch_max_length

        self.attention = nn.CellList(
            [
                MultiHeadAttention(multi_heads_count, in_channels, dropout)
                for _ in range(1 if share_parameter else stacks)
            ]
        )
        self.source_attention = nn.CellList(
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
        self.stacks = stacks
        self.dropout = nn.Dropout(keep_prob=1 - dropout)
        self.layer_norm = nn.LayerNorm([in_channels], epsilon=1e-6)
        self.embedding = nn.Embedding(out_channels, in_channels)
        self.sqrt_model_size = np.sqrt(in_channels)
        self.padding_symbol = padding_symbol
        self.generator = nn.Dense(in_channels, out_channels)

        # mask related
        self.tril = nn.Tril()

    def _generate_target_mask(self, targets: Tensor) -> Tensor:
        target_pad_mask = targets != self.padding_symbol
        target_pad_mask = target_pad_mask[:, None, :, None]
        target_pad_mask = ops.cast(target_pad_mask, ms.int32)
        target_length = targets.shape[1]
        target_sub_mask = self.tril(ops.ones((target_length, target_length), ms.int32))
        target_mask = ops.bitwise_and(target_pad_mask, target_sub_mask)
        return target_mask

    def _decode(
        self,
        inputs: Tensor,
        targets: Tensor,
        source_mask: Optional[Tensor] = None,
        target_mask: Optional[Tensor] = None,
    ) -> Tensor:
        targets = self.embedding(targets) * self.sqrt_model_size
        targets = self.position(targets)
        output = targets
        for i in range(self.stacks):
            if self.share_parameter:
                actual_i = i
            else:
                actual_i = 0

            normed_output = self.layer_norm(output)
            output = output + self.dropout(
                self.attention[actual_i](
                    normed_output, normed_output, normed_output, target_mask
                )
            )
            normed_output = self.layer_norm(output)
            output = output + self.dropout(
                self.source_attention[actual_i](
                    normed_output, inputs, inputs, source_mask
                )
            )
            normed_output = self.layer_norm(output)
            output = output + self.dropout(
                self.position_feed_forward[actual_i](normed_output)
            )
        output = self.layer_norm(output)
        output = self.generator(output)
        return output

    def construct(
        self, inputs: Tensor, targets: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        N = inputs.shape[0]
        num_steps = self.batch_max_length + 1  # for <STOP> symbol

        if targets is not None:
            # training branch
            targets = targets[:, :-1]
            target_mask = self._generate_target_mask(targets)
            logits = self._decode(inputs, targets, target_mask=target_mask)
            return logits
        else:
            # inference branch
            targets = ops.zeros((N, 1), ms.int32)
            probs = ops.ones((N, num_steps + 1), inputs.dtype)

            for i in range(num_steps):
                target_mask = self._generate_target_mask(targets)
                out = self._decode(inputs, targets, target_mask=target_mask)
                prob = ops.softmax(out, axis=-1)
                next_input, next_input_prob = ops.max(prob, axis=-1)
                targets = ops.concat([targets, next_input[:, i : i + 1]], axis=1)
                probs[:, i + 1] = next_input_prob[:, i]

            # remove <GO> symbol
            targets = targets[:, 1:]
            probs = probs[:, 1:]
        return targets, probs
