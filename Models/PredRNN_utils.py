import torch
import torch.nn as nn


class SpatioTemporalLSTMCell(nn.Module):
    """PredRNN (v1) spatiotemporal LSTM cell.

    Couples the standard temporal memory ``c`` with the zigzag spatiotemporal
    memory ``m`` (Wang et al., NeurIPS 2017). LayerNorm is sized for a
    rectangular ``(height, width)`` latent grid (this repo runs the 32x64 USA
    crop; the upstream reference assumes square inputs).

    Args:
        in_channel: Channels of the per-step input frame.
        num_hidden: Hidden channels of this layer.
        height: Latent-grid height (after patch embedding).
        width: Latent-grid width (after patch embedding).
        filter_size: Square conv kernel size.
        stride: Conv stride.
        layer_norm: If ``True``, wrap each gate conv in ``nn.LayerNorm``.
    """

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super().__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(
                    in_channel,
                    num_hidden * 7,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=self.padding,
                    bias=False,
                ),
                nn.LayerNorm([num_hidden * 7, height, width]),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(
                    num_hidden,
                    num_hidden * 4,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=self.padding,
                    bias=False,
                ),
                nn.LayerNorm([num_hidden * 4, height, width]),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(
                    num_hidden,
                    num_hidden * 3,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=self.padding,
                    bias=False,
                ),
                nn.LayerNorm([num_hidden * 3, height, width]),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(
                    num_hidden * 2,
                    num_hidden,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=self.padding,
                    bias=False,
                ),
                nn.LayerNorm([num_hidden, height, width]),
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(
                    in_channel,
                    num_hidden * 7,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=self.padding,
                    bias=False,
                ),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(
                    num_hidden,
                    num_hidden * 4,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=self.padding,
                    bias=False,
                ),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(
                    num_hidden,
                    num_hidden * 3,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=self.padding,
                    bias=False,
                ),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(
                    num_hidden * 2,
                    num_hidden,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=self.padding,
                    bias=False,
                ),
            )
        self.conv_last = nn.Conv2d(
            num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False
        )

    def forward(self, x_t, h_t, c_t, m_t):
        """Advance the cell by one timestep.

        Args:
            x_t: input frame ``(B, in_channel, H, W)``.
            h_t: previous hidden state ``(B, num_hidden, H, W)``.
            c_t: previous temporal memory ``(B, num_hidden, H, W)``.
            m_t: previous spatiotemporal memory ``(B, num_hidden, H, W)``.

        Returns:
            Tuple ``(h_new, c_new, m_new)``, each ``torch.Tensor`` of shape
            ``(B, num_hidden, H, W)``.
        """
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(
            x_concat, self.num_hidden, dim=1
        )
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new


class SpatioTemporalLSTMCellv2(SpatioTemporalLSTMCell):
    """PredRNN-V2 spatiotemporal LSTM cell with memory-decoupling outputs.

    Inherits the constructor (and therefore the exact parameter layout) from
    :class:`SpatioTemporalLSTMCell`; the recurrence is identical except that
    ``forward`` additionally returns the per-step memory increments
    ``delta_c`` / ``delta_m`` consumed by PredRNN-V2's decoupling loss.
    """

    def forward(self, x_t, h_t, c_t, m_t):
        """Advance the cell by one timestep and expose memory increments.

        Args:
            x_t: input frame ``(B, in_channel, H, W)``.
            h_t: previous hidden state ``(B, num_hidden, H, W)``.
            c_t: previous temporal memory ``(B, num_hidden, H, W)``.
            m_t: previous spatiotemporal memory ``(B, num_hidden, H, W)``.

        Returns:
            Tuple ``(h_new, c_new, m_new, delta_c, delta_m)``, each
            ``torch.Tensor`` of shape ``(B, num_hidden, H, W)``; ``delta_c`` and
            ``delta_m`` are the memory increments consumed by the V2 decoupling
            loss.
        """
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(
            x_concat, self.num_hidden, dim=1
        )
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        delta_c = i_t * g_t
        c_new = f_t * c_t + delta_c

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        delta_m = i_t_prime * g_t_prime
        m_new = f_t_prime * m_t + delta_m

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new, delta_c, delta_m
