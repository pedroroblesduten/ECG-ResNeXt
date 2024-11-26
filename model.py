import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class Permute(nn.Module):
    """Permute tensor dimensions."""

    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims).contiguous()


class GRN_point(nn.Module):
    """GRN (Global Response Normalization) layer for pointwise operations."""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        # x: (B, L, C)
        Gx = torch.norm(
            x, p=2, dim=1, keepdim=True
        )  # Compute norm over the sequence length
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)  # Normalize
        return self.gamma * (x * Nx) + self.beta + x


class ResidualBlock(nn.Module):
    """Residual Block with depthwise convolution and pointwise operations."""

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout_rate=0.2,
        original_len=2560,
        ith_block=1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # First branch with depthwise convolution and pointwise operations
        self.first_branch = nn.Sequential(
            # Depthwise convolution
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=10,
                stride=4,
                padding=4,
                groups=in_channels,
                bias=False,
            ),
            # Pointwise convolution to change channel dimensions
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            Permute(0, 2, 1),  # (B, C, L) -> (B, L, C)
            nn.LayerNorm(out_channels, eps=1e-6),
            nn.Linear(out_channels, 4 * out_channels),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            GRN_point(4 * out_channels),
            nn.Linear(4 * out_channels, out_channels),
            Permute(0, 2, 1),  # (B, L, C) -> (B, C, L)
        )

        # Skip connection with downsampling to match dimensions
        self.skip_connection = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
        )

        # Second branch with pointwise operations
        self.second_branch = nn.Sequential(
            Permute(0, 2, 1),
            nn.LayerNorm(out_channels, eps=1e-6),
            nn.Linear(out_channels, 4 * out_channels),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            GRN_point(4 * out_channels),
            nn.Linear(4 * out_channels, out_channels),
            Permute(0, 2, 1),
        )

    def forward(self, x):
        # First branch
        out = self.first_branch(x)

        # Skip connection
        residual = self.skip_connection(x)

        # Add skip connection
        out = out + residual

        # Second branch
        residual = out
        out = self.second_branch(out)

        # Add skip connection
        out = out + residual

        return out


class StackedResidual(nn.Module):
    """Stack multiple Residual Blocks."""

    def __init__(self, channels, num_blocks, dropout_rate=0.2, original_len=2560):
        super().__init__()

        # Create a list of residual blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            in_channels = channels[i]
            out_channels = in_channels + 64  # Adjust as necessary
            self.blocks.append(
                ResidualBlock(
                    in_channels,
                    out_channels,
                    dropout_rate=dropout_rate,
                    original_len=original_len,
                    ith_block=i + 1,
                )
            )

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            x = block(x)
        return x


class ECG_ResNeXt(nn.Module):
    """ECG ResNeXt Model."""

    def __init__(
        self,
        n_classes,
        num_blocks,
        channels,
        dropout_rate=0.2,
        original_len=2560,
        head_init_scale=1.0,
    ):
        super().__init__()

        # Initial convolutional layer with depthwise and pointwise operations
        self.input_layer = nn.Sequential(
            nn.Conv1d(
                12, 12, kernel_size=17, stride=4, padding=8, groups=12, bias=False
            ),
            nn.Conv1d(12, 64, kernel_size=1, stride=1, bias=False),
            Permute(0, 2, 1),
            nn.LayerNorm(64, eps=1e-6),
            nn.Linear(64, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            GRN_point(256),
            nn.Linear(256, 64),
            Permute(0, 2, 1),
        )

        self.residual_blocks = StackedResidual(
            channels, num_blocks, dropout_rate=dropout_rate, original_len=original_len
        )
        self.flatten = nn.Flatten()
        downsample_factor = 4 ** (num_blocks + 1)  # Including initial downsampling
        flattened_size = 64 * (original_len // downsample_factor)
        self.head = nn.Linear(2560, n_classes)
        self.head_init_scale = head_init_scale

        # Apply weight initialization
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_blocks(x)
        x = self.flatten(x)
        logits = self.head(x)
        return logits

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, GRN_point):
            nn.init.constant_(m.gamma, 1.0)
            nn.init.constant_(m.beta, 0)
