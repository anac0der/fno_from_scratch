import torch
from torch import nn
from .spectral_conv import SpectralConv1d

class FourierLayer1d(nn.Module):
    def __init__(self, in_ch, out_ch, n_modes):
        super().__init__()
        
        self.spectral_conv = SpectralConv1d(in_ch, out_ch, n_modes)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.act = nn.GELU()
    
    def forward(self, x):
        x_fc = self.spectral_conv(x)
        x_skip = self.skip(x).to(x.device)
        return self.act(x_skip + x_fc)


class MyFNO1d(nn.Module):
    def __init__(self, n_modes, n_layers, in_channels=3, out_channels=1, hidden_channels=32, projection_channels=32):
        super().__init__()
        
        self.lifting = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        
        self.fno_blocks = nn.Sequential(*[
            FourierLayer1d(hidden_channels, hidden_channels, n_modes=n_modes) for _ in range(n_layers)
        ])

        self.projection = nn.Sequential(*[
            nn.Conv1d(hidden_channels, projection_channels, kernel_size=1),
            nn.Conv1d(projection_channels, out_channels, kernel_size=1),
        ])
    
    def forward(self, x):
        x_in = self.lifting(x)
        x_in = self.fno_blocks(x_in)
        return self.projection(x_in)