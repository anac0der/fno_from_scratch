import torch
from torch import nn

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_modes: int,  fft_norm="backward"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes_trunc = n_modes // 2 + 1
        self.fft_norm = fft_norm
        
        #Kaiming initialization
        self.std = (2 / (in_channels + out_channels))**0.5

        self.weights = nn.Parameter(self.std * torch.rand(in_channels, out_channels, self.n_modes_trunc, dtype=torch.cfloat))

    def forward(self, x):
        batch_size = x.shape[0]
        width = x.shape[-1]

        ft = torch.fft.rfft(x, dim=-1, norm=self.fft_norm)

        last_dim = x.size(-1) // 2  + 1
        output = torch.zeros((batch_size, self.out_channels, last_dim), dtype=torch.cfloat).to(x.device)

        output[:, :, :self.n_modes_trunc] = self.weights_mul(ft[:, :, :self.n_modes_trunc], self.weights)

        x_rec = torch.fft.irfft(output, n=width, norm=self.fft_norm)

        return x_rec

    def weights_mul(self, inp, weights):
        return torch.einsum("bix,iox->box", inp, weights)