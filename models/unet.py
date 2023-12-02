import torch
from torch import nn


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,padding=1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.pooling = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.BatchNorm1d(out_channels)

        self.act = nn.ReLU()
    
    def forward(self, x):
        x_in = self.act(self.norm1(self.conv1(x)))
        x_in = self.act(self.norm2(self.conv2(x_in)))
        return self.pooling(x_in), x_in

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, kernel_size=3, padding=1):
        super().__init__()

        self.ch_reduce = nn.Conv1d(in_channels + skip_channels, in_channels, kernel_size=1)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm1d(out_channels)
        self.upsample = nn.ConvTranspose1d(out_channels, out_channels, kernel_size=2, stride=2)

        self.act = nn.ReLU()

    def forward(self, x):
        x_in = self.act(self.ch_reduce(x))
        x_in = self.act(self.norm(self.conv(x_in)))
        return self.upsample(x_in)

        

class UNet1d(nn.Module):
    def __init__(self, in_channels, start_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        channels = [in_channels, start_channels, 2 * start_channels, 4 * start_channels, 8 * start_channels]
        self.encoder = nn.ModuleList([EncoderBlock(
                                            in_channels=channels[i],
                                            out_channels=channels[i + 1],
                                            kernel_size=kernel_size,
                                            padding=padding)
                                            for i in range(len(channels) - 1)])

        self.skips = list()
        mid_channels = channels[-1]
        after_mid_channels = 2 * mid_channels
        self.mid_layer = nn.Sequential(*[
            nn.Conv1d(mid_channels, after_mid_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(after_mid_channels),
            nn.ReLU(),
            nn.Conv1d(after_mid_channels, after_mid_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(after_mid_channels),
            nn.ReLU(),
            nn.ConvTranspose1d(after_mid_channels, mid_channels, kernel_size=2, stride=2)
        ])

        self.decoder = nn.ModuleList([DecoderBlock(
                                            in_channels=channels[i],
                                            out_channels=channels[i - 1],
                                            skip_channels=channels[i],
                                            kernel_size=kernel_size,
                                            padding=padding)
                                            for i in range(len(channels) - 1, 1, -1)])

        self.output_layer = nn.Sequential(*[
            nn.Conv1d(2 * start_channels, start_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(start_channels),
            nn.ReLU(),
            nn.Conv1d(start_channels, out_channels, kernel_size=1),
        ])
    
    def forward(self, x):
        for block in self.encoder:
            x, x_skip = block(x)
            self.skips.append(x_skip)
        
        x = self.mid_layer(x)
        x = torch.cat([x, self.skips[-1]], 1)
        self.skips[-1] = self.skips[-1].to("cpu")
        for i, block in enumerate(self.decoder):
            x = block(x)
            x = torch.cat([x, self.skips[- i - 2]], 1)
            self.skips[- i - 2] = self.skips[- i - 2].to("cpu")
        
        return self.output_layer(x)