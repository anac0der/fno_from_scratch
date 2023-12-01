import os
from pathlib import Path
from scipy.io import loadmat
import torch
import numpy as np
from .tensor_dataset import TensorDataset


def load_burgers_1dtime(
        data_path, n_train, n_test, batch_size=16, batch_size_test=16,
        temporal_length=101, spatial_length=128, temporal_subsample=1, 
        spatial_subsample=1, pad=0):
    """
    Load burgers.mat data. Given the initial condition (t=0),
    predict timesteps 1 to temporal_length.
    """

    x_data = loadmat(os.path.join(data_path, "x_data.mat"))["input"]
    y_data = loadmat(os.path.join(data_path, "y_data.mat"))["output"]

    x_data = torch.from_numpy(x_data.astype(np.float32))
    x_data = x_data[:, :spatial_length:spatial_subsample]
    y_data = torch.from_numpy(y_data.astype(np.float32))
    y_data = y_data[:, :temporal_length:temporal_subsample, :spatial_length:spatial_subsample]

    x_train = x_data[:n_train]
    y_train = y_data[:n_train]
    x_test = x_data[n_train:n_train+n_test]
    y_test = y_data[n_train:n_train+n_test]

    domain_lengths = [spatial_length / 128, (temporal_length - 1) / 100]
    domain_starts = [0., 0.]

    spatial_length = spatial_length // spatial_subsample
    temporal_length = temporal_length // temporal_subsample

    if pad:
        x_train = torch.nn.ReplicationPad1d(pad)(x_train)
        x_test = torch.nn.ReplicationPad1d(pad)(x_test)
        spatial_length += 2 * pad
        temporal_length += 2 * pad
        incrs = [spatial_subsample / 128, temporal_subsample / 100]
        domain_lengths = [d + incr * pad for d, incr in zip(domain_lengths, incrs)]
        domain_starts = [-incr * pad for incr in incrs]

    # TODO: use include_endpoint arg here
    grid_x = torch.tensor(np.linspace(domain_starts[0], domain_lengths[0], spatial_length + 1)[:-1], dtype=torch.float)
    grid_t = torch.tensor(np.linspace(domain_starts[1], domain_lengths[1], temporal_length), dtype=torch.float)

    grid_x = grid_x.reshape(1, 1, spatial_length)
    grid_t = grid_t.reshape(1, temporal_length, 1)

    x_train = x_train.reshape(n_train, 1, spatial_length).repeat([1, temporal_length, 1])
    x_test = x_test.reshape(n_test, 1, spatial_length).repeat([1, temporal_length, 1])

    # TODO: add option to not have positional encoding
    x_train = torch.stack([x_train, 
                           grid_t.repeat([n_train, 1, spatial_length]),
                           grid_x.repeat([n_train, temporal_length, 1]) 
                           ], dim=3)
    x_test = torch.stack([x_test, 
                          grid_t.repeat([n_test, 1, spatial_length]),
                          grid_x.repeat([n_test, temporal_length, 1]) 
                          ], dim=3)

    x_train = x_train.permute(0, 3, 1, 2)
    x_test = x_test.permute(0, 3, 1, 2)
    y_train = y_train.unsqueeze(1)
    y_test = y_test.unsqueeze(1)

    train_db = TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_db, batch_size=batch_size, shuffle=False)

    test_db = TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_db, batch_size=batch_size_test, shuffle=False)

    output_encoder = None

    return train_loader, test_loader, output_encoder, x_train, x_test
