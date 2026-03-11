"""
Custom policy networks for the trading environment.

Provides a 1-D CNN extractor that treats the OHLCV window (first window_size*5
elements of the observation) as a sequence and extracts temporal features.
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CNN1DExtractor(BaseFeaturesExtractor):
    """
    1-D CNN feature extractor for time-series OHLCV observations.

    Splits the flat observation into:
        - ohlcv_part: first (window_size * n_features) elements — treated as
          a (batch, n_features, window_size) tensor and passed through 1-D CNN.
        - state_part: remaining elements (position state) — passed through MLP.

    The outputs are concatenated to produce the final feature vector.

    Args:
        observation_space: Gym observation space (must be Box 1-D).
        n_features:        Number of features per bar (default 5 for OHLCV).
        window_size:       Number of bars in the OHLCV window.
        cnn_channels:      List of output channels for each Conv1d layer.
        kernel_size:       Kernel size used for all Conv1d layers.
        state_hidden:      Hidden units in the MLP that processes position state.
        features_dim:      Size of the final merged feature vector.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        n_features: int = 5,
        window_size: int = 64,
        cnn_channels: tuple = (32, 64, 64),
        kernel_size: int = 3,
        state_hidden: int = 32,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim=features_dim)

        self.n_features = n_features
        self.window_size = window_size
        ohlcv_size = window_size * n_features
        state_size = observation_space.shape[0] - ohlcv_size

        # --- 1-D CNN for the OHLCV window ---
        cnn_layers = []
        in_ch = n_features
        for out_ch in cnn_channels:
            cnn_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
            ]
            in_ch = out_ch
        cnn_layers.append(nn.AdaptiveAvgPool1d(1))  # global average pool → (batch, C, 1)
        self.cnn = nn.Sequential(*cnn_layers)
        cnn_out_dim = cnn_channels[-1]

        # --- MLP for position state ---
        self.state_mlp = nn.Sequential(
            nn.Linear(state_size, state_hidden),
            nn.ReLU(),
        )

        # --- Merge projection ---
        merged_dim = cnn_out_dim + state_hidden
        self.merge = nn.Sequential(
            nn.Linear(merged_dim, features_dim),
            nn.ReLU(),
        )

        self._ohlcv_size = ohlcv_size
        self._state_size = state_size

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Split observation
        ohlcv_flat = observations[:, : self._ohlcv_size]
        state = observations[:, self._ohlcv_size :]

        # Reshape to (batch, channels, time) for Conv1d
        batch = ohlcv_flat.shape[0]
        ohlcv_seq = ohlcv_flat.view(batch, self.window_size, self.n_features)
        ohlcv_seq = ohlcv_seq.permute(0, 2, 1)  # (batch, features, time)

        cnn_out = self.cnn(ohlcv_seq).squeeze(-1)  # (batch, cnn_channels[-1])
        state_out = self.state_mlp(state)           # (batch, state_hidden)

        merged = torch.cat([cnn_out, state_out], dim=1)
        return self.merge(merged)
