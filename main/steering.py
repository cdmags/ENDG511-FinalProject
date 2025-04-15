import torch
import torch.nn as nn

class GatedSteeringNet(nn.Module):
    def __init__(self, input_dim, smooth_weight=0.7):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

        self.head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Tanh()
        )

        self.model = self.head
        self.register_buffer("smoothed_output", torch.zeros(1, 1))
        self.smooth_weight = smooth_weight

    def forward(self, x):
        gated = self.gate(x) * x
        raw = self.head(gated)
        if self.training:
            return raw
        
        if raw.shape != self.smoothed_output.shape:
            self.smoothed_output = raw.clone().detach()
        smoothed = self.smooth_weight * raw + (1 - self.smooth_weight) * self.smoothed_output
        self.smoothed_output = smoothed.detach()
        return smoothed
