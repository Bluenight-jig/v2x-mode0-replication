import torch, torch.nn as nn

class RSUCritic(nn.Module):
    def __init__(self, global_state_dim:int, hidden:int=256):
        # hidden: 256 for N≤20; 512 for N>20
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(global_state_dim,hidden),
            nn.LayerNorm(hidden),
            # LayerNorm: stabilises training on large/structured inputs.
            # Normalises per-sample (not per-batch) — critical for RL
            # where batch statistics fluctuate between rollouts.
            nn.ReLU(),
            nn.Linear(hidden,hidden), nn.ReLU(),
            nn.Linear(hidden,1))

    def forward(self,state:torch.Tensor)->torch.Tensor:
        return self.net(state)
