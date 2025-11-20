import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(
        self,
        input_shape,  # L
        output_shape,  # L
        hidden_dim=2048,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            *[
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers - 1)
            ],
            nn.Linear(hidden_dim, output_shape),
        )

    def forward(self, x):
        return self.ffn(x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))


if __name__ == "__main__":
    input_shape = 2001
    output_shape = 3
    model = Network(input_shape, output_shape)
    x = torch.randn(16, input_shape)
    y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
