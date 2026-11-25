"""Configurable fully connected neural network."""

import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 16,
        num_hidden_layers: int = 2,
        output_size: int = 2,
        initialization: str = "he",
    ):
        super().__init__()

        if input_size < 1 or hidden_size < 1 or output_size < 1:
            raise ValueError("Layer sizes must be positive integers")
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be at least 1")

        layers = []
        in_features = input_size
        for _ in range(num_hidden_layers):
            layers.extend((nn.Linear(in_features, hidden_size), nn.ReLU()))
            in_features = hidden_size

        layers.append(nn.Linear(in_features, output_size))
        self.network = nn.Sequential(*layers)
        self.initialize_weights(initialization)

    def initialize_weights(self, initialization: str) -> None:
        if initialization not in {"xavier", "he", "default"}:
            raise ValueError(
                "initialization must be 'xavier', 'he', or 'default'"
            )

        for module in self.modules():
            if not isinstance(module, nn.Linear):
                continue
            if initialization == "xavier":
                nn.init.xavier_uniform_(module.weight)
            elif initialization == "he":
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            nn.init.zeros_(module.bias)

    def forward(self, inputs):
        return self.network(inputs)
