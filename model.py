import torch
import torch.nn as nn

# We'll use `nn.Sequential` to chain the layers and activations functions into a single network architecture.
class Cifar10CnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # input: 3 x 32 x 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            # output : 32 x 32 x 32
            nn.ReLU(),
            # output : 32 x 32 x 32
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # output : 64 x 32 x 32
            nn.ReLU(),
            # output : 64 x 32 x 32
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # output: 128 x 16 x 16
            nn.ReLU(),
            # output: 128 x 16 x 16
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            # output: 128 x 16 x 16
            nn.ReLU(),
            # output: 128 x 16 x 16
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # output: 256 x 8 x 8
            nn.ReLU(),
            # output: 256 x 8 x 8
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # output: 256 x 8 x 8
            nn.ReLU(),
            # output: 256 x 8 x 8
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

            nn.Flatten(),
            # This will converts into the vector form
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))
        
    def forward(self, xb):
        return self.network(xb)
    




