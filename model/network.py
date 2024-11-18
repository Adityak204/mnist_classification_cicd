import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            1, 6, kernel_size=3, padding=1
        )  # output size : 28x28x6 (formula: (28 + 2*1 - 3)/1 + 1 = 28)
        self.conv2 = nn.Conv2d(
            6, 6, kernel_size=3, padding=1
        )  # output size : 28x28x6 (formula: (28 + 2*1 - 3)/1 + 1 = 28)
        self.fc1 = nn.Linear(6 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(
            self.relu(self.conv1(x))
        )  # output size : 14x14x6 (formula: (28 - 2)/2 + 1 = 14)
        x = self.pool(
            self.relu(self.conv2(x))
        )  # output size : 7x7x6 (formula: (14 - 2)/2 + 1 = 7)
        x = x.view(-1, 6 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# if __name__ == "__main__":
#     model = SimpleCNN()
#     print(model)
#     print(sum(p.numel() for p in model.parameters()))
