import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.network import SimpleCNN
from datetime import datetime
import os


def train():
    # Force CPU usage
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Enhanced data augmentation pipeline
    transform = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=10,  # Random rotation up to 10 degrees
                translate=(0.1, 0.1),  # Random translation up to 10%
                scale=(0.9, 1.1),  # Random scaling between 90% and 110%
            ),
            transforms.RandomPerspective(
                distortion_scale=0.2, p=0.5
            ),  # Random perspective
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
            transforms.RandomErasing(p=0.2),  # Randomly erase parts of image
        ]
    )

    train_dataset = datasets.MNIST(
        "data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )

    # Initialize model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train for 1 epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(
        model.state_dict(),
        f"models/model_{timestamp}.pth",
    )


if __name__ == "__main__":
    train()
