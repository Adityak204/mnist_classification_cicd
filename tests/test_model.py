import torch
import pytest
from model.network import SimpleCNN
from torchvision import datasets, transforms


def test_model_parameters():
    model = SimpleCNN()
    total_params = sum(p.numel() for p in model.parameters())
    assert (
        total_params < 25000
    ), f"Model has {total_params} parameters, should be less than 25000"


def test_input_output_shape():
    model = SimpleCNN()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"


def test_model_train_accuracy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)

    # Load the latest model
    import glob
    import os

    model_files = glob.glob("models/*.pth")
    latest_model = max(model_files, key=os.path.getctime)
    model.load_state_dict(torch.load(latest_model))

    # Load train dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(
        "data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    train_accuracy = 100 * correct / total
    assert (
        train_accuracy > 95
    ), f"Model training accuracy is {train_accuracy}%, should be > 95%"


def test_model_test_accuracy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)

    # Load the latest model
    import glob
    import os

    model_files = glob.glob("models/*.pth")
    latest_model = max(model_files, key=os.path.getctime)
    model.load_state_dict(torch.load(latest_model))

    # Load test dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    test_dataset = datasets.MNIST(
        "data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_accuracy = 100 * correct / total
    assert (
        test_accuracy > 95
    ), f"Model test accuracy is {test_accuracy}%, should be > 95%"
