# MNIST Classification Model with CI/CD Pipeline

This repository contains a machine learning project with a complete CI/CD pipeline using GitHub Actions. The project implements a simple Convolutional Neural Network (CNN) for MNIST digit classification with automated testing and deployment.

## Project Structure
```
.
├── model/
│ ├── __init__.py
│ └── network.py # CNN architecture definition
├── tests/
│ └── test_model.py # Model tests
├── train.py # Training script
├── requirements.txt # Project dependencies
├── .github/
│ └── workflows/
│   └── ml-pipeline.yml # GitHub Actions workflow
├── .gitignore # Git ignore rules
└── README.md # Project documentation
```

## Model Architecture

The project implements a lightweight CNN with:
- Input layer: Accepts 28x28 grayscale images
- First Conv layer: 6 filters, 3x3 kernel, padding=1 (output: 28x28x6)
- Second Conv layer: 6 filters, 3x3 kernel, padding=1 (output: 28x28x6)
- 2 MaxPooling layers (2x2) reducing dimensions to 7x7x6
- First FC layer: 294 (6*7*7) → 64 units
- Output layer: 64 → 10 units (digit classes)
- ReLU activations throughout

The model is designed to be lightweight (<100,000 parameters) while achieving >95% accuracy on MNIST.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- pytest

## Local Setup

1. Clone the repository:

2. Create and activate a virtual environment:
```bash
conda create -n ml-pipeline python=3.10
conda activate ml-pipeline
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

To train the model:
```bash
python train.py
```

The training process:
- Uses MNIST dataset (automatically downloaded)
- Trains for 1 epoch
- Uses Adam optimizer with learning rate 0.001
- Uses CrossEntropyLoss
- Batch size of 64
- Prints loss every 100 batches
- Saves model with timestamp in `models/` directory

The trained model will be saved as `model_YYYYMMDD_HHMMSS.pth`.

## Testing

Run the tests using:
```bash
pytest tests/test_model.py
```

The tests verify:
1. Model Architecture Test:
   - Verifies model has less than 25,000 parameters
   - Checks input shape (28x28) and output shape (10 classes)

2. Model Performance Test:
   - Loads the latest trained model
   - Tests on MNIST test dataset
   - Verifies accuracy > 95%

## CI/CD Pipeline Details

The GitHub Actions workflow is defined in `.github/workflows/ml-pipeline.yml` and consists of the following steps:

### Trigger
```yaml
on: [push]
```
The pipeline is triggered automatically on every push to any branch.

### Environment
- Runs on: Ubuntu Latest
- Python version: 3.8

### Job Steps

1. **Checkout Code**
   - Uses: `actions/checkout@v4`
   - Clones the repository into the GitHub Actions runner

2. **Setup Python**
   - Uses: `actions/setup-python@v5`
   - Configures Python 3.8 environment

3. **Install Dependencies**
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Train Model**
   - Runs training script: `python train.py`
   - Creates model file with timestamp

5. **Run Tests**
   - Executes: `pytest tests/`
   - Validates:
     - Model architecture
     - Parameter count
     - Input/output shapes
     - Model accuracy

6. **Artifact Upload**
   - Uses: `actions/upload-artifact@v4`
   - Uploads trained model to GitHub Actions artifacts
   - Artifact name: "trained-model"
   - Path: "models/"
   - Retention period: 90 days

### Workflow Status
You can monitor the workflow status:
1. Go to your GitHub repository
2. Click on "Actions" tab
3. View latest workflow runs
4. Download model artifacts from successful runs

### Error Handling
The workflow will fail if:
- Dependencies fail to install
- Training fails
- Any test fails (architecture or accuracy)
- Model file fails to save
