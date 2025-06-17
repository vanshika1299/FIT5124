# Load models
from a3_mnist_fixed import My_MNIST
from model_extraction_attack_defended import Surrogate
import torch

target = My_MNIST()
target.load_state_dict(torch.load("target_model.pth"))
target.eval()

surrogate = Surrogate()
surrogate.load_state_dict(torch.load("extracted_model_strong_defence.pth"))
surrogate.eval()

# Evaluate both on MNIST test set
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_set = MNIST('./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

def evaluate(model, name):
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target_labels in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target_labels).sum().item()
            total += target_labels.size(0)
    print(f"{name} Accuracy: {100. * correct / total:.2f}%")

def agreement(model1, model2):
    agree = 0
    total = 0
    with torch.no_grad():
        for data, _ in test_loader:
            pred1 = model1(data).argmax(dim=1)
            pred2 = model2(data).argmax(dim=1)
            agree += (pred1 == pred2).sum().item()
            total += len(data)
    print(f"Prediction Agreement: {100. * agree / total:.2f}%")

evaluate(target, "Target Model")
evaluate(surrogate, "Surrogate Model")
agreement(target, surrogate)
