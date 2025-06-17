import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from a3_mnist_fixed import My_MNIST
import numpy as np

# Step 1: Load target model
target_model = My_MNIST()
target_model.load_state_dict(torch.load("target_model.pth"))
target_model.eval()

# Step 2: Generate synthetic query dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

query_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
query_loader = torch.utils.data.DataLoader(query_dataset, batch_size=64, shuffle=False)

inputs, soft_labels = [], []

with torch.no_grad():
    for data, _ in query_loader:
        output = target_model(data)
        inputs.append(data)
        soft_labels.append(F.softmax(output, dim=1))  # Get confidence vectors

X = torch.cat(inputs)
Y = torch.cat(soft_labels)

# Step 3: Define surrogate model
class Surrogate(nn.Module):
    def __init__(self):
        super(Surrogate, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return F.log_softmax(self.fc(x), dim=1)

surrogate_model = Surrogate()

# Step 4: Train surrogate model using KL-Divergence (distillation)
optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=0.001)
loss_fn = nn.KLDivLoss(reduction="batchmean")

batch_size = 64
epochs = 5
dataset = torch.utils.data.TensorDataset(X, Y)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    total_loss = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        pred = surrogate_model(batch_x)
        loss = loss_fn(pred, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Save extracted model
torch.save(surrogate_model.state_dict(), "extracted_model.pth")
print("✅ Surrogate model saved as 'extracted_model.pth'")
