import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, random_split

# Step 1: Load trained target model
from a3_mnist_fixed import My_MNIST
model = My_MNIST()
model.load_state_dict(torch.load("target_model.pth"))
model.eval()

# Step 2: Load data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
full_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_set, val_set = random_split(full_data, [30000, 30000])  # simulate known & unknown

train_loader = DataLoader(train_set, batch_size=64, shuffle=False)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

# Step 3: Collect confidence vectors
def get_confidences(loader, label):
    X, y = [], []
    with torch.no_grad():
        for data, _ in loader:
            logits = model(data)
            probs = F.softmax(logits, dim=1)  # shape: [batch_size, 10]
            X.append(probs)
            y.append(torch.full((data.shape[0], 1), label))  # 1 for 'IN', 0 for 'OUT'
    return torch.cat(X), torch.cat(y)


X_train, y_train = get_confidences(train_loader, label=1)  # member
X_val, y_val = get_confidences(val_loader, label=0)        # non-member

X_attack = torch.cat([X_train, X_val])
y_attack = torch.cat([y_train, y_val])

# Step 4: Define and train attack model
class AttackModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x)

attack_model = AttackModel()
optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

attack_dataset = TensorDataset(X_attack, y_attack)
attack_loader = DataLoader(attack_dataset, batch_size=64, shuffle=True)

# Training loop
for epoch in range(5):
    total_loss = 0
    for xb, yb in attack_loader:
        optimizer.zero_grad()
        pred = attack_model(xb)
        loss = loss_fn(pred, yb.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

# Step 5: Evaluate attack model
with torch.no_grad():
    pred = (attack_model(X_attack) > 0.5).float()
    acc = (pred == y_attack).float().mean().item()
    print(f"✅ Attack Model Accuracy: {acc * 100:.2f}%")
