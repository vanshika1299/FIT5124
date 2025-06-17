import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from a3_mnist_fixed import My_MNIST

# ---------- STRONG DEFENCE FUNCTION ----------
def defend_output(logits, mode="adaptive", noise_level=0.25, flip_prob=0.25):
    probs = F.softmax(logits, dim=1)
    batch_size, num_classes = probs.shape

    if mode == "adaptive":
        # Step 1: Flip top-1 label with probability `flip_prob`
        top_class = torch.argmax(probs, dim=1)
        flip_mask = torch.rand(batch_size) < flip_prob

        random_wrong = torch.randint_like(top_class, low=0, high=num_classes)
        random_wrong = torch.where(random_wrong == top_class, (random_wrong + 1) % num_classes, random_wrong)

        final_labels = torch.where(flip_mask, random_wrong, top_class)

        # Step 2: Generate misleading soft label
        noisy_probs = torch.full((batch_size, num_classes), noise_level / (num_classes - 1))
        for i in range(batch_size):
            noisy_probs[i][final_labels[i]] = 1.0 - noise_level
        return noisy_probs

    return probs  # fallback

# ---------- STEP 1: LOAD TARGET MODEL ----------
target_model = My_MNIST()
target_model.load_state_dict(torch.load("target_model.pth"))
target_model.eval()

# ---------- STEP 2: GENERATE DEFENDED OUTPUT ----------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

query_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
query_loader = torch.utils.data.DataLoader(query_dataset, batch_size=64, shuffle=False)

inputs, soft_labels = [], []
defence_mode = "adaptive"

with torch.no_grad():
    for data, _ in query_loader:
        logits = target_model(data)
        defended_output = defend_output(logits, mode=defence_mode, noise_level=0.25, flip_prob=0.25)
        inputs.append(data)
        soft_labels.append(defended_output)

X = torch.cat(inputs)
Y = torch.cat(soft_labels)

# ---------- STEP 3: DEFINE SURROGATE MODEL ----------
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

# ---------- STEP 4: TRAIN SURROGATE MODEL ----------
optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

batch_size = 64
epochs = 5
dataset = torch.utils.data.TensorDataset(X, Y)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    total_loss = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        labels = torch.argmax(batch_y, dim=1)
        pred = surrogate_model(batch_x)
        loss = loss_fn(pred, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

# ---------- STEP 5: SAVE EXTRACTED MODEL ----------
torch.save(surrogate_model.state_dict(), "extracted_model_strong_defence.pth")
print("✅ Surrogate model saved as 'extracted_model_strong_defence.pth'")
