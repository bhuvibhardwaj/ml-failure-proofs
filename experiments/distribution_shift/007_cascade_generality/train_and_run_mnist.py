"""
Train a small CNN on MNIST, apply a distribution shift, and run Cascade
against it — plus the MSP/entropy baseline comparison.

Run with: python3 train_and_run.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from cascade import Cascade, fragility_profile
from cascade.diagnose import diagnose

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
clean_transform = T.Compose([T.ToTensor()])
shift_transform = T.Compose(
    [T.RandomRotation(degrees=(75, 75)), T.GaussianBlur(kernel_size=7), T.ToTensor()]
)

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=clean_transform
)
test_dataset_clean = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=clean_transform
)
test_dataset_shift = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=shift_transform
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader_clean = torch.utils.data.DataLoader(test_dataset_clean, batch_size=1, shuffle=False)
test_loader_shift = torch.utils.data.DataLoader(test_dataset_shift, batch_size=1, shuffle=False)


# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


model = CNN().to(device)

# ---------------------------------------------------------------------------
# 3. Train (5 epochs — quick, matches earlier notebook)
# ---------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nTraining...")
for epoch in range(5):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}/5 — Loss: {running_loss / len(train_loader):.4f}")

# ---------------------------------------------------------------------------
# 4. Evaluate clean vs shifted, collect misclassified shifted samples
# ---------------------------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    misclassified = []
    with torch.no_grad():
        for image, lbl in loader:
            image, lbl = image.to(device), lbl.to(device)
            output = model(image)
            _, pred = torch.max(output, 1)
            total += 1
            if pred == lbl:
                correct += 1
            else:
                misclassified.append((image, lbl.item(), pred.item()))
    return correct / total, misclassified


clean_acc, _ = evaluate(model, test_loader_clean)
shifted_acc, misclassified_shift = evaluate(model, test_loader_shift)

print(f"\nClean accuracy:   {clean_acc:.4f}")
print(f"Shifted accuracy: {shifted_acc:.4f}")
print(f"Misclassified under shift: {len(misclassified_shift)}")

# ---------------------------------------------------------------------------
# 5. Build (clean_img, shifted_img, true_label, pred_label) tuples for Cascade
# ---------------------------------------------------------------------------
n_samples = 100  # keep small for a first local run
samples = []
for i, (shifted_img, true_label, pred_label) in enumerate(misclassified_shift[:n_samples]):
    clean_img, _ = test_dataset_clean[i]
    clean_img = clean_img.unsqueeze(0)  # match batch dim of shifted_img
    samples.append((clean_img, shifted_img, true_label, pred_label))

print(f"\nRunning Cascade on {len(samples)} samples...")
cascade = Cascade(model, device=str(device))
profile = fragility_profile(cascade, samples)
print()
print(profile.summary())

# ---------------------------------------------------------------------------
# 6. Per-sample diagnosis on the first few samples
# ---------------------------------------------------------------------------
print("\n--- Per-sample diagnosis (first 3 samples) ---")
for i, (clean_img, shifted_img, true_label, pred_label) in enumerate(samples[:3]):
    result = diagnose(cascade, clean_img, shifted_img, true_label, pred_label)
    print(f"\nSample {i}: true={true_label}, pred={pred_label}")
    print(result.summary())

# ---------------------------------------------------------------------------
# 7. Baseline comparison — MSP and entropy rejection on the same samples
# ---------------------------------------------------------------------------
def predict_with_confidence(model, image, threshold=0.7):
    model.eval()
    with torch.no_grad():
        output = model(image.to(device))
        probs = F.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)
    if confidence.item() < threshold:
        return "REJECT", confidence.item()
    return pred.item(), confidence.item()


def entropy_score(model, image):
    model.eval()
    with torch.no_grad():
        output = model(image.to(device))
        probs = F.softmax(output, dim=1)
    return -torch.sum(probs * torch.log(probs + 1e-9), dim=1).item()


print("\n--- Baseline comparison (MSP / entropy) on same samples ---")
for i, (clean_img, shifted_img, true_label, pred_label) in enumerate(samples[:3]):
    msp_result, msp_conf = predict_with_confidence(model, shifted_img)
    ent = entropy_score(model, shifted_img)
    print(f"Sample {i}: MSP={msp_result} (conf={msp_conf:.3f}), entropy={ent:.3f}")