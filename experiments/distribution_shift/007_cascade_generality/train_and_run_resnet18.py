"""
Train ResNet18 on CIFAR-10, apply a distribution shift, and run the actual
Cascade package (not raw notebook code) against it — this time with 8
real conv layers, so diagnose() (point-of-no-return + stable/unstable)
finally has enough depth to potentially say something meaningful.

Run with: python3 train_and_run_resnet18.py

Note: training ResNet18 on CIFAR-10 for 10 epochs on a Mac (even with MPS)
will be noticeably slower than the tiny MNIST CNN — expect this to take
a while longer. If it's too slow, drop `epochs` to 3-5 for a first pass;
the model doesn't need to be great, it just needs to be shifted-fragile.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as T

from cascade import Cascade, fragility_profile
from cascade.diagnose import diagnose

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# 1. Data — CIFAR-10 channel stats, deterministic shift (rotation + blur)
# ---------------------------------------------------------------------------
transform_norm = T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
clean_transform = T.Compose([T.ToTensor(), transform_norm])

# IMPORTANT: degrees=(45, 45) forces exactly +45 every time (deterministic).
# A random range like RandomRotation(45) would give a *different* image on
# every access, corrupting D(k) since clean/shifted must be the same
# underlying image.
shift_transform = T.Compose(
    [T.RandomRotation(degrees=(45, 45)), T.GaussianBlur(5), T.ToTensor(), transform_norm]
)

BATCH_SIZE = 128
train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=clean_transform
)
test_dataset_clean = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=clean_transform
)
test_dataset_shift = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=shift_transform
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader_clean = torch.utils.data.DataLoader(test_dataset_clean, batch_size=BATCH_SIZE, shuffle=False)
test_loader_shift = torch.utils.data.DataLoader(test_dataset_shift, batch_size=BATCH_SIZE, shuffle=False)


# ---------------------------------------------------------------------------
# 2. Model — ResNet18 adapted for 32x32 CIFAR-10 images
# ---------------------------------------------------------------------------
model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Linear(512, 10)
model = model.to(device)

# ---------------------------------------------------------------------------
# 3. Train (start with 5 epochs — reduce further if too slow on your machine)
# ---------------------------------------------------------------------------
EPOCHS = 5
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f"\nTraining for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}/{EPOCHS} — Loss: {running_loss / len(train_loader):.4f}")

# ---------------------------------------------------------------------------
# 4. Evaluate clean vs shifted
# ---------------------------------------------------------------------------
def evaluate_accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            _, predicted = torch.max(model(images), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


clean_acc = evaluate_accuracy(model, test_loader_clean)
shifted_acc = evaluate_accuracy(model, test_loader_shift)
print(f"\nClean accuracy:   {clean_acc:.2f}%")
print(f"Shifted accuracy: {shifted_acc:.2f}%")

# ---------------------------------------------------------------------------
# 5. Collect misclassified shifted samples (dataset indices, not tensors —
#    lighter on memory, matches how the original notebook did it)
# ---------------------------------------------------------------------------
misclassified_shift = []
model.eval()
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader_shift):
        images, labels = images.to(device), labels.to(device)
        _, predicted = torch.max(model(images), 1)
        for j in range(len(labels)):
            if predicted[j] != labels[j]:
                dataset_idx = batch_idx * BATCH_SIZE + j
                if dataset_idx < len(test_dataset_shift):
                    misclassified_shift.append(
                        (dataset_idx, labels[j].item(), predicted[j].item())
                    )

print(f"Found {len(misclassified_shift)} misclassified samples in the shifted test set.")

# ---------------------------------------------------------------------------
# 6. Build (clean_img, shifted_img, true_label, pred_label) tuples for Cascade
# ---------------------------------------------------------------------------
N_SAMPLES = 100  # keep modest for a first local run
samples = []
for dataset_idx, true_label, pred_label in misclassified_shift[:N_SAMPLES]:
    clean_img, _ = test_dataset_clean[dataset_idx]
    shifted_img, _ = test_dataset_shift[dataset_idx]
    samples.append((clean_img, shifted_img, true_label, pred_label))

# ---------------------------------------------------------------------------
# 7. Run Cascade — auto-discovers all Conv2d layers in ResNet18 (many more
#    than 2 this time), giving diagnose() actual room to work with.
#    max_layers=8 caps it to 8 evenly-sampled layers across depth, matching
#    the original notebook's 8 hand-picked sublayers, but done generically.
# ---------------------------------------------------------------------------
print(f"\nRunning Cascade on {len(samples)} samples (8 layers, auto-discovered)...")
cascade = Cascade(model, max_layers=8, device=str(device))
print(f"Instrumented layers: {cascade.layer_names}")

profile = fragility_profile(cascade, samples)
print()
print(profile.summary())

# ---------------------------------------------------------------------------
# 8. Per-sample diagnosis — the actual test of whether PNR and stable/
#    unstable can differentiate between samples now that there's real depth
# ---------------------------------------------------------------------------
print("\n--- Per-sample diagnosis (first 5 samples) ---")
for i, (clean_img, shifted_img, true_label, pred_label) in enumerate(samples[:5]):
    result = diagnose(cascade, clean_img, shifted_img, true_label, pred_label)
    print(f"\nSample {i}: true={true_label}, pred={pred_label}")
    print(result.summary())

# ---------------------------------------------------------------------------
# 9. Baseline comparison — MSP and entropy on the same samples
# ---------------------------------------------------------------------------
def predict_with_confidence(model, image, threshold=0.7):
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        probs = F.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)
    if confidence.item() < threshold:
        return "REJECT", confidence.item()
    return pred.item(), confidence.item()


def entropy_score(model, image):
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        probs = F.softmax(output, dim=1)
    return -torch.sum(probs * torch.log(probs + 1e-9), dim=1).item()


print("\n--- Baseline comparison (MSP / entropy) on same samples ---")
for i, (clean_img, shifted_img, true_label, pred_label) in enumerate(samples[:5]):
    msp_result, msp_conf = predict_with_confidence(model, shifted_img)
    ent = entropy_score(model, shifted_img)
    print(f"Sample {i}: MSP={msp_result} (conf={msp_conf:.3f}), entropy={ent:.3f}")

print("\n--- Comparison summary ---")
print("Check above: do Cascade's PNR layer + stable/unstable verdict actually")
print("differ between samples now (unlike the 2-layer MNIST run)? And does")
print("that variation line up with which samples MSP/entropy flagged as")
print("most uncertain?")