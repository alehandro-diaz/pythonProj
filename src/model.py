import torch.nn as nn
import torch.optim as optim
import torch as torch
import epochs as epo
import Data as dt
import plots as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleFcModel(nn.Module):
    def __init__(self, num_classes, img_size):
        super().__init__()

        H, W = img_size
        input_dim = 1 * H * W
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x


img_size = (100, 100)
model = SimpleFcModel(7, img_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

n_epoch = 50
best_f1 = 0.0

history = {
    "train_loss": [], "test_loss": [],
    "train_f1": [], "test_f1": [],
    "train_acc": [], "test_acc": []
}

for epoch in range(n_epoch):
    print(f"\nEPOCH {epoch+1}/{n_epoch}")
    train_loss, train_acc, train_f1 = epo.train_epoch(model, dt.train_loader, optimizer, criterion, device)
    test_loss, test_acc, test_f1 = epo.evaluate_epoch(model, dt.test_loader, criterion, device)

history["train_loss"].append(train_loss)
history["test_loss"].append(test_loss)
history["train_f1"].append(train_f1)
history["test_f1"].append(test_f1)
history["train_acc"].append(train_acc)
history["test_acc"].append(test_acc)

print(f"Train F1: {train_f1:.3f} | Test F1: {test_f1:.3f}")
if test_f1 > best_f1:
    best_f1 = test_f1
    torch.save(model.state_dict(), "models/best_model_fc.pth")

plt.plot_history(history)