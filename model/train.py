import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from unet import UNet


# ---------------- Dataset ---------------- #
class UnderwaterDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.images = os.listdir(input_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]

        inp = cv2.imread(os.path.join(self.input_dir, name))
        tar = cv2.imread(os.path.join(self.target_dir, name))

        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        tar = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)

        inp = cv2.resize(inp, (256, 256)) / 255.0
        tar = cv2.resize(tar, (256, 256)) / 255.0

        inp = torch.tensor(inp).permute(2, 0, 1).float()
        tar = torch.tensor(tar).permute(2, 0, 1).float()

        return inp, tar


# ---------------- Training ---------------- #
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 Training started on", device)

    train_ds = UnderwaterDataset("../data/train/input", "../data/train/target")
    val_ds   = UnderwaterDataset("../data/val/input", "../data/val/target")

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = UNet().to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    best_val_loss = float("inf")
    epochs = 5

    for epoch in range(1, epochs + 1):
        print(f"\n🔁 Epoch {epoch}/{epochs} started")
        model.train()
        train_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        train_loss /= len(train_loader)

        # -------- Validation -------- #
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item()

        val_loss /= len(val_loader)

        print(f"✅ Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "unet_best.pth")
            print("💾 Best model saved")

    print("\n🎉 Training completed successfully!")


if __name__ == "__main__":
    main()
