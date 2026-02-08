import os
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from src.dataset import PatchDataset
from src.model import BuildingSegmentationModel
from src.losses import combined_loss
import random


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используем устройство: {device}")

    train_ds = PatchDataset(
        image_dir="../data/train/images",
        gt_dir="../data/train/gt",
        patch_size=256,
        stride=200
    )

    num_epochs = 20
    batch_size = 16
    patches_per_epoch = 20000
    checkpoint_dir = "../checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = BuildingSegmentationModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        total_patches = len(train_ds)
        n_patches = min(patches_per_epoch, total_patches)
        indices = random.sample(range(total_patches), n_patches)
        subset = Subset(train_ds, indices)

        train_loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

        model.train()
        total_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, (images, masks) in enumerate(progress_bar):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = combined_loss(outputs, masks, alpha=0.4, beta=0.4, gamma=0.2, kernel_size=3)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} finished. Avg loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), f"{checkpoint_dir}/model_epoch_{epoch+1}.pth")

    print("Finished!")


if __name__ == '__main__':
    main()
