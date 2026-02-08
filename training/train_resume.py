import os
import torch
import random
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split

from src.dataset import PatchDataset
from src.model import BuildingSegmentationModel
from src.losses import combined_loss


def calculate_iou(pred, target, threshold=0.5):
    with torch.no_grad():
        pred = (pred > threshold).float()
        target = (target > 0.5).float()
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = (pred + target - pred * target).sum(dim=(1, 2, 3))
        iou = intersection / (union + 1e-6)
        return iou.mean().item()


def split_patches_by_image(patches, test_size=0.2, random_state=42):
    image_to_patches = defaultdict(list)
    for i, (img_path, gt_path, x, y) in enumerate(patches):
        image_to_patches[Path(img_path).name].append(i)

    image_names = list(image_to_patches.keys())
    train_names, val_names = train_test_split(
        image_names, test_size=test_size, random_state=random_state
    )

    train_indices = [idx for name in train_names for idx in image_to_patches[name]]
    val_indices = [idx for name in val_names for idx in image_to_patches[name]]

    return train_indices, val_indices


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используем устройство: {device}")

    model = BuildingSegmentationModel().to(device)
    checkpoint_path = "../checkpoints/model_epoch_39.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    full_ds = PatchDataset(
        image_dir="../data/train/images",
        gt_dir="../data/train/gt",
        patch_size=1024,
        stride=860
    )

    train_indices, val_indices = split_patches_by_image(full_ds.patches, test_size=0.2)
    print(f"Train patches: {len(train_indices)}, Val patches: {len(val_indices)}")

    train_ds = Subset(full_ds, train_indices)
    val_ds = Subset(full_ds, val_indices)

    start_epoch = 39
    total_epochs = 43
    batch_size = 2
    patches_per_epoch = 20000
    checkpoint_dir = "../checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_iou = 0.7105
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

    for epoch in range(start_epoch, total_epochs):
        indices = random.sample(range(len(train_ds)), min(patches_per_epoch, len(train_ds)))
        epoch_subset = Subset(train_ds, indices)
        train_loader = DataLoader(epoch_subset, batch_size=batch_size, shuffle=True, num_workers=0)

        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs}")

        for batch in progress_bar:
            images = batch[0].to(device)
            masks = batch[1].to(device)

            outputs = model(images)
            loss = combined_loss(outputs, masks, alpha=0.4, beta=0.4, gamma=0.2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_iou = 0.0
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        with torch.no_grad():
            for batch in val_loader:
                images = batch[0].to(device)
                masks = batch[1].to(device)
                outputs = model(images)
                val_iou += calculate_iou(outputs, masks)
        avg_val_iou = val_iou / len(val_loader)

        print(f"\nEpoch {epoch + 1}")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val IoU:    {avg_val_iou:.4f}")

        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pth")

        torch.save(model.state_dict(), f"{checkpoint_dir}/model_epoch_{epoch + 1}.pth")

    print(f"Finished, best IoU: {best_val_iou:.4f}")


if __name__ == '__main__':
    main()
