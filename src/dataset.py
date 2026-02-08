import tifffile
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PatchDataset(Dataset):
    def __init__(self, image_dir, gt_dir, patch_size=256, stride=200):
        self.patch_size = patch_size
        self.stride = stride
        self.patches = []

        image_paths = sorted(Path(image_dir).glob("*.tif"))
        gt_paths = sorted(Path(gt_dir).glob("*.tif"))

        for img_path, gt_path in zip(image_paths, gt_paths):
            h, w = tifffile.imread(str(img_path)).shape[:2]

            y_positions = self._generate_patch_positions(h)
            x_positions = self._generate_patch_positions(w)

            for y in y_positions:
                for x in x_positions:
                    self.patches.append((img_path, gt_path, x, y))

        # print(f"Создано {len(self.patches)} патчей (size = {patch_size}, stride = {stride}) из {len(image_paths)} изображений.")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_path, gt_path, x, y = self.patches[idx]

        full_img = tifffile.imread(str(img_path))
        full_gt = tifffile.imread(str(gt_path))

        img_patch = full_img[y:y + self.patch_size, x:x + self.patch_size]
        gt_patch = full_gt[y:y + self.patch_size, x:x + self.patch_size]

        img = torch.from_numpy(img_patch).permute(2, 0, 1).float()
        gt = torch.from_numpy(gt_patch).unsqueeze(0).float()

        if img.max() > 1.0:
            img = img / 255.0
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = normalize(img)

        gt = (gt > 0.5).float()

        return img, gt, x, y, img_path.name

    def _generate_patch_positions(self, length):
        positions = []
        pos = 0

        while pos + self.patch_size <= length:
            positions.append(pos)
            pos += self.stride

        last_pos = length - self.patch_size
        if last_pos >= 0 and (len(positions) == 0 or positions[-1] != last_pos):
            positions.append(last_pos)

        return positions


class InferencePatchDataset:
    def __init__(self, img_rgb, patch_size=512, stride=None):
        self.img_rgb = img_rgb
        self.patch_size = patch_size
        self.stride = stride if stride is not None else int(patch_size * 0.8)
        self.h, self.w = img_rgb.shape[:2]

        self.patches = []
        y = 0
        while y < self.h:
            x = 0
            while x < self.w:
                self.patches.append((x, y))
                x += self.stride
            y += self.stride

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        x, y = self.patches[idx]
        patch = self.img_rgb[y:y + self.patch_size, x:x + self.patch_size]
        ph, pw = patch.shape[:2]
        if ph < self.patch_size or pw < self.patch_size:
            pad_h, pad_w = self.patch_size - ph, self.patch_size - pw
            patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        return patch, x, y
