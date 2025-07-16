from PIL import Image
import os
from torch.utils.data import Dataset
import torch
import numpy as np
from collections import Counter


class ChestXrayDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, device='cuda'):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.device = device
        self.label_map = self._create_label_map()
        self.label_counter = self._create_label_counter()

    def _create_label_counter(self):
        label_counter = Counter()
        for labels in self.df['Finding Labels']:
            for label in labels.split('|'):
                label_counter[label] += 1
        return label_counter

    def _create_label_map(self):
        all_labels = set()
        for labels in self.df['Finding Labels']:
            all_labels.update(labels.split('|'))

        return {label: idx for idx, label in enumerate(sorted(all_labels))}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['Image Index'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image).to(self.device)

        labels = row['Finding Labels'].split('|')
        target = np.zeros(len(self.label_map), dtype=np.float32)

        for label in labels:
            if label in self.label_map:
                target[self.label_map[label]] = 1.0
        return image, torch.tensor(target, dtype=torch.float32).to(self.device)
