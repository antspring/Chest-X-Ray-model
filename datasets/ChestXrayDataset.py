from PIL import Image
import os
from torch.utils.data import Dataset


class ChestXrayDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, device='cuda'):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.device = device
        self.classes = {label: idx for idx, label in enumerate(df['Finding Labels'].unique())}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['Image Index'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image).to(self.device)

        label = self.classes[row['Finding Labels']]
        return image, label
