import pandas as pd
from torch.utils.data import DataLoader
from datasets.ChestXrayDataset import ChestXrayDataset


def load_chest_xray_dataset(csv_file):
    df = pd.read_csv(csv_file)

    with open('data/train_val_list.txt', 'r') as f:
        train_list = set(f.read().splitlines())

    df['split'] = df['Image Index'].apply(lambda x: 'train' if x in train_list else 'test')

    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']

    return train_df, test_df


def get_dataloaders(train_df, test_df, image_dir, transform, device, batch_size=32):
    train_dataset = ChestXrayDataset(train_df, image_dir, transform, device)
    test_dataset = ChestXrayDataset(test_df, image_dir, transform, device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
