import pandas as pd
from torch.utils.data import DataLoader
from datasets.ChestXrayDataset import ChestXrayDataset
import matplotlib.pyplot as plt


def load_chest_xray_dataset(csv_file):
    df = pd.read_csv(csv_file)

    with open('data/train_val_list.txt', 'r') as f:
        train_list = set(f.read().splitlines())

    df['split'] = df['Image Index'].apply(lambda x: 'train' if x in train_list else 'test')

    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']

    return train_df, test_df


def get_dataloaders(train_df, test_df, image_dir, train_transform, test_transform, device, batch_size=32):
    train_dataset = ChestXrayDataset(train_df, image_dir, train_transform, device)
    test_dataset = ChestXrayDataset(test_df, image_dir, test_transform, device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset


def plot_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()


def get_chest_xray_dataset():
    train_df, _ = load_chest_xray_dataset("data/Data_Entry_2017.csv")

    return ChestXrayDataset(train_df, 'data/images')

def plot_grad_map(original_image, result):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original_image)
    axes[0].axis('off')
    axes[0].set_title('Original Image')

    axes[1].imshow(result)
    axes[1].axis('off')
    axes[1].set_title('Grad-CAM Result')

    plt.tight_layout()
    plt.savefig('grad_cam_result.png')
    plt.close()
