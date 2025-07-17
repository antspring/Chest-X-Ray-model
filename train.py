import torch
import os
from torchvision.models import DenseNet121_Weights
from utils import get_dataloaders, load_chest_xray_dataset, plot_history
from torchvision import transforms, models
from tqdm import tqdm
from sklearn.metrics import classification_report


class Trainer:
    def __init__(self, model, train_loader, test_loader, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device

        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self._create_pos_weight())
        self.optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)

        self.save_dir = config.get('save_dir', 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)

        self.best_acc = 0.0

    def _create_pos_weight(self):
        pos_weight = []
        for label in self.train_loader.dataset.label_map.keys():
            positives = self.train_loader.dataset.label_counter.get(label, 1)
            negatives = len(self.train_loader.dataset) - positives
            pos_weight.append(negatives / positives)

        return torch.tensor(pos_weight).float().to(self.device)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []

        progress_bar = tqdm(self.train_loader, desc='Training')

        for images, labels in progress_bar:
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            correct += self.loose_accuracy(outputs, labels) * images.size(0)
            total += images.size(0)
            total_acc = (correct / total).item()

            all_labels.append(labels.cpu())
            outputs = torch.sigmoid(outputs)
            all_predictions.append((outputs > 0.5).int().cpu())

            progress_bar.set_postfix(loss=loss.item(), accuracy=f"{total_acc * 100:.2f}%")

        y_true = torch.cat(all_labels).detach().numpy()
        y_pred = torch.cat(all_predictions).detach().numpy()

        print(classification_report(y_true, y_pred, target_names=list(self.train_loader.dataset.label_map.keys()),
                                    zero_division=0))

        return total_loss / len(self.train_loader), total_acc * 100

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(self.test_loader, desc='Validating')

        with torch.no_grad():
            for images, labels in progress_bar:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                correct += self.loose_accuracy(outputs, labels) * images.size(0)
                total += images.size(0)
                total_loss += loss.item()

        accuracy = (correct / total).item()

        if accuracy > self.best_acc:
            self.best_acc = accuracy
            self.save_checkpoint('best_model.pth')

        return total_loss / len(self.test_loader), accuracy * 100

    def save_checkpoint(self, filename):
        checkpoint_path = os.path.join(self.save_dir, filename)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def loose_accuracy(self, outputs, labels):
        preds = (torch.sigmoid(outputs) > 0.5).float()
        return (preds == labels).float().mean()

    def train(self):
        num_epochs = self.config.get('num_epochs', 10)
        train_loss_list = []
        train_acc_list = []
        test_loss_list = []
        test_acc_list = []

        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')

            train_loss, train_acc = self.train_epoch()
            print(f'Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.2f}%')

            if epoch + 1 == self.config.get('require_grad_epochs', 1):
                for param in self.model.parameters():
                    param.requires_grad = True

                # for param in self.optimizer.param_groups:
                #     param['lr'] = 1e-5

            test_loss, test_acc = self.validate()
            print(f'Validation accuracy: {test_acc:.2f}%')

            if (epoch + 1) % self.config.get('save_epochs', 1) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')

            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)

        return {
            'train_loss': train_loss_list,
            'train_acc': train_acc_list,
            'test_loss': test_loss_list,
            'test_acc': test_acc_list
        }


def main():
    config = {
        'num_epochs': 10,
        'save_epochs': 1,
        'save_dir': 'checkpoints',
        'require_grad_epochs': 1,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_df, test_df = load_chest_xray_dataset("data/Data_Entry_2017.csv")
    train_loader, test_loader, dataset = get_dataloaders(train_df, test_df, 'data/images', train_transform,
                                                         test_transform, device)

    model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_classes = len(dataset.label_map)

    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)

    trainer = Trainer(model, train_loader, test_loader, config, device)

    history = trainer.train()

    plot_history(history)


if __name__ == '__main__':
    main()
