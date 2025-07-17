from torchvision import transforms, models
from PIL import Image
import torch
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
from utils import get_chest_xray_dataset, plot_grad_map
import argparse


def prepare_model_and_image(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.densenet121(num_classes=15)
    model.load_state_dict(torch.load('checkpoints/best_model.pth')['model_state_dict'])
    model.to(device)
    model.eval()

    original_image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = transform(original_image).to(device).unsqueeze(0)

    return model, original_image, image


def create_gradcam(model, original_image, image):
    cam_extractor = GradCAM(model, target_layer='features.denseblock4.denselayer16.conv2')
    pred = model(image)
    class_idx = pred.squeeze().argmax().item()
    activation_map = cam_extractor(class_idx, pred)
    mask = activation_map[0]
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    result = overlay_mask(original_image, to_pil_image(mask), alpha=0.5)

    plot_grad_map(original_image, result)


def print_predictions(pred, dataset):
    probs = torch.sigmoid(pred).squeeze()
    top_probs, top_classes = torch.topk(probs, len(dataset.label_map))

    for prob, idx in zip(top_probs, top_classes):
        print(f'{list(dataset.label_map.keys())[idx]}: {prob.item() * 100:.2f}%')


def main(image_path):
    model, original_image, image = prepare_model_and_image(image_path)
    dataset = get_chest_xray_dataset()

    create_gradcam(model, original_image, image)

    pred = model(image)
    print_predictions(pred, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to the chest X-ray image')
    args = parser.parse_args()
    main(args.image_path)
