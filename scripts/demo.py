from PIL import Image
from matplotlib import pyplot as plt

import torch
from torchvision.transforms import ToTensor

from models import get_model
from loader import vit_transforms



def test_and_show(img_dir, weight_dir):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # open and transform image for vit
    image = Image.open(img_dir)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = ToTensor()(image)
    image_vit = vit_transforms(image)
    image_vit = image_vit.unsqueeze(0)
    image_vit = image_vit.to(device)

    # get model and predict
    model = get_model()
    model = model.to(device)
    model.load_state_dict(torch.load(weight_dir, map_location=device))
    model.eval()
    with torch.no_grad():
        pred = model(image_vit)

    # plot
    plt.imshow(image.cpu().detach().numpy().transpose(1, 2, 0))
    plt.axis("off")
    plt.title(f"Predicted BMI: {pred.item():>5f}")
    plt.show()

    return pred.item()


if __name__ == "__main__":
    pred = test_and_show('../data/test_pic.jpg', '../weights/no_aug_epoch_10.pt')
    print(f'Predicted BMI: {pred}')