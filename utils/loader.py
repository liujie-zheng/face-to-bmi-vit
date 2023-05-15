import os
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, RandomCrop, RandomRotation, ColorJitter



class BMIDataset(Dataset):
    def __init__(self, csv_path, image_folder, y_col_name, transform=None):
        self.csv = pd.read_csv(csv_path)
        self.image_folder = image_folder
        self.y_col_name = y_col_name
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.csv.iloc[idx, 4])
        image = Image.open(image_path)
        y = self.csv.loc[idx, self.y_col_name]

        if self.transform:
            image = self.transform(image)

        return image, y



def show_sample_image(dataset):
    loader = DataLoader(dataset)

    train_features, train_labels = next(iter(loader))
    print(f"Feature batch shape: {train_features.size()}")
    image = train_features[0].squeeze()
    label = train_labels[0].item()

    plt.imshow(image.detach().cpu().permute(1, 2, 0), cmap="gray")
    plt.title('ground truth BMI: ' + str(label))
    plt.axis(False)
    plt.show()



def train_val_test_split(dataset):
    val_size = int(0.1 * len(dataset))
    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset



def get_dataloaders(train_dataset, val_dataset, test_dataset, augmented=True):
    return




train_transforms = [
    Compose([ToTensor()]),  # no augmentation
    Compose([RandomHorizontalFlip(), ToTensor()]),
    Compose([RandomCrop(32, padding=4), ToTensor()]),
    Compose([RandomRotation(30), ToTensor()]),
    Compose([ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), ToTensor()])
]

if __name__ == "__main__":
    bmi_dataset = BMIDataset('../data/data.csv', '../data/Images', 'bmi', ToTensor())
    show_sample_image(bmi_dataset)
