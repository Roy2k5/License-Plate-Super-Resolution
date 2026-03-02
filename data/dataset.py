from torch.utils.data import Dataset
import os
from PIL import Image


class ICPRDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.track_list = os.listdir(self.root)

    def __len__(self):
        return 5 * len(self.track_list)

    def __getitem__(self, index):
        base = index // 5
        offset = index % 5 + 1
        hr_path = f"hr-00{offset}.png"
        lr_path = f"lr-00{offset}.png"
        image = Image.open(os.path.join(self.root, self.track_list[base], lr_path))
        label = Image.open(os.path.join(self.root, self.track_list[base], hr_path))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


if __name__ == "__main__":
    from torchvision.transforms import Compose, Resize, ToTensor

    transform = Compose([Resize((32, 64)), ToTensor()])
    dataset = ICPRDataset(
        os.path.join("data", "train", "Scenario-A", "Brazilian"),
        transform=transform,
        target_transform=transform,
    )
    img, label = dataset[0]
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(img.permute(1, 2, 0))
    plt.figure()
    plt.imshow(label.permute(1, 2, 0))
    plt.show()
