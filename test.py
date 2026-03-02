import os
import hydra
from omegaconf import DictConfig
import torch
from models.module import ResolutionUNet
from utils import load_checkpoint
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


@hydra.main("config", "test", version_base=None)
def test(args: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = Compose([Resize((32, 64)), ToTensor(), Normalize(0.5, 0.5)])
    scenario = "Scenario-" + args.scenario
    root = os.path.join("data", "train", scenario, "Brazilian")
    model = ResolutionUNet(3, 3).to(device)
    load_checkpoint(model, None, device, is_best=True)
    fig, ax = plt.subplots(nrows=5, ncols=3, figsize=(20, 10))
    for i in range(1, 6):
        lr_path = f"lr-00{i}.jpg"
        hr_path = f"hr-00{i}.jpg"
        lr_image = Image.open(os.path.join(root, args.track, lr_path))
        hr_image = Image.open(os.path.join(root, args.track, hr_path))
        lr_image_i = transform(lr_image)
        gen = model(lr_image_i.unsqueeze(0).to(device))

        ax[i - 1][0].imshow(
            torch.clamp((gen + 1) / 2, 0, 1)
            .detach()
            .cpu()
            .numpy()
            .squeeze(0)
            .transpose(1, 2, 0)
        )
        ax[i - 1][0].set_title("Gen")
        ax[i - 1][0].axis("off")
        ax[i - 1][1].imshow(lr_image)
        ax[i - 1][1].set_title("Low")
        ax[i - 1][1].axis("off")
        ax[i - 1][2].imshow(hr_image)
        ax[i - 1][2].set_title("High")
        ax[i - 1][2].axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test()
