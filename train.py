import argparse
import os
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import numpy as np
from data.dataset import ICPRDataset
from models.module import ResolutionUNet
from utils import load_checkpoint, save_checkpoint

torch.manual_seed(42)


def get_args():
    parser = argparse.ArgumentParser("Train Unet for resolution task")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--continue-train", action="store_true")
    return parser.parse_args()


def train(args):
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = Compose([Resize((32, 64)), ToTensor(), Normalize(0.5, 0.5)])
    dataset = ICPRDataset(
        os.path.join("data", "train", "Scenario-A", "Brazilian"),
        transform=transform,
        target_transform=transform,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = ResolutionUNet(in_channel=3, out_channel=3)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    best_loss = 1e7
    start = 0
    model = model.to(device)
    if args.continue_train:
        best, start = load_checkpoint(model, optimizer, device)

    writer = SummaryWriter("logs/experiment1", flush_secs=30)
    for epoch in range(start, epochs):
        progress_bar = tqdm(dataloader, colour="cyan", leave=False)
        losses = []
        for image, label in progress_bar:
            image = image.to(device)
            label = label.to(device)
            logits = model(image)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().item())
            progress_bar.set_description(f"Loss: {loss.cpu().item()}")
        lr_image = torch.clamp((image + 1) / 2, 0, 1)
        hr_image = torch.clamp((label + 1) / 2, 0, 1)
        gen_image = torch.clamp((logits + 1) / 2, 0, 1)
        lr_image = make_grid(lr_image, 4, pad_value=1)
        hr_image = make_grid(hr_image, 4, pad_value=1)
        gen_image = make_grid(gen_image, 4, pad_value=1)

        mean_loss = np.sum(losses) / len(losses)
        writer.add_scalar("Loss/MSE", mean_loss, epoch)
        writer.add_image("Image/Low", lr_image, epoch)
        writer.add_image("Image/High", hr_image, epoch)
        writer.add_image("Image/Gen", gen_image, epoch)
        if best_loss > mean_loss:
            best_loss = mean_loss
            save_checkpoint(model, optimizer, best_loss, epoch + 1, True)
        save_checkpoint(model, optimizer, best_loss, epoch + 1, False)


if __name__ == "__main__":
    train(get_args())
