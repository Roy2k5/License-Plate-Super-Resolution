import torch
import os


def load_checkpoint(model, optimizer, device, is_best=False, path="checkpoint"):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    filename = "best.pt.tar" if is_best else "last.pt.tar"
    file_path = os.path.join(path, filename)
    print("[LOAD CHECKPOINT] Doing ...!")
    checkpoint = torch.load(file_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["best"], checkpoint["epoch"]


def save_checkpoint(model, optimizer, best, epoch, is_best=False, path="checkpoint"):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    filename = "best.pt.tar" if is_best else "last.pt.tar"
    file_path = os.path.join(path, filename)
    print("[SAVE CHECKPOINT] Doing ...!")
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best": best,
        "epoch": epoch,
    }
    torch.save(checkpoint, file_path)
