import argparse
import os
import random

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from Loss import CtdetLoss
from backbone.dlanet_dcn import MyNet
from dataset import ctDataset


def parse_opt():
    parser = argparse.ArgumentParser(description="Train EDP2Net/ElDet ellipse detector.")
    parser.add_argument("--data", type=str, required=True, help="training data folder")
    parser.add_argument("--val_data", type=str, default=None, help="validation data folder")
    parser.add_argument("--save_path", type=str, default="results/train", help="checkpoint and log directory")
    parser.add_argument("--epochs", "--epoch", dest="epochs", type=int, default=120, help="number of epochs")
    parser.add_argument("--device", type=str, default="cuda:0", help="torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--num_workers", type=int, default=8, help="dataloader workers")
    parser.add_argument("--save_interval", type=int, default=10, help="save last checkpoint every N epochs")

    parser.add_argument("--batch_size", type=int, default=16, help="train batch size")
    parser.add_argument("--val_batch_size", type=int, default=8, help="validation batch size")
    parser.add_argument("--lr", type=float, default=1.25e-4, help="learning rate")
    parser.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--w_hm", type=float, default=1, help="heatmap loss weight")
    parser.add_argument("--w_ab", type=float, default=0.1, help="axis size loss weight")
    parser.add_argument("--w_ang", type=float, default=0.1, help="angle loss weight")
    parser.add_argument("--w_reg", type=float, default=0.1, help="offset loss weight")
    parser.add_argument("--w_iou", type=float, default=15, help="Gaussian IoU loss weight")
    parser.add_argument("--w_mask", type=float, default=1, help="mask loss weight")
    parser.add_argument("--hmTop_weight", type=float, default=5, help="top-point heatmap loss weight")
    parser.add_argument("--comp_weight", type=float, default=1, help="compactness loss weight")

    parser.add_argument("--use_wandb", action="store_true", help="enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="edp2net", help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")
    return parser.parse_args()


def seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_device(device_name):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def build_wandb_logger(opt):
    if not opt.use_wandb:
        return None
    import wandb

    wandb.init(
        project=opt.wandb_project,
        name=opt.wandb_run_name,
        tags=[os.path.basename(os.path.normpath(opt.data))],
        config=vars(opt),
    )
    return wandb


def train(opt):
    if opt.seed is not None:
        seed_torch(opt.seed)

    save_path = opt.save_path
    os.makedirs(save_path, exist_ok=True)
    writer = SummaryWriter(save_path)
    wandb_logger = build_wandb_logger(opt)

    model = MyNet(34)
    loss_weight = {
        "hm_weight": opt.w_hm,
        "ab_weight": opt.w_ab,
        "ang_weight": opt.w_ang,
        "reg_weight": opt.w_reg,
        "iou_weight": opt.w_iou,
        "mask_weight": opt.w_mask,
        "hmTop_weight": opt.hmTop_weight,
        "comp_weight": opt.comp_weight,
    }
    criterion = CtdetLoss(loss_weight)

    device = build_device(opt.device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, eps=opt.eps, amsgrad=True)
    train_dataset = ctDataset(data_dir=opt.data, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        drop_last=True,
    )

    val_loader = None
    if opt.val_data:
        val_dataset = ctDataset(data_dir=opt.val_data, split="val")
        val_loader = DataLoader(
            val_dataset,
            batch_size=opt.val_batch_size,
            shuffle=False,
            num_workers=opt.num_workers,
        )
        print("the val dataset has %d images" % len(val_dataset))

    print("the train dataset has %d images" % len(train_dataset))

    best_val_loss = float("inf")
    for epoch in range(opt.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(
            total=len(train_dataset),
            desc="[Epoch {:03d}]".format(epoch + 1),
            mininterval=0.1,
            postfix={"Loss": np.inf, "Average_loss": np.inf},
        )

        for i, sample in enumerate(train_loader):
            for k in sample:
                sample[k] = sample[k].to(device=device, non_blocking=True)

            pred = model(sample["input"], True)
            loss, loss_show = criterion(pred, sample)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step = epoch * len(train_loader) + i
            avg_loss = total_loss / (i + 1)
            writer.add_scalar("Train/Loss", avg_loss, global_step=global_step)
            writer.add_scalar("Train/hm_loss", loss_show[0], global_step=global_step)
            writer.add_scalar("Train/ab_loss", loss_show[1], global_step=global_step)
            writer.add_scalar("Train/ang_loss", loss_show[2], global_step=global_step)
            writer.add_scalar("Train/reg_loss", loss_show[3], global_step=global_step)
            writer.add_scalar("Train/iou_loss", loss_show[4], global_step=global_step)

            if wandb_logger is not None and i % 20 == 0:
                wandb_logger.log(
                    {
                        "Train Loss": avg_loss,
                        "hm Loss": loss_show[0],
                        "iou_loss": loss_show[4],
                        "center_offset loss": loss_show[3],
                    }
                )

            pbar.set_postfix({"Loss": loss.item(), "Average_loss": avg_loss, "losses": loss_show})
            pbar.update(opt.batch_size)

        pbar.close()

        if val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device)
            writer.add_scalar("Val/Loss", val_loss, global_step=epoch + 1)
            if wandb_logger is not None:
                wandb_logger.log({"Val Loss": val_loss, "epoch": epoch + 1})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("* Best val loss %.5f" % best_val_loss)
                torch.save(model.state_dict(), os.path.join(save_path, "best.pth"))

        if opt.save_interval > 0 and (epoch + 1) % opt.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(save_path, "last.pth"))

    torch.save(model.state_dict(), os.path.join(save_path, "last.pth"))
    writer.close()


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for sample in val_loader:
            for k in sample:
                sample[k] = sample[k].to(device=device, non_blocking=True)
            pred = model(sample["input"])
            loss, _ = criterion(pred, sample)
            total_loss += loss.item()
    return total_loss / max(len(val_loader), 1)


if __name__ == "__main__":
    train(parse_opt())
