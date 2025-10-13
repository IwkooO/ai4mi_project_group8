#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec, Caroline Magg

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import warnings
from typing import Any
from pathlib import Path
from pprint import pprint
from operator import itemgetter
from shutil import copytree, rmtree

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
import wandb
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))
from functools import partial 

from dataset import SliceDataset
from ShallowNet import shallowCNN
from ENet import ENet
from TransUNet import TransUNet2D
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   save_images)

from losses import (CrossEntropy)

datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the classes with C (often used for the number of Channel)
datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR"] = {'K': 5, 'net': TransUNet2D, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_CLEAN"] = {'K': 5, 'net': TransUNet2D, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_PREPROCESSED"] = datasets_params["SEGTHOR_CLEAN"]

def img_transform(img):
        img = img.convert('L')
        img = np.array(img)[np.newaxis, ...]
        img = img / 255  # max <= 1
        img = torch.tensor(img, dtype=torch.float32)
        return img

def gt_transform(K, img):
        img = np.array(img)[...]
        # The idea is that the classes are mapped to {0, 255} for binary cases
        # {0, 85, 170, 255} for 4 classes
        # {0, 51, 102, 153, 204, 255} for 6 classes
        # Very sketchy but that works here and that simplifies visualization
        img = img / (255 / (K - 1)) if K != 5 else img / 63  # max <= 1
        img = torch.tensor(img, dtype=torch.int64)[None, ...]  # Add one dimension to simulate batch
        img = class2one_hot(img, K=K)
        return img[0]

def worker_init_fn(worker_id):
    """Initialize worker with deterministic seed for reproducibility"""
    np.random.seed(torch.initial_seed() % 2**32)

def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Networks and scheduler
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")
    print(f">> Using random seed: {args.seed}")

    K: int = datasets_params[args.dataset]['K']
    kernels = None
    factor = None
    # Only pass supported arguments to the network constructor
    if datasets_params[args.dataset]['net'] is TransUNet2D:
        # net = TransUNet2D(
        #     in_ch=1,
        #     num_classes=5,
        #     base_ch=32,
        #     vit_embed_dim=128,
        #     vit_depth=4,
        #     vit_heads=8,
        #     vit_mlp_ratio=2.0,
        #     vit_dropout=0.1
        # )
        net = TransUNet2D(
            in_ch=1,
            num_classes=5,
            base_ch=16,
            vit_embed_dim=128,
            vit_depth=2,
            vit_heads=2,
            vit_mlp_ratio=2.0,
            vit_dropout=0.1
        )
    else:
        kernels = datasets_params[args.dataset]['kernels'] if 'kernels' in datasets_params[args.dataset] else 8
        factor = datasets_params[args.dataset]['factor'] if 'factor' in datasets_params[args.dataset] else 2
        net = datasets_params[args.dataset]['net'](1, K, kernels=kernels, factor=factor)
    if hasattr(net, 'init_weights'):
        net.init_weights()
    net.to(device)

    lr = 0.0005
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
    # optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    # optimizer = Ranger(
    #     net.parameters(),
    #     lr=lr,
    #     weight_decay=0.01,   # tune 0.0–0.01
    # )
    
    # Dataset part
    B: int = datasets_params[args.dataset]['B']
    root_dir = Path("data") / args.dataset

    train_set = SliceDataset('train',
                             root_dir,
                             img_transform=img_transform,
                             gt_transform= partial(gt_transform, K),
                             debug=args.debug)
    train_loader = DataLoader(train_set,
                              batch_size=B,
                              num_workers=5,
                              shuffle=True,
                              worker_init_fn=worker_init_fn,
                              generator=torch.Generator().manual_seed(args.seed))

    val_set = SliceDataset('val',
                           root_dir,
                           img_transform=img_transform,
                           gt_transform=partial(gt_transform, K),
                           debug=args.debug)
    val_loader = DataLoader(val_set,
                            batch_size=B,
                            num_workers=5,
                            shuffle=False,
                            worker_init_fn=worker_init_fn,
                            generator=torch.Generator().manual_seed(args.seed))

    args.dest.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    run_name = args.run_name if args.run_name is not None else f"{args.dataset}_{args.mode}_seed{args.seed}"
    wandb.init(
        project="segthor-segmentation",
        name=run_name,
        config={
            "epochs": args.epochs,
            "dataset": args.dataset,
            "mode": args.mode,
            "seed": args.seed,
            "gpu": args.gpu,
            "debug": args.debug,
            "learning_rate": lr,
            "batch_size": B,
            "num_classes": K,
            "architecture": datasets_params[args.dataset]['net'].__name__,
            "kernels": kernels,
            "factor": factor
        }
    )

    return (net, optimizer, device, train_loader, val_loader, K)


def runTraining(args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")
    net, optimizer, device, train_loader, val_loader, K = setup(args)

    if args.mode == "full":
        loss_fn = CrossEntropy(idk=list(range(K)))  # Supervise both background and foreground
    elif args.mode in ["partial"] and args.dataset == 'SEGTHOR':
        loss_fn = CrossEntropy(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
    else:
        raise ValueError(args.mode, args.dataset)

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

    best_dice: float = 0

    for e in range(args.epochs):
        for m in ['train', 'val']:
            match m:
                case 'train':
                    net.train()
                    opt = optimizer
                    cm = Dcm
                    desc = f">> Training   ({e: 4d})"
                    loader = train_loader
                    log_loss = log_loss_tra
                    log_dice = log_dice_tra
                case 'val':
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val

            with cm():  # Either dummy context manager, or the torch.no_grad for validation
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data['images'].to(device)
                    gt = data['gts'].to(device)

                    if opt:  # So only for training
                        opt.zero_grad()

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    B, _, W, H = img.shape

                    pred_logits = net(img)
                    pred_probs = F.softmax(1 * pred_logits, dim=1)  # 1 is the temperature parameter

                    # Metrics computation, not used for training
                    pred_seg = probs2one_hot(pred_probs)
                    log_dice[e, j:j + B, :] = dice_coef(pred_seg, gt)  # One DSC value per sample and per class

                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = loss.item()  # One loss value per batch (averaged in the loss)

                    if opt:  # Only for training
                        loss.backward()
                        opt.step()

                    if m == 'val':
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning)
                            predicted_class: Tensor = probs2class(pred_probs)
                            mult: int = 63 if K == 5 else (255 / (K - 1))
                            save_images(predicted_class * mult,
                                        data['stems'],
                                        args.dest / f"iter{e:03d}" / m)

                    j += B  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    postfix_dict: dict[str, str] = {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                                                    "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    if K > 2:
                        postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                                         for k in range(1, K)}
                    tq_iter.set_postfix(postfix_dict)

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)

        current_dice: float = log_dice_val[e, :, 1:].mean().item()
        if current_dice > best_dice:
            message = f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC"
            print(message)
            # wandb logging
            epoch_train_loss = log_loss_tra[e].mean().item()
            epoch_val_loss = log_loss_val[e].mean().item()
            epoch_train_dice = log_dice_tra[e, :, 1:].mean().item()
            epoch_val_dice = log_dice_val[e, :, 1:].mean().item()
            wandb_log = {
                "epoch": e,
                "train/loss": epoch_train_loss,
                "val/loss": epoch_val_loss,
                "train/dice": epoch_train_dice,
                "val/dice": epoch_val_dice,
            }
            if K > 2:
                for k in range(1, K):
                    wandb_log[f"train/dice_class_{k}"] = log_dice_tra[e, :, k].mean().item()
                    wandb_log[f"val/dice_class_{k}"] = log_dice_val[e, :, k].mean().item()
            wandb.log(wandb_log)
            best_dice = current_dice
            with open(args.dest / "best_epoch.txt", 'w') as f:
                f.write(message)

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)
            copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")
            # Save model checkpoint to wandb
            wandb.save(str(args.dest / "bestweights.pt"))

def runInference(net, device, args):
    print(f">>> Setting up to run inference on {args.dataset}")
    net.to(device)
    net.eval()
    K = datasets_params[args.dataset]['K']
    B = datasets_params[args.dataset]['B']
    root_dir = Path("data") / args.dataset

    test_set = SliceDataset(
        'test',
        root_dir,
        img_transform=img_transform,
        gt_transform=lambda x: x,
        debug=args.debug
    )
    test_loader = DataLoader(
        test_set,
        batch_size=B,
        num_workers=5,
        shuffle=False,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(args.seed)
    )

    out_dir = args.dest / "test"
    out_dir.mkdir(parents=True, exist_ok=True)

    mult = 63 if K == 5 else (255 / (K - 1))

    with torch.no_grad():
        tq = tqdm_(test_loader, total=len(test_loader), desc=">> Testing")
        for batch in tq:
            img = batch['images'].to(device)          # [B, 1, H, W]
            stems = batch['stems']                    # list[str]

            logits = net(img)                         # [B, K, H, W]
            probs = torch.softmax(logits, dim=1)
            pred_cls = probs2class(probs)             # [B, 1, H, W] (class ids)
            save_images(pred_cls * mult, stems, out_dir)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--dataset', default='TOY2', choices=datasets_params.keys())
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results (predictions and weights).")
    parser.add_argument('--run_name', type=str, default=None,
                        help="Custom wandb run name. If not set, uses default format.")

    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--seed', type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logics around epochs and logging easily.")
    parser.add_argument('--inference_pkl', type=Path, default=None,
                        help="Path to the .pkl of the model for inference.")

    args = parser.parse_args()

    pprint(args)

    if args.inference_pkl is not None:
        print(f">>> Running inference only, loading model from {args.inference_pkl}")
        assert args.inference_pkl.exists(), args.inference_pkl
        device = torch.device("cuda") if args.gpu and torch.cuda.is_available() else torch.device("cpu")
        net = torch.load(args.inference_pkl, map_location=device, weights_only=False)
        runInference(net, device, args)
    else:
        runTraining(args)


if __name__ == '__main__':
    main()
