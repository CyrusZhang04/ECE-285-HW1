"""Train a convolutional VAE on the chest X-ray dataset.

Example:
    python train.py --recon sigma --epochs 300 --out ../runs/sigma
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from model import VAE, vae_loss


def load_split(cache, split, size, device):
    d = np.load(os.path.join(cache, f"{split}_{size}.npz"))
    return torch.from_numpy(d["images"]).float().div(255.0).unsqueeze(1).to(device)


def augment(x, max_rot=5.0, max_shift=0.05, scale=(0.95, 1.05)):
    """Small random rotation / translation / zoom on the GPU (no flips: anatomy is not symmetric)."""
    b = x.shape[0]
    ang = (torch.rand(b, device=x.device) * 2 - 1) * max_rot * torch.pi / 180
    s = torch.empty(b, device=x.device).uniform_(*scale)
    t = (torch.rand(b, 2, device=x.device) * 2 - 1) * max_shift * 2
    cos, sin = torch.cos(ang) / s, torch.sin(ang) / s
    theta = torch.stack([torch.stack([cos, -sin, t[:, 0]], 1), torch.stack([sin, cos, t[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def evaluate(model, data, args):
    model.eval()
    tot = rec = kl = 0.0
    with torch.no_grad():
        for i in range(0, len(data), 256):
            x = data[i:i + 256]
            x_hat, mu, logvar = model(x)
            l, r, k = vae_loss(x, x_hat, mu, logvar, args.recon, args.beta)
            tot += l.item() * len(x); rec += r.item() * len(x); kl += k.item() * len(x)
    n = len(data)
    return tot / n, rec / n, kl / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="../data_cache")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--latent", type=int, default=128)
    p.add_argument("--ch", type=int, default=32)
    p.add_argument("--recon", default="sigma", choices=["sigma", "bce", "mse"])
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--kl_warmup", type=int, default=10, help="epochs of linear KL annealing")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--aug", action="store_true", help="random affine data augmentation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="../runs/sigma")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "samples"), exist_ok=True)

    train = load_split(args.cache, "train", args.size, device)
    test = load_split(args.cache, "test", args.size, device)
    print(f"train {tuple(train.shape)}  test {tuple(test.shape)}")

    model = VAE(args.latent, args.size, args.ch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.2f}M")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = (len(train) + args.bs - 1) // args.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.05)

    fixed_z = torch.randn(64, args.latent, device=device)
    fixed_x = test[torch.randperm(len(test), generator=torch.Generator().manual_seed(0))[:32].to(device)]
    hist = {k: [] for k in ["epoch", "beta", "train_loss", "train_rec", "train_kl",
                            "test_loss", "test_rec", "test_kl", "time"]}
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        beta = args.beta * min(1.0, epoch / max(1, args.kl_warmup))
        perm = torch.randperm(len(train), device=device)
        tot = rec = kl = 0.0
        for i in range(steps_per_epoch):
            x = train[perm[i * args.bs:(i + 1) * args.bs]]
            if args.aug:
                x = augment(x)
            x_hat, mu, logvar = model(x)
            loss, r, k = vae_loss(x, x_hat, mu, logvar, args.recon, beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(x); rec += r.item() * len(x); kl += k.item() * len(x)
        n = len(train)
        # test loss is the full negative ELBO (beta = target beta), comparable across epochs
        te = evaluate(model, test, args)
        for k_, v in zip(hist.keys(), [epoch, beta, tot / n, rec / n, kl / n, *te, time.time() - t0]):
            hist[k_].append(v)
        if epoch % 10 == 0 or epoch == 1:
            print(f"ep {epoch:4d} beta {beta:.2f} | train {tot / n:10.2f} (rec {rec / n:10.2f} kl {kl / n:7.2f})"
                  f" | test {te[0]:10.2f} (rec {te[1]:10.2f} kl {te[2]:7.2f}) | {time.time() - t0:6.0f}s", flush=True)
        if epoch % 50 == 0 or epoch == args.epochs or epoch in (1, 10):
            model.eval()
            with torch.no_grad():
                save_image(model.decode(fixed_z), os.path.join(args.out, "samples", f"sample_{epoch:04d}.png"), nrow=8)
                rec_x = model.decode(model.encode(fixed_x)[0])
                save_image(torch.cat([fixed_x, rec_x]), os.path.join(args.out, "samples", f"recon_{epoch:04d}.png"), nrow=8)

    torch.save({"model": model.state_dict(), "args": vars(args)}, os.path.join(args.out, "vae.pt"))
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(hist, f)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({**vars(args), "n_params": n_params}, f, indent=2)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
