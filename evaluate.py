"""Compute Inception Score and FID of VAE samples with PyTorch-Ignite.

Follows https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/ : an ignite
Engine produces (fake, real) batches, which are resized to 299x299 and fed to the
ignite FID and InceptionScore metrics (InceptionV3 features). Grayscale X-rays are
repeated to 3 channels. The real reference set is the training split.

Example:
    python evaluate.py --runs ../runs/sigma_aug --out ../results
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from ignite.engine import Engine
from ignite.metrics import FID, InceptionScore
from ignite.metrics.gan.utils import InceptionModel
from torchvision.utils import save_image

from model import VAE
from train import load_split


def interpolate(batch):
    # Same role as the blog's PIL-based interpolate(): resize to InceptionV3's 299x299
    # (bilinear), done on the GPU; grayscale -> 3 channels.
    batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
    return batch.clamp(0, 1).repeat(1, 3, 1, 1)


_POOL = {}


def _pool_features(device):
    if device not in _POOL:
        _POOL[device] = InceptionModel(return_features=True, device=device)
    return _POOL[device]


def compute_is_fid(fake_fn, real, n, bs, device, seed=0):
    """fake_fn(i, k) -> k fake images in [0,1]; real: (N,1,H,W) tensor in [0,1]."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(real), generator=g)[:n]
    batches = [idx[i:i + bs] for i in range(0, n, bs)]
    batches = [b for b in batches if len(b) > 1]  # ignite needs batch size > 1

    def evaluation_step(engine, batch_idx):
        with torch.no_grad():
            fake = interpolate(fake_fn(engine.state.iteration, len(batch_idx)).to(device))
            real_b = interpolate(real[batch_idx.to(real.device)].to(device))
            return fake, real_b

    evaluator = Engine(evaluation_step)
    # NOTE: ignite's FID() default uses the 1000-d softmax output of InceptionV3 as features, which gives
    # tiny, uninformative values (pure noise scores ~0.09). We use the standard 2048-d pool features instead.
    FID(num_features=2048, feature_extractor=_pool_features(device), device=device).attach(evaluator, "fid")
    InceptionScore(device=device, output_transform=lambda x: x[0]).attach(evaluator, "is")
    state = evaluator.run(batches, max_epochs=1)
    return {"IS": float(state.metrics["is"]), "FID": float(state.metrics["fid"]), "N": int(sum(map(len, batches)))}


def load_model(run, device):
    ck = torch.load(os.path.join(run, "vae.pt"), map_location=device)
    a = ck["args"]
    m = VAE(a["latent"], a["size"], a["ch"]).to(device)
    m.load_state_dict(ck["model"])
    return m.eval(), a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["../runs/sigma"])
    p.add_argument("--cache", default="../data_cache")
    p.add_argument("--out", default="../results")
    p.add_argument("--bs", type=int, default=100)
    p.add_argument("--temps", type=float, nargs="+", default=[1.0])
    args = p.parse_args()
    device = "cuda"
    os.makedirs(args.out, exist_ok=True)

    train = load_split(args.cache, "train", 128, device)
    test = load_split(args.cache, "test", 128, device)
    n_full, n_test = len(train), len(test)
    results = []

    def log(name, r):
        r = {"name": name, **r}
        print(f"{name:40s} N={r['N']:5d}  IS={r['IS']:.3f}  FID={r['FID']:.2f}", flush=True)
        results.append(r)

    # Reference: real held-out test images vs. real training images.
    log("real test images", compute_is_fid(lambda i, k: test[(i - 1) * args.bs:(i - 1) * args.bs + k],
                                           train, n_test, args.bs, device))
    log("uniform noise (sanity check)", compute_is_fid(
        lambda i, k: torch.rand(k, 1, 128, 128, device=device), train, n_test, args.bs, device))
    log("real train images (IS only, FID=0 by def.)",
        compute_is_fid(lambda i, k: train[(i - 1) * args.bs:(i - 1) * args.bs + k], train, n_full, args.bs, device))

    for run in args.runs:
        model, a = load_model(run, device)
        tag = os.path.basename(os.path.normpath(run))
        # reconstructions of test images (posterior mean)
        with torch.no_grad():
            rec_test = torch.cat([model.decode(model.encode(test[i:i + 256])[0]) for i in range(0, n_test, 256)])
        log(f"{tag}: reconstructions of test", compute_is_fid(
            lambda i, k: rec_test[(i - 1) * args.bs:(i - 1) * args.bs + k], train, n_test, args.bs, device))
        for t in args.temps:
            torch.manual_seed(123)
            fake_fn = lambda i, k, t=t: model.sample(k, device, temperature=t)
            log(f"{tag}: samples T={t}", compute_is_fid(fake_fn, train, n_full, args.bs, device))
            torch.manual_seed(123)
            log(f"{tag}: samples T={t} (N=624)", compute_is_fid(fake_fn, train, n_test, args.bs, device))
            torch.manual_seed(7)
            save_image(model.sample(64, device, temperature=t), os.path.join(args.out, f"{tag}_samples_T{t}.png"),
                       nrow=8, padding=2, pad_value=1.0)
        # reconstructions grid: rows of real / recon
        x = test[torch.randperm(n_test, generator=torch.Generator().manual_seed(1))[:8].to(device)]
        with torch.no_grad():
            r = model.decode(model.encode(x)[0])
        save_image(torch.cat([x, r]), os.path.join(args.out, f"{tag}_recon.png"), nrow=8, padding=2, pad_value=1.0)

    real_grid = train[torch.randperm(n_full, generator=torch.Generator().manual_seed(7))[:64].to(device)]
    save_image(real_grid, os.path.join(args.out, "real_train.png"), nrow=8, padding=2, pad_value=1.0)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
