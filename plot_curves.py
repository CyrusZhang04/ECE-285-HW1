"""Plot training / test loss curves from history.json files."""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIN_C, TEST_C = "#2a78d6", "#eb6834"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["../runs/sigma_aug"])
    p.add_argument("--titles", nargs="+", default=["VAE"])
    p.add_argument("--out", default="../results/loss_curves.pdf")
    args = p.parse_args()

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    hs = [json.load(open(os.path.join(r, "history.json"))) for r in args.runs]
    fig, axes = plt.subplots(1, 2 * len(hs), figsize=(3.3 * len(hs), 1.6))
    for j, (h, title) in enumerate(zip(hs, args.titles)):
        ax = axes[2 * j]
        ax.plot(h["epoch"], h["train_loss"], color=TRAIN_C, lw=1.2, label="train")
        ax.plot(h["epoch"], h["test_loss"], color=TEST_C, lw=1.2, label="test")
        ax.set_title(f"{title}: negative ELBO", fontsize=8)
        ax.set_xlabel("epoch")
        lo = min(min(h["train_loss"][5:]), min(h["test_loss"][5:]))
        hi = max(max(h["train_loss"][5:]), max(h["test_loss"][5:]))
        pad = 0.05 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(frameon=False, fontsize=7)
        ax = axes[2 * j + 1]
        ax.plot(h["epoch"], h["train_kl"], color=TRAIN_C, lw=1.2, label="train")
        ax.plot(h["epoch"], h["test_kl"], color=TEST_C, lw=1.2, label="test")
        ax.set_title(f"{title}: KL term (nats)", fontsize=8)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
