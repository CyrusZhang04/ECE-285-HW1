"""Preprocess the Kaggle chest X-ray (pneumonia) dataset into a compact uint8 cache.

Each image is converted to grayscale, center-cropped to a square and resized to
IMG_SIZE x IMG_SIZE. The result is saved as a .npz file per split so training can
keep the whole dataset on the GPU.
"""
import argparse
import os
from glob import glob

import numpy as np
from PIL import Image
from tqdm import tqdm


def load_square(path, size):
    img = Image.open(path)
    img.draft("L", (size * 2, size * 2))  # fast JPEG downscale while decoding
    img = img.convert("L")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    img = img.resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="folder containing train/ val/ test/")
    p.add_argument("--out", default="../data_cache")
    p.add_argument("--size", type=int, default=128)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for split in ["train", "val", "test"]:
        files, labels = [], []
        for label, cls in enumerate(["NORMAL", "PNEUMONIA"]):
            fs = sorted(glob(os.path.join(args.root, split, cls, "*.jpeg")))
            files += fs
            labels += [label] * len(fs)
        imgs = np.stack([load_square(f, args.size) for f in tqdm(files, desc=split)])
        out = os.path.join(args.out, f"{split}_{args.size}.npz")
        np.savez_compressed(out, images=imgs, labels=np.array(labels, dtype=np.int64))
        print(f"{split}: {imgs.shape} -> {out}  (normal={labels.count(0)}, pneumonia={labels.count(1)})")


if __name__ == "__main__":
    main()
