#!/usr/bin/env python3
"""
dedup.py
Usage:
  python dedup.py --src "/path/to/images/nili_ravi_aug"
"""

import os
import argparse
import shutil
from PIL import Image
import imagehash
from tqdm import tqdm

def dedupe_images(src, hash_size=16, threshold=5):
    """
    Remove near-duplicate images from folder using perceptual hashing.
    - hash_size: larger = more sensitive (default 16)
    - threshold: max Hamming distance allowed before two images are considered different
    """
    dup_dir = os.path.join(src, "duplicates")
    os.makedirs(dup_dir, exist_ok=True)

    # Include common extensions (case insensitive)
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.jfif')

    hashes = {}
    kept = 0
    moved = 0

    files = [f for f in os.listdir(src) if f.lower().endswith(exts)]
    for fname in tqdm(files, desc=f"Checking {src}"):
        path = os.path.join(src, fname)
        try:
            img = Image.open(path).convert("RGB")
            h = imagehash.phash(img, hash_size=hash_size)
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            continue

        is_duplicate = False
        for existing_hash in hashes:
            if h - existing_hash <= threshold:
                # Move duplicate
                shutil.move(path, os.path.join(dup_dir, fname))
                print(f"Moved duplicate: {fname} (similar to {hashes[existing_hash]})")
                moved += 1
                is_duplicate = True
                break

        if not is_duplicate:
            hashes[h] = fname
            kept += 1

    print(f"\nFinished {src}: kept {kept}, moved {moved} to {dup_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Path to image folder (e.g. images/nili_ravi_aug)")
    parser.add_argument("--hash-size", type=int, default=16, help="Hash size for pHash (larger = more sensitive)")
    parser.add_argument("--threshold", type=int, default=5, help="Hamming distance threshold for duplicates")
    args = parser.parse_args()

    dedupe_images(args.src, hash_size=args.hash_size, threshold=args.threshold)
