#!/usr/bin/env python3
"""
split_train_val.py (safer)
Usage:
  python split_train_val.py --src "/Users/..../Breed data" --dst "/Users/.../Breed data/dataset" --val-ratio 0.2
This script:
 - Only processes folders that look like breed folders (not folders that contain 'test' or 'duplicate')
 - Handles folders named e.g. 'nili_ravi_aug' as breed folders (it will use the folder name as the class label)
 - Does NOT touch existing test folders (you said you already created them)
"""
import os, argparse, random, shutil

def is_image_file(f):
    return f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.jfif'))

def should_skip_folder(folder_name):
    lower = folder_name.lower()
    if 'test' in lower or 'duplicate' in lower or 'duplicates' in lower or lower.startswith('.'):
        return True
    return False

def split_dataset(src_root, dst_root, val_ratio=0.2, seed=42):
    random.seed(seed)
    entries = sorted(os.listdir(src_root))
    # Consider only directories that are not to be skipped and contain images
    breed_dirs = []
    for e in entries:
        p = os.path.join(src_root, e)
        if os.path.isdir(p) and not should_skip_folder(e):
            # check if directory contains image files
            files = [f for f in os.listdir(p) if is_image_file(f)]
            if len(files) > 0:
                breed_dirs.append(e)
    if not breed_dirs:
        print("No candidate breed folders found in", src_root)
        return

    print("Found breed folders:", breed_dirs)
    for breed in breed_dirs:
        src = os.path.join(src_root, breed)
        files = [f for f in os.listdir(src) if is_image_file(f)]
        if not files:
            print("  Skipping (no images):", breed)
            continue
        random.shuffle(files)
        split_idx = int(len(files) * (1 - val_ratio))
        train_files = files[:split_idx]
        val_files = files[split_idx:]

        train_dir = os.path.join(dst_root, "train", breed)
        val_dir   = os.path.join(dst_root, "val", breed)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        for f in train_files:
            shutil.copy(os.path.join(src, f), os.path.join(train_dir, f))
        for f in val_files:
            shutil.copy(os.path.join(src, f), os.path.join(val_dir, f))

        print(f"{breed}: {len(train_files)} train, {len(val_files)} val (from {len(files)} total)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source root containing breed folders")
    parser.add_argument("--dst", required=True, help="Destination root for dataset (creates train/ and val/)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction for validation (default 0.2)")
    args = parser.parse_args()

    split_dataset(args.src, args.dst, val_ratio=args.val_ratio)
