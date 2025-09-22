#!/usr/bin/env python3
"""
augment_breed_cli.py
Usage:
  python augment_breed_cli.py --src images/nili_ravi --dst Augmented_nili_ravi --target 400
"""
import os, cv2, shutil, argparse, random, time
from tqdm import tqdm
import albumentations as A

def make_transform():
    return A.Compose([
        A.RandomResizedCrop(height=320, width=320, scale=(0.6, 1.0), p=0.9),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.28, contrast_limit=0.28, p=0.6),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.35),
        A.MotionBlur(blur_limit=5, p=0.25),
        A.Cutout(num_holes=1, max_h_size=40, max_w_size=40, p=0.3),
        A.HueSaturationValue(p=0.25)
    ])

def safe_list_images(folder):
    exts = ('.jpg','.jpeg','.png','.bmp')
    return [f for f in os.listdir(folder) if f.lower().endswith(exts)]

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def copy_originals_to_dst(src, dst):
    images = safe_list_images(src)
    for f in images:
        srcp = os.path.join(src, f)
        dstp = os.path.join(dst, f)
        if not os.path.exists(dstp):
            shutil.copy(srcp, dstp)

def augment_until_target(src, dst, target, seed=42):
    random.seed(seed)
    transform = make_transform()
    images = safe_list_images(src)
    if len(images) == 0:
        print("No images found in source:", src)
        return
    existing = safe_list_images(dst)
    existing_count = len(existing)
    to_generate = max(0, target - existing_count)
    print(f"Destination {dst} already has {existing_count} images. Need to generate {to_generate} more to reach {target}.")

    idx = 0
    i = 0
    pbar = tqdm(total=to_generate)
    while i < to_generate:
        img_name = images[idx % len(images)]
        img_path = os.path.join(src, img_name)
        img = cv2.imread(img_path)
        if img is None:
            idx += 1
            continue
        try:
            aug = transform(image=img)['image']
        except Exception as e:
            # fallback: write the original
            aug = img
        out_name = f"{os.path.splitext(img_name)[0]}_aug_{int(time.time()*1000)%100000}_{i}.jpg"
        outp = os.path.join(dst, out_name)
        cv2.imwrite(outp, aug)
        i += 1
        idx += 1
        pbar.update(1)
    pbar.close()
    final_count = len(safe_list_images(dst))
    print(f"Augmentation done. Final count in {dst}: {final_count}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Source folder with original images (do NOT include holdout/test here).")
    p.add_argument("--dst", required=False, help="Destination augmented folder (default: Augmented_<basename_of_src>)")
    p.add_argument("--target", type=int, default=400, help="Target total images in destination (originals + augmented).")
    p.add_argument("--copy-originals", action='store_true', help="If set, copy originals into dst before augmenting.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    src = args.src
    if not os.path.isdir(src):
        print("Source folder not found:", src)
        return
    if args.dst:
        dst = args.dst
    else:
        base = os.path.basename(src.rstrip("/"))
        dst = f"Augmented_{base}"
    ensure_dir(dst)

    if args.copy_originals:
        print("Copying originals to destination...")
        copy_originals_to_dst(src, dst)

    # If user didn't copy originals, we still want originals in dst to preserve originals
    # Copy any missing originals so dst contains originals + augmented
    copy_originals_to_dst(src, dst)

    augment_until_target(src, dst, args.target, seed=args.seed)

if __name__ == "__main__":
    main()
