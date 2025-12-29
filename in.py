import argparse
import math
import os
import random
import sys
from typing import List, Optional, Sequence, Tuple

from PIL import Image


def find_split_root(root_dir: str, split: Optional[str]) -> str:
    if split:
        candidate = os.path.join(root_dir, split)
        if os.path.isdir(candidate):
            return candidate
    return root_dir


def try_import_imagefolder():
    try:
        from torchvision.datasets import ImageFolder  # type: ignore

        return ImageFolder
    except Exception:
        return None


def collect_images_manual(root_dir: str) -> Tuple[List[str], List[str]]:
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    class_names: List[str] = []
    image_paths: List[str] = []

    # Expect ImageNet-style: root_dir/class_name/*.jpg
    for entry in sorted(os.scandir(root_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        class_dir = entry.path
        class_name = entry.name
        any_found = False
        for dirpath, _, filenames in os.walk(class_dir):
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext in valid_exts:
                    image_paths.append(os.path.join(dirpath, fname))
                    any_found = True
        if any_found:
            class_names.append(class_name)

    return image_paths, class_names


def load_samples(
    root_dir: str,
    split: Optional[str],
    num_images: int,
    shuffle: bool,
    seed: Optional[int],
) -> Tuple[List[Tuple[Image.Image, str]], List[str]]:
    split_root = find_split_root(root_dir, split)

    ImageFolder = try_import_imagefolder()
    samples: List[Tuple[Image.Image, str]] = []
    classes: List[str] = []

    if ImageFolder is not None and os.path.isdir(split_root):
        dataset = ImageFolder(split_root)
        classes = list(dataset.classes)
        indices: Sequence[int]
        total = len(dataset)
        if total == 0:
            return [], classes

        if shuffle:
            rng = random.Random(seed)
            indices = rng.sample(range(total), k=min(num_images, total))
        else:
            indices = list(range(min(num_images, total)))

        for idx in indices:
            img, label_idx = dataset[idx]
            label = (
                classes[label_idx] if 0 <= label_idx < len(classes) else str(label_idx)
            )
            samples.append((img, label))
        return samples, classes

    # Fallback manual collection if torchvision is unavailable
    if not os.path.isdir(split_root):
        return [], classes

    image_paths, class_names = collect_images_manual(split_root)
    if not image_paths:
        return [], class_names
    classes = class_names

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(image_paths)
    selected_paths = image_paths[: min(num_images, len(image_paths))]

    def label_from_path(p: str) -> str:
        parent = os.path.basename(os.path.dirname(p))
        return parent

    for p in selected_paths:
        try:
            img = Image.open(p).convert("RGB")
            samples.append((img, label_from_path(p)))
        except Exception:
            continue

    return samples, classes


def show_grid(
    samples: List[Tuple[Image.Image, str]],
    cols: int,
    title: Optional[str],
    save_path: Optional[str],
    show: bool,
) -> None:
    import matplotlib.pyplot as plt  # Local import to avoid hard dependency

    n = len(samples)
    if n == 0:
        print("No images to display.", file=sys.stderr)
        return

    cols = max(1, cols)
    rows = math.ceil(n / cols)
    # Reasonable size: ~2 inches per cell on width
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.8))
    if title:
        fig.suptitle(title)

    if rows == 1 and cols == 1:
        axes = [[axes]]  # type: ignore[assignment]
    elif rows == 1:
        axes = [axes]  # type: ignore[assignment]
    elif cols == 1:
        axes = [[ax] for ax in axes]  # type: ignore[misc]

    flat_axes = [ax for row_axes in axes for ax in row_axes]  # type: ignore[union-attr]
    for i, ax in enumerate(flat_axes):
        if i < n:
            img, label = samples[i]
            ax.imshow(img)
            ax.set_title(label, fontsize=8)
        ax.axis("off")

    plt.tight_layout(pad=0.3)

    if save_path:
        try:
            fig.savefig(save_path, dpi=180, bbox_inches="tight")
            print(f"Saved grid to: {save_path}")
        except Exception as e:
            print(f"Failed to save figure: {e}", file=sys.stderr)

    if show:
        try:
            plt.show()
        except Exception as e:
            print(
                f"Display failed (headless environment?). Error: {e}", file=sys.stderr
            )
    else:
        plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display a grid of ImageNet images (100 by default). "
        "Expects ImageNet-style folder layout: root/[train|val]/<class_name>/*.jpg"
    )
    default_root = os.environ.get("IMAGENET_DIR", "")
    parser.add_argument(
        "--root",
        type=str,
        default=default_root,
        help="Root directory of the dataset (reads IMAGENET_DIR if unset).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "none"],
        default="val",
        help="Subfolder split to use. Use 'none' if root directly contains class folders.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=100,
        help="Number of images to display.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=10,
        help="Number of columns in the grid.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for shuffling/sample selection.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling; take the first N images.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save the grid image (e.g., grid.png).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a GUI window; useful in headless environments.",
    )
    args = parser.parse_args(argv)
    if args.split == "none":
        args.split = None
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root_dir = args.root
    if not root_dir:
        print(
            "Please provide --root or set IMAGENET_DIR to your ImageNet root.",
            file=sys.stderr,
        )
        return 1
    if not os.path.isdir(root_dir):
        print(f"Root directory not found: {root_dir}", file=sys.stderr)
        return 1

    samples, classes = load_samples(
        root_dir=root_dir,
        split=args.split,
        num_images=args.num,
        shuffle=not args.no_shuffle,
        seed=args.seed,
    )
    if not samples:
        split_str = args.split if args.split else "(none)"
        print(
            f"No images found under: {find_split_root(root_dir, args.split)} (split={split_str})",
            file=sys.stderr,
        )
        return 2

    title = f"ImageNet samples ({len(samples)})"
    show_grid(
        samples=samples,
        cols=args.cols,
        title=title,
        save_path=args.save,
        show=not args.no_show,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
