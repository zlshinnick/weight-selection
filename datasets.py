import os
import shutil
import zipfile
from urllib.request import Request, urlopen
from torchvision import datasets, transforms
from torch.utils.data import Subset

from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from timm.data import create_transform

import oxford_flowers_dataset
import oxford_pets_dataset

TINY_IMAGENET_URLS = [
    # Primary
    "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
    # HTTPS fallback (same host)
    "https://cs231n.stanford.edu/tiny-imagenet-200.zip",
]


def _download_file_with_retries(
    urls, dest_path: str, retries: int = 3, chunk_size: int = 1 << 20
) -> None:
    """Download a file robustly with retries and simple validation.

    - Tries each URL in order; for each URL, retries a few times
    - Streams to a temporary .part file and then atomically renames
    - Validates ZIP signature if destination ends with .zip
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_path = dest_path + ".part"

    def _is_valid_zip(path: str) -> bool:
        # Fast check using zipfile header; falls back to zipfile.is_zipfile
        try:
            with open(path, "rb") as f:
                header = f.read(4)
            if header != b"PK\x03\x04":
                return False
        except Exception:
            return False
        return zipfile.is_zipfile(path)

    last_error = None
    for url in urls:
        for attempt in range(1, retries + 1):
            try:
                print(
                    f"Downloading TinyImageNet from {url} (attempt {attempt}/{retries}) ..."
                )
                # Use urlopen to check status and stream
                req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urlopen(req, timeout=60) as resp, open(tmp_path, "wb") as out:
                    if hasattr(resp, "status") and resp.status != 200:
                        raise RuntimeError(f"HTTP {resp.status}")
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        out.write(chunk)

                # Validate zip if applicable
                if dest_path.lower().endswith(".zip"):
                    if not _is_valid_zip(tmp_path):
                        raise zipfile.BadZipFile("Downloaded file is not a valid ZIP")

                # Move into place
                os.replace(tmp_path, dest_path)
                return
            except Exception as e:
                last_error = e
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
                if attempt < retries:
                    continue
        # try next URL
    # If we reached here, all URLs and retries failed
    raise RuntimeError(
        f"Failed to download TinyImageNet archive. Last error: {last_error}"
    )


def _prepare_tiny_imagenet_val(root_dir: str) -> None:
    val_dir = os.path.join(root_dir, "val")
    images_dir = os.path.join(val_dir, "images")
    ann_file = os.path.join(val_dir, "val_annotations.txt")
    prepared_flag = os.path.join(val_dir, ".prepared")

    if not os.path.isdir(val_dir) or not os.path.isfile(ann_file):
        return
    if os.path.exists(prepared_flag):
        return

    # If class subfolders already exist alongside images/, assume prepared
    entries = [
        e
        for e in os.listdir(val_dir)
        if os.path.isdir(os.path.join(val_dir, e)) and e != "images"
    ]
    if entries:
        # Mark as prepared to avoid repeated work
        with open(prepared_flag, "w") as f:
            f.write("ok")
        return

    # Create class folders and move images accordingly
    class_to_files = {}
    with open(ann_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                filename, wnid = parts[0], parts[1]
                class_to_files.setdefault(wnid, []).append(filename)

    for wnid, files in class_to_files.items():
        target_dir = os.path.join(val_dir, wnid)
        os.makedirs(target_dir, exist_ok=True)
        for fname in files:
            src = os.path.join(images_dir, fname)
            dst = os.path.join(target_dir, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                try:
                    shutil.move(src, dst)
                except Exception:
                    # If move fails for any reason, skip
                    pass

    # Optionally remove empty images/ dir
    try:
        if os.path.isdir(images_dir) and not os.listdir(images_dir):
            os.rmdir(images_dir)
    except Exception:
        pass

    with open(prepared_flag, "w") as f:
        f.write("ok")


def _ensure_tiny_imagenet(base_path: str) -> str:
    """
    Ensure TinyImageNet-200 is present under base_path. If missing, download and extract.
    Returns the absolute path to the dataset root containing 'train' and 'val'.
    """
    os.makedirs(base_path, exist_ok=True)

    # Accept either base_path == tiny-imagenet-200, or base_path containing it
    candidate_roots = [
        base_path,
        os.path.join(base_path, "tiny-imagenet-200"),
    ]
    for root in candidate_roots:
        if os.path.isdir(os.path.join(root, "train")) and os.path.isdir(
            os.path.join(root, "val")
        ):
            _prepare_tiny_imagenet_val(root)
            return root

    # Download zip into base_path and extract
    zip_path = os.path.join(base_path, "tiny-imagenet-200.zip")

    def _zip_ok(path: str) -> bool:
        try:
            return zipfile.is_zipfile(path)
        except Exception:
            return False

    # Download if missing or invalid
    if not os.path.isfile(zip_path) or not _zip_ok(zip_path):
        if os.path.isfile(zip_path):
            try:
                os.remove(zip_path)
            except Exception:
                pass
        _download_file_with_retries(TINY_IMAGENET_URLS, zip_path)

    # Try extraction; on BadZipFile, force re-download once
    try:
        print("Extracting TinyImageNet ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(base_path)
    except zipfile.BadZipFile:
        try:
            os.remove(zip_path)
        except Exception:
            pass
        print("Corrupt archive detected. Re-downloading ...")
        _download_file_with_retries(TINY_IMAGENET_URLS, zip_path)
        print("Extracting TinyImageNet ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(base_path)

    dataset_root = os.path.join(base_path, "tiny-imagenet-200")
    _prepare_tiny_imagenet_val(dataset_root)
    return dataset_root


def build_dataset(is_train, args, transform_train=None):
    if not transform_train:
        transform_train = is_train
    transform = build_transform(transform_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == "CIFAR10":
        dataset = datasets.CIFAR10(
            args.data_path, train=is_train, download=True, transform=transform
        )
        nb_classes = 10
    elif args.data_set == "CIFAR100":
        dataset = datasets.CIFAR100(
            args.data_path, train=is_train, download=True, transform=transform
        )
        nb_classes = 100
    elif args.data_set == "IMNET":
        root = os.path.join(args.data_path, "train" if is_train else "val")
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set in ("TINYIMNET", "TINYIMAGENET", "TIN"):
        dataset_root = _ensure_tiny_imagenet(args.data_path)
        root = os.path.join(dataset_root, "train" if is_train else "val")
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 200
    elif args.data_set in ("IMNET100", "IN100"):
        # ImageNet-100 subset: filter ImageFolder by a provided class list (wnids) if available.
        root = os.path.join(args.data_path, "train" if is_train else "val")
        full_dataset = datasets.ImageFolder(root, transform=transform)

        # Determine subset list file
        subset_file = getattr(args, "subset_file", None)
        if subset_file is None:
            for candidate in ("imagenet100.txt", "imagenet-100.txt", "wnids100.txt"):
                candidate_path = os.path.join(args.data_path, candidate)
                if os.path.isfile(candidate_path):
                    subset_file = candidate_path
                    break

        if subset_file is None:
            # No list provided; require the directory to already contain exactly 100 classes
            if len(full_dataset.classes) != 100:
                raise ValueError(
                    f"IMNET100 selected but no subset_file provided and directory has {len(full_dataset.classes)} classes. "
                    f"Provide --subset_file pointing to a list of 100 class folder names."
                )
            dataset = full_dataset
            nb_classes = 100
        else:
            with open(subset_file, "r") as f:
                allowed_classes = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.strip().startswith("#")
                ]
            # Map class folder names (wnids) to indices present in ImageFolder
            class_to_idx = full_dataset.class_to_idx
            missing = [c for c in allowed_classes if c not in class_to_idx]
            if missing:
                raise ValueError(
                    f"Subset file contains classes not found in dataset at {root}: {missing[:5]}"
                    + (" ..." if len(missing) > 5 else "")
                )

            allowed_indices_in_order = [class_to_idx[c] for c in allowed_classes]
            allowed_index_set = set(allowed_indices_in_order)
            # Build mapping from original class index -> new contiguous index following the order in the list
            new_index_map = {
                orig_idx: new_idx
                for new_idx, orig_idx in enumerate(allowed_indices_in_order)
            }

            # Select only samples whose target is in allowed set
            selected_indices = [
                i
                for i, (_, y) in enumerate(full_dataset.samples)
                if y in allowed_index_set
            ]

            # Remap labels to 0..N-1 via target_transform
            def _target_remap(y):
                return new_index_map[y]

            full_dataset.target_transform = _target_remap
            dataset = Subset(full_dataset, selected_indices)
            nb_classes = len(allowed_indices_in_order)
    elif args.data_set in ("FRACTALDB", "FRACTAL"):
        # Expect a directory containing class subfolders (ImageFolder format)
        # Use separate eval root if provided
        root = args.data_path if is_train else (args.eval_data_path or args.data_path)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = len(dataset.classes)
    elif args.data_set == "flowers":
        dataset = oxford_flowers_dataset.Flowers(
            root=args.data_path, train=is_train, download=False, transform=transform
        )
        nb_classes = 102
    elif args.data_set == "pets":
        dataset = oxford_pets_dataset.Pets(
            root=args.data_path, train=is_train, download=True, transform=transform
        )
        nb_classes = 37
    elif args.data_set == "stl10":
        if is_train:
            dataset = datasets.STL10(
                root=args.data_path, split="train", download=True, transform=transform
            )
        else:
            dataset = datasets.STL10(
                root=args.data_path, split="test", download=True, transform=transform
            )
        nb_classes = 10
    elif args.data_set == "food101":
        if is_train:
            dataset = datasets.Food101(
                root=args.data_path, split="train", download=True, transform=transform
            )
        else:
            dataset = datasets.Food101(
                root=args.data_path, split="test", download=True, transform=transform
            )
        nb_classes = 101
    else:
        raise NotImplementedError()
    args.nb_classes = nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = (
        IMAGENET_INCEPTION_MEAN
        if not imagenet_default_mean_and_std
        else IMAGENET_DEFAULT_MEAN
    )
    std = (
        IMAGENET_INCEPTION_STD
        if not imagenet_default_mean_and_std
        else IMAGENET_DEFAULT_STD
    )
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:
            t.append(
                transforms.Resize(
                    (args.input_size, args.input_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
            )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
