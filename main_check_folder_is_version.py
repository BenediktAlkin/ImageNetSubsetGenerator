import os
from pathlib import Path
import argparse
import torch

from imagenet_subset_generator import VERSIONS, get_classes_and_info


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=VERSIONS, required=True)
    parser.add_argument(
        "--path", type=str, required=True,
        help="path to the directory which should be of the specified version"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    classes, _ = get_classes_and_info(version=args.version)
    path = Path(args.path).expanduser()
    assert path.exists() and path.is_dir()
    root_folders = list(filter(lambda p: (path / p).is_dir(), os.listdir(path)))
    if {"train", "val"} != set(root_folders):
        print(f"root_folder should only contain 'train' and 'val' folder (contains {root_folders})")
        print("ERROR")
        return
    for split in ["train", "val"]:
        class_folders = list(filter(lambda p: (path / split / p).is_dir(), os.listdir(path / split)))
        class_folders = set(sorted(class_folders))
        classes = set(sorted(classes))
        missing_classes = list(classes - class_folders)
        if len(missing_classes) != 0:
            print(f"{len(missing_classes)} classes are missing ({split} split): {missing_classes}")
            print("ERROR")
            return
        unnecessary_classes = list(class_folders - classes)
        if len(unnecessary_classes) != 0:
            print(f"{len(unnecessary_classes)} classes should not be present ({split} split): {unnecessary_classes}")
            print("ERROR")
            return
    print(f"{path} contains all classes of {args.version}")
    print("SUCCESS")


if __name__ == "__main__":
    main()
