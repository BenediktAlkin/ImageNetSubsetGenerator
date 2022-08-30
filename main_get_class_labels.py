import argparse
import torch

from imagenet_subset_generator import VERSIONS, get_classes_and_info


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=VERSIONS, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    classes, _ = get_classes_and_info(version=args.version)
    id_to_label, _ = torch.load("res/meta.bin")
    for cls in classes:
        print(f"{cls}: {id_to_label[cls]}")


if __name__ == "__main__":
    main()
