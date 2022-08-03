from imagenet_subset_generator import generate_subset, VERSIONS
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in1k_path", required=True, help="path to the directory where imagenet is stored")
    parser.add_argument("--out_path", required=True, help="path to the directory where imagenet is stored")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--version", choices=VERSIONS, help="use predefined classes from a popular version")
    grp.add_argument("--n_classes", type=int, help="use the first n_classes from ImageNet1K")
    return parser.parse_args()


if __name__ == "__main__":
    generate_subset(**parse_args().__dict__)