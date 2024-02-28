import argparse

from imagenet_subset_generator import generate_subset, VERSIONS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in1k_path", required=True, help="path to the directory where imagenet is stored")
    parser.add_argument("--out_path", required=True, help="path to the directory where imagenet is stored")
    parser.add_argument("--version", choices=VERSIONS, help="use predefined classes from a popular version")
    parser.add_argument("--mode", choices=["copy", "link"], help="Whether to copy files to the subset or link them")
    return parser.parse_args()


if __name__ == "__main__":
    generate_subset(**parse_args().__dict__)
