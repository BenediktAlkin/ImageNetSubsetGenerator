import argparse

from imagenet_subset_generator import generate_dummy_dataset, VERSIONS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", required=True, help="path to the directory where dummy should be stored")
    parser.add_argument("--train_size", type=int, default=13, help="size of the train set (per class)")
    parser.add_argument("--valid_size", type=int, default=1, help="size of the valid set (per class)")
    parser.add_argument("--resolution_min", type=int, default=16, help="minimum resolution of the image")
    parser.add_argument("--resolution_max", type=int, default=16, help="maximum resolution of the image")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--version", choices=VERSIONS, help="use predefined classes from a popular version")
    grp.add_argument("--n_classes", type=int, help="use the first n_classes from ImageNet1K")
    return parser.parse_args()


if __name__ == "__main__":
    generate_dummy_dataset(**parse_args().__dict__)
