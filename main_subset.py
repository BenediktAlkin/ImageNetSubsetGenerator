import argparse

from imagenet_subset_generator import generate_subset, VERSIONS, MODES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in1k_path", required=True, help="path to the directory where imagenet is stored")
    parser.add_argument("--out_path", required=True, help="path to the directory where imagenet is stored")
    parser.add_argument("--mode", type=str, choices=MODES, required=True)
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--version", choices=VERSIONS, help="use predefined classes from a popular version")
    grp.add_argument("--n_classes", type=int, help="use the first n_classes from ImageNet1K")
    parser.add_argument("--train_fraction_from", type=float, default=None, help="use only a fraction of the data")
    parser.add_argument("--train_fraction_to", type=float, default=None, help="use only a fraction of the data")
    parser.add_argument(
        "--train_fraction_seed",
        type=int,
        default=None,
        help="seed to shuffle before selecting the samples for train_fraction",
    )
    parser.add_argument(
        "--max_train_samples_per_class", type=int, default=None,
        help="upper limit on number of train samples per class"
    )
    parser.add_argument("--h5_compression", type=int, help="degree of compression for h5 file (0 lowest - 9 highest)")
    return parser.parse_args()


if __name__ == "__main__":
    generate_subset(**parse_args().__dict__)
