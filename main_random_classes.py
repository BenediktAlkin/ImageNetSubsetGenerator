import argparse
import numpy as np
from imagenet_subset_generator.versions.in1k import CLASSES

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_classes", required=True, type=int, help="number of classes to select")
    parser.add_argument("--seed", required=True, type=int, help="seed for random number generator")
    return parser.parse_args()

def main():
    args = parse_args()
    rng = np.random.default_rng(seed=args.seed)
    for cls in rng.choice(CLASSES, args.n_classes, replace=False):
        print(f"\t\"{cls}\",")


if __name__ == "__main__":
    main()