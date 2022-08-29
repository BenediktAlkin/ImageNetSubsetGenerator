import argparse
import numpy as np
from imagenet_subset_generator.versions.in1k import CLASSES as IN1K_CLASSES
from imagenet_subset_generator.versions.in100_sololearn import CLASSES as SOLO_CLASSES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int, help="seed for random number generator")
    return parser.parse_args()

def main():
    args = parse_args()
    rng = np.random.default_rng(seed=args.seed)
    remaining_classes = [cls for cls in IN1K_CLASSES if cls not in SOLO_CLASSES]
    rng.shuffle(remaining_classes)
    for i in range(9):
        print(f"in100_sololearn_cv{i+1}")
        print("INFO = [")
        print(f"\t\"'cross validation split {i+1}' with in100_sololearn as base split\"")
        print(f"\t\"generated with python main_solocrossvalid_classes --seed {args.seed}\"")
        print("]")
        classes = remaining_classes[:100]
        remaining_classes = remaining_classes[100:]
        print("CLASSES = [")
        for cls in classes:
            print(f"\t\"{cls}\",")
        print("]")


if __name__ == "__main__":
    main()