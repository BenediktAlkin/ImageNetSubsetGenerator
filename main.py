import in100gen
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in1k_path", required=True, help="path to the directory where imagenet is stored")
    parser.add_argument("--out_path", required=True, help="path to the directory where imagenet is stored")
    parser.add_argument(
        "--version", required=True, choices=in100gen.VERSIONS,
        help="which imagenet100 version to use",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    in100gen.generate(args.in1k_path, args.out_path, version=args.version)



if __name__ == "__main__":
    main()