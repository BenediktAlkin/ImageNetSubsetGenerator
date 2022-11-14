import shutil
import argparse
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in1k_path", required=True, help="path to the directory where imagenet is stored")
    parser.add_argument("--out_path", required=True, help="path to the directory where imagenet is stored")
    parser.add_argument("--filelist", required=True, help="path to the filelist")
    return parser.parse_args()

def main(in1k_path, out_path, filelist):
    in1k_path = Path(in1k_path).expanduser()
    assert in1k_path.exists()
    in1k_train_path = in1k_path / "train"
    assert in1k_train_path.exists()
    out_path = Path(out_path).expanduser()
    assert out_path.exists()
    filelist = Path(filelist)
    assert filelist.exists()

    with open(filelist) as f:
        filelist = yaml.safe_load(f)

    for f in filelist:
        folder = f.split("_")[0]
        src_path = in1k_train_path / folder / f
        dst_path = out_path / folder / f
        print(f"copy {src_path} to {dst_path}")
        shutil.copy(src_path, dst_path)



if __name__ == "__main__":
    main(**vars(parse_args()))
