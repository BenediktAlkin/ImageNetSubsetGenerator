# imagenet100_sololearn: 126689 samples
# python main_imagenet_filenames.py --end_index 126 --version imagenet100_sololearn
# imagenet1k: 1281167 samples
# python main_imagenet_filenames.py --end_index 128 --version imagenet1k
from argparse import ArgumentParser
from torchvision.datasets import ImageFolder
from pathlib import Path
from collections import defaultdict
import numpy as np

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--end_index", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return vars(parser.parse_args())

def main(root, end_index, seed):
    root = Path(root).expanduser()
    assert root.exists()

    class_to_fnames = defaultdict(list)
    ds = ImageFolder(root=root)
    print(f"sampling {end_index} samples from {root} (len={len(ds)}) with seed={seed}")
    fname_class_tuple = ds.imgs
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(fname_class_tuple)
    for (fname, cls) in fname_class_tuple:
        if len(class_to_fnames[cls]) < end_index:
            class_to_fnames[cls].append(fname)

    for cls, fnames in class_to_fnames.items():
        for fname in fnames:
            print(f"{fname}")



if __name__ == "__main__":
    main(**parse_args())