import numpy as np
import os
import shutil
from pathlib import Path

from .util import parse_version

def generate_subset(in1k_path, out_path, version, log=print):
    in1k_path = Path(in1k_path).expanduser()
    out_path = Path(out_path).expanduser()
    assert in1k_path.exists(), f"invalid path to ImageNet1K: {in1k_path}"
    nonempty_outpath_msg = f"out_path already exists and is not empty: {out_path}"
    assert not out_path.exists() or len(os.listdir(out_path)) == 0, nonempty_outpath_msg
    log(f"generating subset of {in1k_path} in {out_path}")

    # get classes and info
    classes, files, info = parse_version(version=version, log=log)
    # max number of digits
    lpad = len(str(len(classes))) if classes is not None else 0

    # link folders
    out_path.mkdir(exist_ok=True)
    if classes is not None:
        for split in ["train", "val"]:
            i = 0
            log(f"processing {split}")
            split_path = in1k_path / split
            split_out_path = out_path / split
            split_out_path.mkdir(exist_ok=True, parents=True)
            assert len(os.listdir(split_out_path)) == 0, f"{split_out_path} is not empty"

            for folder in os.listdir(split_path):
                if folder in classes:
                    i += 1
                    istr = str(i).zfill(lpad)
                    log(f"linking {split}/{folder} ({istr}/{len(classes)})")
                    os.symlink(split_path / folder, split_out_path / folder, target_is_directory=True)

    # link files (only makes sense for train split)
    if files is not None:
        split_path = in1k_path / "train"
        split_out_path = out_path / "train"
        split_out_path.mkdir(exist_ok=True, parents=True)
        assert len(os.listdir(split_out_path)) == 0, f"{split_out_path} is not empty"
        for i, file in enumerate(files):
            folder = file.split("_")[0]
            src_file = split_path / folder / file
            assert src_file.exists()
            log(f"linking {file}")
            dst_folder = split_out_path / folder
            dst_folder.mkdir(exist_ok=True)
            os.symlink(src_file, dst_folder / file, target_is_directory=False)

    # link metafile
    meta_path = in1k_path / "meta.bin"
    if meta_path.exists():
        log("linking meta.bin")
        os.symlink(in1k_path / "meta.bin", out_path / "meta.bin", target_is_directory=False)
    else:
        log(f"no meta.bin file found")

    # write readme
    log("creating README.txt")
    with open(out_path / "README.txt", "w") as f:
        f.write("\n".join(info))
    log("created README.txt")

    #
    log(f"created subset '{out_path}' ({version})")
