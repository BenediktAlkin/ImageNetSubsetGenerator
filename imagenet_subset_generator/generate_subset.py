import os
import shutil
from pathlib import Path

from .util import get_classes_and_info


def generate_subset(in1k_path, out_path, version=None, classes=None, n_classes=None, log=print):
    in1k_path = Path(in1k_path).expanduser()
    out_path = Path(out_path).expanduser()
    assert in1k_path.exists(), f"invalid path to ImageNet1K: {in1k_path}"
    nonempty_outpath_msg = f"out_path already exists and is not empty: {out_path}"
    assert not out_path.exists() or len(os.listdir(out_path)) == 0, nonempty_outpath_msg

    # get classes and info
    classes, info = get_classes_and_info(
        version=version,
        classes=classes,
        n_classes=n_classes,
        log=log,
    )
    # max number of digits
    lpad = len(str(len(classes)))

    # generate
    for split in ["train", "val"]:
        i = 0
        log(f"copying {split}")
        split_path = in1k_path / split
        split_out_path = out_path / split
        split_out_path.mkdir(exist_ok=True, parents=True)
        assert len(os.listdir(split_out_path)) == 0, f"{split_out_path} is not empty"
        for folder in os.listdir(split_path):
            if folder in classes:
                i += 1
                istr = str(i).zfill(lpad)
                log(f"copying {split}/{folder} ({istr}/{len(classes)})")
                shutil.copytree(split_path / folder, split_out_path / folder)
        log(f"finished copying {split}")

    # copy metafile
    meta_path = in1k_path / "meta.bin"
    if meta_path.exists():
        log("copying meta.bin")
        shutil.copy(in1k_path / "meta.bin", out_path / "meta.bin")
        log("copied meta.bin")
    else:
        log(f"no meta.bin file found --> generated dataset will not be usable with torchvision.dataset.ImageNet")

    # write readme
    if info is not None:
        log("creating README.txt")
        with open(out_path / "README.txt", "w") as f:
            f.write("\n".join(info))
        log("created README.txt")
