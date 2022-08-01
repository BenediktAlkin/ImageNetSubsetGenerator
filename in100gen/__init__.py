import os
from functools import partial
from pathlib import Path

VERSIONS = ["kaggle", "solo-learn"]


def generate(in1k_path, out_path, version=None, classes=None, log_fn=print, verbose=True):
    in1k_path = Path(in1k_path)
    out_path = Path(out_path)
    assert in1k_path.exists(), f"invalid path to ImageNet1K: {in1k_path}"

    # initialize log function
    log = partial(_log, log_fn=log_fn, verbose=verbose)

    # get classes from version
    assert (version is None) ^ (classes is None), "use either a predefined version or define your own classes"
    if version is not None:
        assert version in VERSIONS, f"invalid version '{version}' use one of {VERSIONS}"
        if version == "kaggle":
            from .versions.kaggle import CLASSES, INFO
        elif version == "solo_learn" or version == "solo-learn":
            from .versions.kaggle import CLASSES, INFO
        else:
            raise RuntimeError
        classes = CLASSES
        log(f"generating ImageNet100-{version}")
        log(f"classes: {classes}")
    else:
        invalid_classes_msg = "expected list of strings as classes parameter"
        assert isinstance(classes, list) and all(map(lambda c: isinstanc(c, str), classes)), invalid_classes_msg
        log(f"generating ImageNet100-{version}")
        log(f"classes: {classes}")
        INFO = None
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
    if INFO is not None:
        log("creating README.txt")
        with open(out_path / "README.txt", "w") as f:
            f.writelines(INFO)
        log("created README.txt")


def _log(*args, log_fn=print, verbose=True, **kwargs):
    if verbose:
        log_fn(*args, **kwargs)
