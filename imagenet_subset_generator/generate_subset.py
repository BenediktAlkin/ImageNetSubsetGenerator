import numpy as np
import os
import shutil
from pathlib import Path

import h5py

from .util import get_classes_and_info


def generate_subset(
        in1k_path,
        out_path,
        version=None,
        classes=None,
        n_classes=None,
        train_fraction_from=None,
        train_fraction_to=None,
        train_fraction_seed=None,
        h5=False,
        h5_compression=None,
        log=print,
):
    assert train_fraction_from is None or 0. <= train_fraction_from < 1.
    assert train_fraction_to is None or 0. < train_fraction_to <= 1.
    train_fraction_from = train_fraction_from or 0.
    train_fraction_to = train_fraction_to or 1.
    assert train_fraction_from < train_fraction_to
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

    # rng
    rng = np.random.default_rng(seed=train_fraction_seed) if train_fraction_seed is not None else None

    # generate
    out_path.mkdir(exist_ok=True)
    logstr = "collecting fileinfos from" if h5 else "copying"
    for split in ["train", "val"]:
        i = 0

        filelist = {}

        log(f"{logstr} {split}")
        split_path = in1k_path / split
        split_out_path = out_path / split
        if h5:
            assert not (out_path / f"{split}.h5").exists(), f"{split}.h5 already exists"
        else:
            split_out_path.mkdir(exist_ok=True, parents=True)
            assert len(os.listdir(split_out_path)) == 0, f"{split_out_path} is not empty"

        for folder in os.listdir(split_path):
            if split == "val" or (train_fraction_from == 0. and train_fraction_to == 1.):
                # copy full folders
                if folder in classes:
                    i += 1
                    istr = str(i).zfill(lpad)
                    log(f"{logstr} {split}/{folder} ({istr}/{len(classes)})")
                    if not h5:
                        shutil.copytree(split_path / folder, split_out_path / folder)
                    filelist[folder] = list(sorted(os.listdir(split_path / folder)))
            else:
                if folder not in classes:
                    continue
                i += 1
                istr = str(i).zfill(lpad)
                # make folder
                if not h5:
                    (split_out_path / folder).mkdir(exist_ok=True, parents=True)
                # filter out images
                images = list(sorted([fname for fname in os.listdir(split_path / folder) if fname.endswith(".JPEG")]))
                # copy a fraction of the images (currently take only the first <train_fraction>%
                start_idx = int(len(images) * train_fraction_from)
                end_idx = int(len(images) * train_fraction_to)
                if rng is None:
                    cur_images = images[start_idx:end_idx]
                    log(
                        f"{logstr} images with indices in [{start_idx},{end_idx}) from {split}/{folder} "
                        f"({istr}/{len(classes)})"
                    )
                else:
                    all_indices = list(range(len(images)))
                    rng.shuffle(all_indices)
                    indices = all_indices[start_idx:end_idx]
                    cur_images = [images[idx] for idx in indices]
                    log(
                        f"{logstr} images from {split}/{folder} ({istr}/{len(classes)}) "
                        f"with random indices {np.array2string(indices)}"
                    )

                filelist[folder] = []
                for image in cur_images:
                    if not h5:
                        shutil.copyfile(split_path / folder / image, split_out_path / folder / image)
                    filelist[folder].append(image)

        log(f"finished copying {split}")
        log("writing filelist")
        with open(out_path / f"{split}_filelist.txt", "w") as f:
            classnames = list(sorted(filelist.keys()))
            for i, classname in enumerate(classnames):
                for filename in filelist[classname]:
                    f.write(f"{classname} {filename} {i}\n")
        if h5:
            log(f"generating h5")
            with h5py.File(out_path / f"{split}.h5", "w") as f:
                for i, classname in enumerate(classnames):
                    classgroup = f.create_group(classname)
                    for filename in filelist[classname]:
                        with open(split_path / classname / filename, "rb") as img:
                            img_data_binary = img.read()
                        img_data = np.frombuffer(img_data_binary, dtype="uint8")
                        if h5_compression is None:
                            log(f"no h5_compression defined...using highest compression by default (9)")
                            h5_compression = 9
                        else:
                            assert 0 <= h5_compression <= 9
                        classgroup.create_dataset(
                            filename,
                            data=img_data,
                            shape=img_data.shape,
                            compression="gzip",
                            compression_opts=h5_compression,
                        )

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
        if train_fraction_from != 0. or train_fraction_to != 1.0:
            info.append(f"using only {(train_fraction_to-train_fraction_from)*100}% of the training data")
            info.append(f"samples are taken from indices of {train_fraction_from*100}% to {train_fraction_to*100}%")
        log("creating README.txt")
        with open(out_path / "README.txt", "w") as f:
            f.write("\n".join(info))
        log("created README.txt")
