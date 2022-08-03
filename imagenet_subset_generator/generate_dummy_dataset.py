import os
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image

from .util import get_classes_and_info


def generate_dummy_dataset(
        out_path, train_size, valid_size, resolution_min, resolution_max,
        classes=None, version=None, n_classes=None, log=print
):
    # check parameters
    assert isinstance(train_size, int) and train_size > 0
    assert isinstance(valid_size, int) and valid_size > 0
    assert isinstance(resolution_min, int) and resolution_min > 1
    assert isinstance(resolution_max, int) and resolution_max >= resolution_min
    out_path = Path(out_path).expanduser()
    assert out_path.exists(), f"out_path {out_path} doesn't exist"
    assert len(os.listdir(out_path)) == 0, f"out_path {out_path} is not empty"

    # get classes and info
    classes, info = get_classes_and_info(
        version=version,
        classes=classes,
        n_classes=n_classes,
        use_in1k_as_default=True,
        log=log,
    )

    # log parameters
    log(f"generating dummy imagenet in: {out_path}")
    log(f"train_size: {train_size} * {len(classes)} = {train_size * len(classes)}")
    log(f"valid_size: {valid_size} * {len(classes)} = {valid_size * len(classes)}")
    log(f"resolution in [{resolution_min};{resolution_max}]")
    total_size = (train_size + valid_size) * len(classes)
    log(f"train+test size: {total_size}")

    # expected size
    average_resolution = (resolution_max + resolution_min) / 2
    expected_size = average_resolution ** 2 * 3 * total_size
    unit_names = ["bytes", "KB", "MB", "GB", "TB"]
    unit_name_idx = 0
    while expected_size >= 1024 and unit_name_idx < len(unit_names) - 1:
        expected_size /= 1024
        unit_name_idx += 1
    log(f"expected total size: {expected_size:.2f} {unit_names[unit_name_idx]}")

    # create subdirs
    train_path = out_path / "train"
    train_path.mkdir()
    valid_path = out_path / "val"
    valid_path.mkdir()

    def _create_image():
        if resolution_min == resolution_max:
            height = width = resolution_min
        else:
            height = torch.randint(resolution_min, resolution_max, size=(1,)).item()
            width = torch.randint(resolution_min, resolution_max, size=(1,)).item()
        img_tensor = torch.rand(3, height, width)
        return to_pil_image(img_tensor)

    # create images
    for i, cls in enumerate(classes):
        log(f"creating dummy data for {cls} ({i + 1}/{len(classes)})")
        cls_train_path = train_path / cls
        cls_train_path.mkdir()
        cls_valid_path = valid_path / cls
        cls_valid_path.mkdir()

        # create train images
        for j in range(train_size):
            img = _create_image()
            img.save(cls_train_path / f"{cls}_{j}.JPEG")
        # create valid images
        for j in range(valid_size):
            img = _create_image()
            img.save(cls_valid_path / f"{cls}_{j}.JPEG")
