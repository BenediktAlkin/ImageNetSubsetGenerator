import importlib
import itertools
import os
import numpy as np
from pathlib import Path


def n_files_in_directory(root):
    return sum([len(files) for _, _, files in os.walk(root)])

def file_names_in_directory(root):
    return list(itertools.chain(*[files for _, _, files in os.walk(root)]))

def n_folders_in_directory(root):
    return sum([len(dirs) for _, dirs, _ in os.walk(root)])

def folder_names_in_directory(root):
    return list(itertools.chain(*[dirs for _, dirs, _ in os.walk(root)]))

def get_versions():
    path = Path(__file__).parent
    versions = os.listdir(f"{path}/versions")
    versions.remove("__init__.py")
    if "__pycache__" in versions:
        versions.remove("__pycache__")
    versions = list(map(lambda v: v[:-3], versions))
    return versions


VERSIONS = get_versions()

def get_classes_and_info(classes=None, version=None, n_classes=None, use_in1k_as_default=False, log=print):
    """
    classes is not None -> check if classes are all strings
    version is not None -> check version and return classes from version
    n_classes is not None -> return n_classes random classes from ImageNet1K
    """
    if sum([classes is not None, version is not None, n_classes is not None]) != 1:
        if use_in1k_as_default:
            from .versions.in1k import CLASSES, INFO
            return CLASSES, INFO
        else:
            raise AssertionError("define classes, version or n_classes but not multiple of them")

    if classes is not None:
        invalid_classes_msg = "expected list of strings as classes parameter"
        assert isinstance(classes, list) and all(map(lambda c: isinstance(c, str), classes)), invalid_classes_msg
        log(f"ImageNet100-{version}")
        log(f"classes: {classes}")
        return classes, None

    if version is not None:
        assert version in VERSIONS, f"invalid version '{version}' use one of {VERSIONS}"
        version_module = importlib.import_module(f".versions.{version}", package="imagenet_subset_generator")
        log(f"{version}")
        log(f"classes: {version_module.CLASSES}")
        # sanity check to avoid duplicates
        assert len(np.unique(version_module.CLASSES)) == len(version_module.CLASSES)
        return version_module.CLASSES, version_module.INFO

    if n_classes is not None:
        assert isinstance(n_classes, int) and n_classes >= 1, "n_classes needs to be int and >= 1"
        assert n_classes < 1000, "n_classes needs to be < 1000"
        from .versions.in1k import CLASSES, INFO
        log(f"subset of ImageNet1K with the first {n_classes} classes")
        log(f"classes: {CLASSES}")
        return CLASSES[:n_classes], [
            f"subset with the first {n_classes} classes of the original ImageNet1K dataset",
            "https://image-net.org/",
        ]

    raise RuntimeError
