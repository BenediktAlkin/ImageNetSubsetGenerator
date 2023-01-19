import importlib
import itertools
import os
import numpy as np
from pathlib import Path
from collections import defaultdict

def n_files_in_directory(root):
    return sum([len(files) for _, _, files in os.walk(root)])

def n_folders_in_directory(root):
    return sum([len(dirs) for _, dirs, _ in os.walk(root)])

def folder_names_in_directory(root):
    return list(itertools.chain(*[dirs for _, dirs, _ in os.walk(root)]))

def n_files_in_subdirectories(root):
    results = defaultdict(int)
    for subdir in os.listdir(root):
        subdir_path = root / subdir
        if not subdir_path.is_dir():
            continue
        for item in os.listdir(subdir_path):
            assert not (subdir_path / item).is_dir()
            results[subdir] += 1
    return results

def get_versions():
    path = Path(__file__).parent
    versions = os.listdir(f"{path}/versions")
    versions.remove("__init__.py")
    if "__pycache__" in versions:
        versions.remove("__pycache__")
    versions = list(map(lambda v: v[:-3], versions))
    return versions


VERSIONS = get_versions()

def parse_version(version, log=print):
    assert version in VERSIONS, f"invalid version '{version}' use one of {VERSIONS}"
    version_module = importlib.import_module(f".versions.{version}", package="imagenet_subset_generator")
    log(f"{version}")
    classes = files = None
    if hasattr(version_module, "CLASSES"):
        # sanity check to avoid duplicates
        assert len(np.unique(version_module.CLASSES)) == len(version_module.CLASSES)
        classes = version_module.CLASSES
    if hasattr(version_module, "FILES"):
        # sanity check to avoid duplicates
        assert len(np.unique(version_module.FILES)) == len(version_module.FILES)
        files = version_module.FILES
    return classes, files, version_module.INFO