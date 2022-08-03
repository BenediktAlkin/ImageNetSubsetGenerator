import argparse
from pathlib import Path
from imagenet_subset_generator import n_files_in_directory, n_folders_in_directory, folder_names_in_directory


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to the directory of which statistics should be printed")
    return parser.parse_args()

def main(path):
    path = Path(path).expanduser()
    assert path.exists(), f"path doesn't exist: {path}"
    train_path = path / "train"
    assert train_path.exists(), f"train_path doesn't exist: {train_path}"
    valid_path = path / "val"
    assert valid_path.exists(), f"valid_path doesn't exist: {valid_path}"
    print(f"train n_classes: {n_folders_in_directory(train_path)}")
    print(f"valid n_classes: {n_folders_in_directory(valid_path)}")
    print(f"train n_samples: {n_files_in_directory(train_path)}")
    print(f"valid n_samples: {n_files_in_directory(valid_path)}")
    print(f"train classes: {folder_names_in_directory(train_path)}")
    print(f"valid classes: {folder_names_in_directory(valid_path)}")


if __name__ == "__main__":
    main(**parse_args().__dict__)
