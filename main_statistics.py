import argparse
from pathlib import Path
from imagenet_subset_generator import n_files_in_directory, n_folders_in_directory, folder_names_in_directory


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to the directory of which statistics should be printed")
    parser.add_argument("--verbose", action="store_true", help="also print the class names")
    return parser.parse_args()

def main(path, verbose):
    path = Path(path).expanduser()
    assert path.exists(), f"path doesn't exist: {path}"
    
    train_path = path / "train"
    if train_path.exists():
        print(f"train n_classes: {n_folders_in_directory(train_path)}")
        print(f"train n_samples: {n_files_in_directory(train_path)}")
        if verbose:
            print(f"train classes: {folder_names_in_directory(train_path)}")

    valid_path = path / "val"
    if valid_path.exists():
        print(f"valid n_classes: {n_folders_in_directory(valid_path)}")
        print(f"valid n_samples: {n_files_in_directory(valid_path)}")
        if verbose:
            print(f"valid classes: {folder_names_in_directory(valid_path)}")


if __name__ == "__main__":
    main(**parse_args().__dict__)
