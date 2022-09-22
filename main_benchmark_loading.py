from imagenet_subset_generator.filelist_image_folder import FilelistImageFolder
import os
from pathlib import Path
from imagenet_subset_generator.h5_image_folder import H5ImageFolder
import argparse
import kappaprofiler as kp
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Resize, Compose


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--num_workers", type=int, required=True)
    return parser.parse_args()

def benchmark(dataset, num_workers, name):
    loader = DataLoader(dataset, batch_size=64, num_workers=num_workers)
    loader_iter = iter(loader)
    n_batches = len(loader)
    i = 0
    with kp.Stopwatch() as sw:
        while i < n_batches:
            next(loader_iter)
            i += 1
            if i % 10 == 0:
                print(i)
    print(f"{name} with {len(dataset)} samples took {sw.elapsed_seconds}")


def main(data_path, num_workers):
    data_path = Path(data_path).expanduser()
    train_image_folder = data_path / "train"
    val_image_folder = data_path / "val"
    train_h5_file = data_path / "train.h5"
    train_filelist = data_path / "train_filelist.txt"
    val_h5_file = data_path / "val.h5"
    val_filelist = data_path / "val_filelist.txt"
    tt = Compose([Resize((224, 224)), ToTensor()])
    if train_image_folder.exists():
        benchmark(ImageFolder(train_image_folder, transform=tt), num_workers=num_workers, name="train ImageFolder")
    if val_image_folder.exists():
        benchmark(ImageFolder(val_image_folder, transform=tt), num_workers=num_workers, name="val ImageFolder")
    if train_image_folder.exists() and train_filelist.exists():
        benchmark(
            FilelistImageFolder(train_image_folder, filelist_uri=train_filelist),
            num_workers=num_workers,
            name="train FilelistImageFolder",
        )
    if val_image_folder.exists() and val_filelist.exists():
        benchmark(
            FilelistImageFolder(val_image_folder, filelist_uri=val_filelist),
            num_workers=num_workers,
            name="val FilelistImageFolder",
        )
    if train_h5_file.exists() and train_filelist.exists():
        ds = H5ImageFolder(h5_file_uri=train_h5_file, filelist_uri=train_filelist)
        benchmark(ds, num_workers=num_workers, name="train H5")
    if val_h5_file.exists() and val_filelist.exists():
        ds = H5ImageFolder(h5_file_uri=val_h5_file, filelist_uri=val_filelist)
        benchmark(ds, num_workers=num_workers, name="val H5")



if __name__ == "__main__":
    main(**parse_args().__dict__)