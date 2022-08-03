# ImageNet subset generator

Generate a subset (e.g. ImageNet100) from the original ImageNet1K dataset.

# Usage

## Generate subset

- `git clone https://github.com/BenediktAlkin/ImageNetSubsetGenerator`
- `cd ImageNet100Generator`
- `python main_subset.py --in1k_path <ImageNet1K_path> --out_path <out_path> --version in100_sololearn`
- this will copy the corresponding classes from the `ImageNet1K_path` to `out_path`
- it can then be readily used with e.g. torchvision ImageFolder `subset = ImageFolder(root=<out_path>)`

## Generate dummy dataset

Generate a lightweight dummy dataset for easy debugging/development

- `git clone https://github.com/BenediktAlkin/ImageNetSubsetGenerator`
- `cd ImageNet100Generator`
- `python main_dummy_dataset.py --out_path <out_path> --version in100_sololearn`

# Versions

- `in1k` - original ImageNet1K (only available for generating a dummy dataset based on this version)
- `in100_kaggle` - subset of ImageNet1K with 100
  classes ([source](https://www.kaggle.com/datasets/ambityga/imagenet100))
- `in100_sololearn` - subset of ImageNet1K with 100
  classes ([source](https://github.com/vturrisi/solo-learn/issues/137))
- `in10_m3ae` - classes used for visualization in [M3AE](https://arxiv.org/abs/2205.14204)