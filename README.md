# ImageNet subset generator

Generate a subset (e.g. ImageNet100) from the original ImageNet1K dataset.

# Usage
- `git clone https://github.com/BenediktAlkin/ImageNetSubsetGenerator`
- `cd ImageNet100Generator`
## Generate subset

- `python main_subset.py --in1k_path <ImageNet1K_path> --out_path <out_path> --version in100_sololearn`
- this will copy the corresponding classes from the `ImageNet1K_path` to `out_path`
- it can then be readily used with e.g. torchvision ImageFolder `subset = ImageFolder(root=<out_path>)`

## Generate dummy dataset

Generate a lightweight dummy dataset for easy debugging/development

- `python main_dummy_dataset.py --out_path <out_path> --version in100_sololearn`

## Check classes/samples of dataset

- `python main_statistics.py <path>`
```
train n_classes: 1000
valid n_classes: 1000
train n_samples: 1282169
valid n_samples: 50000
train classes: ['n01440764', ...]
valid classes: ['n01440764', ...]
```


# Versions

- `in1k` - original ImageNet1K (only available for generating a dummy dataset based on this version)
- `in100_kaggle` - subset of ImageNet1K with 100
  classes ([source](https://www.kaggle.com/datasets/ambityga/imagenet100))
- `in100_sololearn` - subset of ImageNet1K with 100
  classes ([source](https://github.com/vturrisi/solo-learn/issues/137))
- `in10_m3ae` - classes used for visualization in [M3AE](https://arxiv.org/abs/2205.14204)