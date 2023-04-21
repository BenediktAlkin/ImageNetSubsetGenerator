# ImageNet subset generator

Generate a subsets from the original ImageNet1K dataset.
Some commonly used subsets:
- [ImageNet 10% subset](https://github.com/google-research/simclr/blob/master/imagenet_subsets/10percent.txt)
- [ImageNet 1% subset](https://github.com/google-research/simclr/blob/master/imagenet_subsets/1percent.txt)
- Extreme low-shot subsets from [MSN](https://github.com/facebookresearch/msn)


# Usage
- `git clone https://github.com/BenediktAlkin/ImageNetSubsetGenerator`
- `cd ImageNetSubsetGenerator`


## Generate subset

- `python main_subset.py --in1k_path <ImageNet1K_path> --out_path <out_path> --version in100_sololearn`
- this will copy the corresponding samples from the `ImageNet1K_path` to `out_path`
- it can then be readily used with e.g. torchvision ImageFolder `subset = ImageFolder(root=<out_path>)`

For example: `python main_subset.py --in1k_path /data/imagenet1k --out_path /data/imagenet1k_10percent_simclrv2 --version in1k_10percent_simclrv2`


You can find all supported versions [here](https://github.com/BenediktAlkin/ImageNetSubsetGenerator/tree/main/imagenet_subset_generator/versions) or via `python main_subset.py --help`.



## Check classes/samples of dataset

`python main_statistics.py <path>`
```
train n_classes: 1000
valid n_classes: 1000
train n_samples: 1282169
valid n_samples: 50000
train classes: ['n01440764', ...]
valid classes: ['n01440764', ...]
```

