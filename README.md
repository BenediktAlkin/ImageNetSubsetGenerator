# ImageNet subset generator

- Generate a subset (e.g. ImageNet100) from the original ImageNet1K dataset.
- Generate lightweight dummy datasets for development/debugging environments. 
  These dummy datasets have the same classes as the specified version but take only minimal disk space.
- Get readable labels of any ImageNet version ('ice bear' instead of 'n02134084')

# Usage
- `git clone https://github.com/BenediktAlkin/ImageNetSubsetGenerator`
- `cd ImageNetSubsetGenerator`
- `python <SCRIPT> <ARGS>`

## Generate subset

- `python main_subset.py --in1k_path <ImageNet1K_path> --out_path <out_path> --version in100_sololearn`
- this will copy the corresponding classes from the `ImageNet1K_path` to `out_path`
- it can then be readily used with e.g. torchvision ImageFolder `subset = ImageFolder(root=<out_path>)`

## Generate dummy dataset

Generate a lightweight dummy dataset for debugging/development environments.

`python main_dummy_dataset.py --out_path <out_path> --version <VERSION> [--train_size <TRAIN_SIZE>] [--valid_size <VALID_SIZE>] [--resolution_min <RES_MIN>] [--resolution_max <RES_MAX>]`

Optional CLI arguments:
- `--train_size <TRAIN_SIZE>` how many train samples per class
- `--valid_size <VALID_SIZE>` how many validaion samples per class
- `--resolution_min <RESOLUTION_MIN>` each generated dummy sample has a random resolution sampled from `[RESOLUTION_MIN; RESOLUTION_MAX]` (16 by default)
- `--resolution_max <RESOLUTION_MAX>` each generated dummy sample has a random resolution sampled from `[RESOLUTION_MIN; RESOLUTION_MAX]` (16 by default)

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


# Versions

- `in1k` - original ImageNet1K (only available for generating a dummy dataset based on this version)
- `in100_kaggle` - subset of ImageNet1K with 100
  classes ([source](https://www.kaggle.com/datasets/ambityga/imagenet100))
- `in100_sololearn` - subset of ImageNet1K with 100
  classes ([source](https://github.com/vturrisi/solo-learn/issues/137))
- `in100_sololearn_cv1` - "cross validation split" with `in100_sololearn` as base <br/>
   (100 random classes that don't occour in `in100_sololearn`)
- `in100_sololearn_cv2` - "cross validation split" with `in100_sololearn` as base <br/> 
  (100 random classes excluding `in100_sololearn` and all previous `in100_sololearn_cv` classes)
- `in100_sololearn_cv3` - "cross validation split" with `in100_sololearn` as base <br/>
  (100 random classes excluding `in100_sololearn` and all previous `in100_sololearn_cv` classes)
- `in100_sololearn_cv4` - "cross validation split" with `in100_sololearn` as base <br/>
  (100 random classes excluding `in100_sololearn` and all previous `in100_sololearn_cv` classes)
- `in100_sololearn_cv5` - "cross validation split" with `in100_sololearn` as base <br/> 
  (100 random classes excluding `in100_sololearn` and all previous `in100_sololearn_cv` classes)
- `in100_sololearn_cv6` - "cross validation split" with `in100_sololearn` as base <br/>
  (100 random classes excluding `in100_sololearn` and all previous `in100_sololearn_cv` classes)
- `in100_sololearn_cv7` - "cross validation split" with `in100_sololearn` as base <br/> 
  (100 random classes excluding `in100_sololearn` and all previous `in100_sololearn_cv` classes)
- `in100_sololearn_cv8` - "cross validation split" with `in100_sololearn` as base <br/>
  (100 random classes excluding `in100_sololearn` and all previous `in100_sololearn_cv` classes)
- `in100_sololearn_cv9` - "cross validation split" with `in100_sololearn` as base <br/>
  (100 random classes excluding `in100_sololearn` and all previous `in100_sololearn_cv` classes)
- `in10_m3ae` - classes used for visualization in [M3AE](https://arxiv.org/abs/2205.14204)
- `in100_seed1` - random selection of 100 ImageNet1K classes with `seed=1`
- `in100_seed2` - random selection of 100 ImageNet1K classes with `seed=2`
- `in100_seed3` - random selection of 100 ImageNet1K classes with `seed=3`
- `in100_seed4` - random selection of 100 ImageNet1K classes with `seed=4`
- `in100_seed5` - random selection of 100 ImageNet1K classes with `seed=5`
- `in200_seed1` - random selection of 200 ImageNet1K classes with `seed=1`
- `in200_seed2` - random selection of 200 ImageNet1K classes with `seed=2`
- `in200_seed3` - random selection of 200 ImageNet1K classes with `seed=3`
- `in200_seed4` - random selection of 200 ImageNet1K classes with `seed=4`
- `in200_seed5` - random selection of 200 ImageNet1K classes with `seed=5`
- `in200_sololearn_cv0` - concatenation of `in100_sololearn` and `in100_sololearn_cv1`
- `in200_sololearn_cv1` - concatenation of `in100_sololearn_cv2` and `in100_sololearn_cv3`
- `in200_sololearn_cv2` - concatenation of `in100_sololearn_cv4` and `in100_sololearn_cv5`
- `in200_sololearn_cv3` - concatenation of `in100_sololearn_cv6` and `in100_sololearn_cv7`
- `in200_sololearn_cv4` - concatenation of `in100_sololearn_cv8` and `in100_sololearn_cv9`


## Get class labels
`python main_get_class_labels --version <VERSION>`

`python main_get_class_labels --version in10_m3ae`
```
generating in10_m3ae
classes: ['n01440764', 'n01773797', 'n02071294', 'n02104029', 'n02134084', 'n02484975', 'n02835271', 'n03127747', 'n03492542', 'n03786901']
n01440764: ('tench', 'Tinca tinca')
n01773797: ('garden spider', 'Aranea diademata')
n02071294: ('killer whale', 'killer', 'orca', 'grampus', 'sea wolf', 'Orcinus orca')
n02104029: ('kuvasz',)
n02134084: ('ice bear', 'polar bear', 'Ursus Maritimus', 'Thalarctos maritimus')
n02484975: ('guenon', 'guenon monkey')
n02835271: ('bicycle-built-for-two', 'tandem bicycle', 'tandem')
n03127747: ('crash helmet',)
n03492542: ('hard disc', 'hard disk', 'fixed disk')
n03786901: ('mortar',)
```

## Check if folder is of version
`python main_check_folder_is_version.py --path <PATH_TO_FOLDER> --version <VERSION>`