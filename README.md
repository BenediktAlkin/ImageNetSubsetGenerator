# ImageNet100Generator

Generate a subset (e.g. ImageNet-100) from the original ImageNet dataset (commonly referred to as ImageNet1K).

# Usage

- `git clone https://github.com/BenediktAlkin/ImageNet100Generator`
- `cd ImageNet100Generator`
- `python main.py --in1k_path <ImageNet1K_path> --out_path <ImageNet100_path> --version solo-learn`
- this will copy the corresponding classes from the `ImageNet1K_path` to `ImageNet100_path`
- it can then be readily used with e.g. torchvision ImageFolder `in100 = ImageFolder(root=<ImageNet100_path>)`
