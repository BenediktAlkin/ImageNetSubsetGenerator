from imagenet_subset_generator.versions.in1k import CLASSES
from imagenet_subset_generator.versions.in_a import CLASSES as CLASSES_A
from imagenet_subset_generator.versions.in_r import CLASSES as CLASSES_R
import yaml

def main():
    cls_to_idx = {cls: i for i, cls in enumerate(CLASSES)}
    for name, classes in [
        ["ImageNet-A", CLASSES_A],
        ["ImageNet-R", CLASSES_R],
    ]:
        idxs = {cls: cls_to_idx[cls] for cls in classes}
        with open(f"res/{name} class_to_idx.yaml", "w") as f:
            yaml.safe_dump(idxs, f)



if __name__ == "__main__":
    main()