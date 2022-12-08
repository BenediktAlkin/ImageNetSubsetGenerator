from imagenet_subset_generator.versions.in1k import CLASSES


def main():
    script = "\n".join([f"os.mkdir(val/{cls})" for cls in CLASSES])
    with open("create_empty_in1k_folders.sh", "w") as f:
        f.write(script)



if __name__ == "__main__":
    main()