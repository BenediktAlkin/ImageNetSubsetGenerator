import os
import itertools

def n_files_in_directory(root):
    return sum([len(files) for _, _, files in os.walk(root)])

def all_folder_names(root):
    return list(itertools.chain(*[dirs for _, dirs, _ in os.walk(root)]))