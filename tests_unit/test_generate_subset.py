import unittest

from imagenet_subset_generator.generate_dummy_dataset import generate_dummy_dataset
from imagenet_subset_generator.generate_subset import generate_subset
from imagenet_subset_generator.versions.in100_sololearn import CLASSES
from tests_util.assert_clean_directory import assert_clean_directory
from imagenet_subset_generator.util import n_files_in_directory, folder_names_in_directory


class TestGenerateSubset(unittest.TestCase):
    def test_in100sololearn(self):
        # generate dummy in1k
        root_in1k = assert_clean_directory(__name__, subfolder="in1k")
        generate_dummy_dataset(
            out_path=root_in1k,
            train_size=1,
            valid_size=1,
            resolution_min=2,
            resolution_max=2,
        )
        self.assertEqual(2000, n_files_in_directory(root_in1k))

        # generate in100
        root_in100 = assert_clean_directory(__name__, subfolder="in100")
        generate_subset(in1k_path=root_in1k, out_path=root_in100, version="in100_sololearn")
        self.assertEqual(201, n_files_in_directory(root_in100))
        self.assertEqual(CLASSES, folder_names_in_directory(root_in100 / "train"))
        self.assertEqual(CLASSES, folder_names_in_directory(root_in100 / "val"))
