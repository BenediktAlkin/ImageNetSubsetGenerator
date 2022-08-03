import unittest
from imagenet_subset_generator.generate_dummy_dataset import generate_dummy_dataset
from tests_util.assert_clean_directory import assert_clean_directory
from tests_util.directory_utils import n_files_in_directory

class TestGenerateDummyDataset(unittest.TestCase):
    def test_2classes(self):
        root = assert_clean_directory(__name__)
        generate_dummy_dataset(
            out_path=root,
            train_size=1,
            valid_size=1,
            n_classes=2,
            resolution_min=2,
            resolution_max=2,
        )
        self.assertEqual(4, n_files_in_directory(root))

    def test_in1k(self):
        root = assert_clean_directory(__name__)
        generate_dummy_dataset(
            out_path=root,
            train_size=1,
            valid_size=1,
            resolution_min=2,
            resolution_max=2,
        )
        self.assertEqual(2000, n_files_in_directory(root))

    def test_in100sololearn(self):
        root = assert_clean_directory(__name__)
        generate_dummy_dataset(
            out_path=root,
            train_size=1,
            valid_size=1,
            resolution_min=2,
            resolution_max=2,
            version="in100_sololearn",
        )
        self.assertEqual(200, n_files_in_directory(root))