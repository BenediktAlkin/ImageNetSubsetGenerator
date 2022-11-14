import unittest

from imagenet_subset_generator.util import parse_version
from imagenet_subset_generator.versions.in100_sololearn import CLASSES as IN100SOLO_CLASSES
from imagenet_subset_generator.versions.in100_sololearn import INFO as IN100SOLO_INFO
from imagenet_subset_generator.versions.in1k import CLASSES as IN1K_CLASSES
from imagenet_subset_generator.versions.in1k import INFO as IN1K_INFO


class TestUtil(unittest.TestCase):
    def test_invalid_version(self):
        with self.assertRaises(AssertionError):
            parse_version(version="invalid")

    def test_version(self):
        classes, files, info = parse_version(version="in100_sololearn")
        self.assertEqual(IN100SOLO_CLASSES, classes)
        self.assertIsNone(files)
        self.assertEqual(IN100SOLO_INFO, info)
