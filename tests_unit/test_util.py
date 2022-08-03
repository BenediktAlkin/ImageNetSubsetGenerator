import unittest

from imagenet_subset_generator.util import get_classes_and_info
from imagenet_subset_generator.versions.in100_sololearn import CLASSES as IN100SOLO_CLASSES
from imagenet_subset_generator.versions.in100_sololearn import INFO as IN100SOLO_INFO
from imagenet_subset_generator.versions.in1k import CLASSES as IN1K_CLASSES
from imagenet_subset_generator.versions.in1k import INFO as IN1K_INFO


class TestUtil(unittest.TestCase):
    def test_noargs_nodefault(self):
        with self.assertRaises(AssertionError):
            get_classes_and_info(use_in1k_as_default=False)

    def test_noargs_default(self):
        classes, info = get_classes_and_info(use_in1k_as_default=True)
        self.assertEqual(IN1K_CLASSES, classes)
        self.assertEqual(IN1K_INFO, info)

    def test_version(self):
        classes, info = get_classes_and_info(version="in100_sololearn")
        self.assertEqual(IN100SOLO_CLASSES, classes)
        self.assertEqual(IN100SOLO_INFO, info)

    def test_classes(self):
        expected = ["n111111", "n111112"]
        classes, info = get_classes_and_info(classes=expected)
        self.assertEqual(expected, classes)
        self.assertIsNone(info)

    def test_nclasses(self):
        classes, _ = get_classes_and_info(n_classes=21)
        self.assertEqual(IN1K_CLASSES[:21], classes)
