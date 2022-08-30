import unittest
from imagenet_subset_generator.util import get_versions, get_classes_and_info

class TestVersionsAreSorted(unittest.TestCase):
    def test_versions_are_sorted(self):
        for version in get_versions():
            classes, _ = get_classes_and_info(version=version)
            self.assertEqual(sorted(classes), classes, f"{version} is not sorted")