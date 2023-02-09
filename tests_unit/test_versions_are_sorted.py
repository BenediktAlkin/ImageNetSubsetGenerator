import unittest
from imagenet_subset_generator.util import get_versions, parse_version

class TestVersionsAreSorted(unittest.TestCase):
    def test_versions_are_sorted(self):
        for version in get_versions():
            classes, files, _ = parse_version(version=version)
            if classes is not None:
                self.assertEqual(sorted(classes), classes, f"{version} is not sorted")
            if files is not None:
                self.assertEqual(sorted(files), files, f"{version} is not sorted")