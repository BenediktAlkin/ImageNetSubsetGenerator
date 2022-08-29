import unittest
from imagenet_subset_generator.versions.in200_sololearn_cv0 import CLASSES as CV0
from imagenet_subset_generator.versions.in200_sololearn_cv1 import CLASSES as CV1
from imagenet_subset_generator.versions.in200_sololearn_cv2 import CLASSES as CV2
from imagenet_subset_generator.versions.in200_sololearn_cv3 import CLASSES as CV3
from imagenet_subset_generator.versions.in200_sololearn_cv4 import CLASSES as CV4


class TestIn200SololearnCv(unittest.TestCase):
    def test_mutually_exclusive(self):
        all_classes = CV0 + CV1 + CV2 + CV3 + CV4
        self.assertEqual(1000, len(all_classes))
        self.assertEqual(1000, len(set(all_classes)))