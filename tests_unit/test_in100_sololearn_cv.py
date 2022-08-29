import unittest
from imagenet_subset_generator.versions.in100_sololearn import CLASSES as CV0
from imagenet_subset_generator.versions.in100_sololearn_cv1 import CLASSES as CV1
from imagenet_subset_generator.versions.in100_sololearn_cv2 import CLASSES as CV2
from imagenet_subset_generator.versions.in100_sololearn_cv3 import CLASSES as CV3
from imagenet_subset_generator.versions.in100_sololearn_cv4 import CLASSES as CV4
from imagenet_subset_generator.versions.in100_sololearn_cv5 import CLASSES as CV5
from imagenet_subset_generator.versions.in100_sololearn_cv6 import CLASSES as CV6
from imagenet_subset_generator.versions.in100_sololearn_cv7 import CLASSES as CV7
from imagenet_subset_generator.versions.in100_sololearn_cv8 import CLASSES as CV8
from imagenet_subset_generator.versions.in100_sololearn_cv9 import CLASSES as CV9


class TestIn100SololearnCv(unittest.TestCase):
    def test_mutually_exclusive(self):
        all_classes = CV0 + CV1 + CV2 + CV3 + CV4 + CV5 + CV6 + CV7 + CV8 + CV9
        self.assertEqual(1000, len(all_classes))
        self.assertEqual(1000, len(set(all_classes)))