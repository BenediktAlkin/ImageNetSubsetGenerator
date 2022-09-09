from .in100_sololearn_cv4 import CLASSES as CV4
from .in100_sololearn_cv5 import CLASSES as CV5

INFO = [
    "'cross validation split 2' with 200 classes with sololearn as base",
    "(concatenation of in100_sololearn_cv4 + in100_sololearn_cv5)",
]
CLASSES = list(sorted(CV4 + CV5))
