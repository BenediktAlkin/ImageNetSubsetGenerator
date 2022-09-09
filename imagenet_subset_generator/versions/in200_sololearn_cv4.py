from .in100_sololearn_cv8 import CLASSES as CV8
from .in100_sololearn_cv9 import CLASSES as CV9

INFO = [
    "'cross validation split 4' with 200 classes with sololearn as base",
    "(concatenation of in100_sololearn_cv8 + in100_sololearn_cv9)",
]
CLASSES = list(sorted(CV8 + CV9))
