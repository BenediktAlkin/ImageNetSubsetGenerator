from .in100_sololearn_cv2 import CLASSES as CV2
from .in100_sololearn_cv3 import CLASSES as CV3

INFO = [
    "'cross validation split 1' with 200 classes with sololearn as base",
    "(concatenation of in100_sololearn_cv2 + in100_sololearn_cv3)",
]
CLASSES = list(sorted(CV2 + CV3))
