"""FairVAL: Fair Visual Active Learning with Information-Theoretic Guarantees."""

__version__ = "0.1.0"

from .algorithm import FairVAL
from .acquisition import AccuracyScorer, FairnessScorer, CompositeScorer
from .scheduler import SigmoidScheduler
from .backbone import load_backbone
from .estimator import DemographicEstimator
from .trainer import FairConstrainedERM
from .metrics import compute_eod, compute_wgr, compute_f1, labels_to_target
