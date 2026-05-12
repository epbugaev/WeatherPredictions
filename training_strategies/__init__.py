"""Training-strategies package: importing populates ``utils.registry.STRATEGIES``.

Five plain ``StepStrategy`` classes (plus shared helpers `_index_maps` and
`_log_figures`) cover every training/validation pattern this project uses.
The mapping of YAML ``training.litmodel`` keys to strategy classes is::

    mutiout_f                  -> SimpleStep
    mutiout                    -> TimestepSelectStep
    multiout_double            -> AutoregressiveStep
    mutiout_imvp               -> IterativeManualStep
    mutiout_imvp_small_world   -> IterativeManualStep   (same class, two keys)
    mutiout_predrnn            -> PredRNNStep
"""

from training_strategies.autoregressive import AutoregressiveStep
from training_strategies.base import StepContext, StepStrategy
from training_strategies.iterative_manual import IterativeManualStep
from training_strategies.predrnn import PredRNNStep
from training_strategies.simple import SimpleStep
from training_strategies.timestep_select import TimestepSelectStep

__all__ = [
    "AutoregressiveStep",
    "IterativeManualStep",
    "PredRNNStep",
    "SimpleStep",
    "StepContext",
    "StepStrategy",
    "TimestepSelectStep",
]
