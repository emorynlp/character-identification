from constants import ExperimentTypes
from experiments.baseline.tools.state import SingEvalCorefState
from experiments.latest.tools.state import PluralCorefState
from experiments.sing_only.tools.state import SingOnlyCorefState


def coref_state_factory(experiment_type):
    if experiment_type == ExperimentTypes.SING_ONLY:
        return SingOnlyCorefState
    elif experiment_type == ExperimentTypes.BASELINE:
        return SingEvalCorefState
    elif experiment_type == ExperimentTypes.LATEST:
        return PluralCorefState

    assert 0, "Bad experiment type: " + experiment_type
