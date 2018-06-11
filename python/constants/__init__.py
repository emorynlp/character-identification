from enum import Enum


class ExperimentTypes(Enum):
    SING_ONLY = "sing-only"
    BASELINE = "baseline"
    LATEST = "latest"


class SubsystemTypes(Enum):
    EXPERIMENT = "experiment"
    COREF = "coref"
    EXPORT_CLUSTERS = "export_clusters"
    ENTITY_LINKING = "linking"
