"""
TMRL Experiment Framework

Enables rapid experimentation with reward functions, neural architectures,
and RL algorithms through configuration-driven development.
"""

from .comparison import ExperimentComparison
from .config import ExperimentConfig
from .factories import AlgorithmFactory, ModelFactory, RewardFactory
from .runner import ExperimentRunner

__all__ = [
    "ExperimentConfig",
    "ExperimentRunner",
    "RewardFactory",
    "ModelFactory",
    "AlgorithmFactory",
    "ExperimentComparison",
]
