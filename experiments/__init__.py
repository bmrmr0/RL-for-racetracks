"""
TMRL Experiment Framework

Enables rapid experimentation with reward functions, neural architectures,
and RL algorithms through configuration-driven development.
"""

from .config import ExperimentConfig
from .runner import ExperimentRunner
from .factories import RewardFactory, ModelFactory, AlgorithmFactory
from .comparison import ExperimentComparison

__all__ = [
    'ExperimentConfig',
    'ExperimentRunner', 
    'RewardFactory',
    'ModelFactory',
    'AlgorithmFactory',
    'ExperimentComparison'
]

