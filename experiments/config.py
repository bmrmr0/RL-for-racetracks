"""
Experiment Configuration

Defines the ExperimentConfig dataclass for managing experiment parameters.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


@dataclass
class ExperimentConfig:
    """
    Configuration for a TMRL experiment.

    Attributes:
        name: Unique experiment identifier
        reward_type: Type of reward function ('baseline', 'time_optimal', 'racing_line', etc.)
        model_type: Neural network architecture ('mlp', 'cnn', 'lstm', etc.)
        algorithm: RL algorithm ('sac', 'redq_sac', etc.)
        reward_params: Parameters for reward function
        model_params: Parameters for model architecture
        training_params: Parameters for training algorithm
        env_params: Parameters for environment
    """

    name: str
    reward_type: str = "baseline"
    model_type: str = "mlp"
    algorithm: str = "sac"

    # Component-specific parameters
    reward_params: Dict[str, Any] = field(default_factory=dict)
    model_params: Dict[str, Any] = field(default_factory=dict)
    training_params: Dict[str, Any] = field(default_factory=dict)
    env_params: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    description: Optional[str] = None
    tags: list = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExperimentConfig":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML config file

        Returns:
            ExperimentConfig instance
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        return cls(**data)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ExperimentConfig":
        """
        Load configuration from JSON file (backward compatibility).

        Args:
            path: Path to JSON config file

        Returns:
            ExperimentConfig instance
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            data = json.load(f)

        return cls(**data)

    def to_yaml(self, path: Union[str, Path]):
        """
        Save configuration to YAML file.

        Args:
            path: Path to save YAML config
        """
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def to_json(self, path: Union[str, Path]):
        """
        Save configuration to JSON file.

        Args:
            path: Path to save JSON config
        """
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def update(self, **kwargs):
        """
        Update config parameters.

        Args:
            **kwargs: Parameters to update (supports nested updates via dot notation)
        """
        for key, value in kwargs.items():
            if "." in key:
                # Handle nested updates: reward_params.speed_weight -> reward_params['speed_weight']
                parts = key.split(".")
                if len(parts) == 2:
                    category, param = parts
                    if hasattr(self, category):
                        getattr(self, category)[param] = value
            elif hasattr(self, key):
                setattr(self, key, value)

    def validate(self) -> bool:
        """
        Validate configuration parameters.

        Returns:
            True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate reward type
        valid_reward_types = ["baseline", "time_optimal", "racing_line", "hybrid", "custom"]
        if self.reward_type not in valid_reward_types:
            raise ValueError(f"Invalid reward_type: {self.reward_type}. Must be one of {valid_reward_types}")

        # Validate model type
        valid_model_types = ["mlp", "cnn", "lstm", "lstm_cnn", "custom"]
        if self.model_type not in valid_model_types:
            raise ValueError(f"Invalid model_type: {self.model_type}. Must be one of {valid_model_types}")

        # Validate algorithm
        valid_algorithms = ["sac", "redq_sac", "ppo", "td3"]
        if self.algorithm not in valid_algorithms:
            raise ValueError(f"Invalid algorithm: {self.algorithm}. Must be one of {valid_algorithms}")

        # Validate name
        if not self.name or not self.name.strip():
            raise ValueError("Experiment name cannot be empty")

        return True

    def __repr__(self) -> str:
        """String representation of config."""
        return (
            f"ExperimentConfig(name='{self.name}', "
            f"reward='{self.reward_type}', "
            f"model='{self.model_type}', "
            f"algorithm='{self.algorithm}')"
        )
