"""
Component Factories

Factories for creating reward functions, models, and algorithms from configuration.
"""

import logging
from pathlib import Path
from typing import Any


class RewardFactory:
    """Factory for creating reward function instances."""

    @staticmethod
    def create(config):
        """
        Create reward function from configuration.

        Args:
            config: ExperimentConfig instance

        Returns:
            RewardFunction instance
        """
        # Import here to avoid circular dependencies
        import tmrl.config.config_constants as cfg
        from tmrl.custom.tm.utils.compute_reward import RewardFunction

        # Get reward parameters with defaults
        params = config.reward_params.copy()

        # Set default reward path if not specified
        if "reward_data_path" not in params:
            params["reward_data_path"] = cfg.REWARD_PATH

        # Add websocket client (required for current implementation)
        if "ws_client" not in params:
            # Import and create websocket client
            try:
                from tmrl.logging.graphing_client import WebSocketClient

                ws_client = WebSocketClient("ws://127.0.0.1:6789")
                ws_client.start()
                params["ws_client"] = ws_client
            except Exception as e:
                logging.warning(f"Could not create websocket client: {e}")
                params["ws_client"] = None

        reward_type = config.reward_type.lower()

        if reward_type == "baseline":
            # Use standard reward function
            return RewardFunction(**params)

        elif reward_type == "time_optimal":
            # Import TimeOptimalReward
            try:
                from tmrl.custom.tm.utils.compute_reward import TimeOptimalReward

                return TimeOptimalReward(**params)
            except ImportError:
                logging.warning("TimeOptimalReward not found, using baseline")
                return RewardFunction(**params)

        elif reward_type == "racing_line":
            # Import RacingLineReward
            try:
                from tmrl.custom.tm.utils.compute_reward import RacingLineReward

                return RacingLineReward(**params)
            except ImportError:
                logging.warning("RacingLineReward not found, using baseline")
                return RewardFunction(**params)

        elif reward_type == "hybrid":
            # Import HybridReward
            try:
                from tmrl.custom.tm.utils.compute_reward import HybridReward

                return HybridReward(**params)
            except ImportError:
                logging.warning("HybridReward not found, using baseline")
                return RewardFunction(**params)

        elif reward_type == "custom":
            # Allow custom reward function via module path
            if "module" not in params:
                raise ValueError("Custom reward type requires 'module' parameter")

            module_path = params.pop("module")
            module_name, class_name = module_path.rsplit(".", 1)

            import importlib

            module = importlib.import_module(module_name)
            reward_class = getattr(module, class_name)

            return reward_class(**params)

        else:
            raise ValueError(f"Unknown reward type: {reward_type}")


class ModelFactory:
    """Factory for creating model instances."""

    @staticmethod
    def create(config, observation_space, action_space):
        """
        Create model from configuration.

        Args:
            config: ExperimentConfig instance
            observation_space: Gymnasium observation space
            action_space: Gymnasium action space

        Returns:
            Model instance (ActorCritic)
        """
        # Import here to avoid circular dependencies
        from tmrl.custom.custom_models import (
            MLPActorCritic,
            REDQMLPActorCritic,
            VanillaCNNActorCritic,
            VanillaColorCNNActorCritic,
        )

        params = config.model_params.copy()
        model_type = config.model_type.lower()

        # Select model based on type and algorithm
        if model_type == "mlp":
            if config.algorithm == "redq_sac":
                return REDQMLPActorCritic(observation_space, action_space, **params)
            else:
                return MLPActorCritic(observation_space, action_space, **params)

        elif model_type == "cnn":
            # Determine grayscale vs color from config or observation space
            grayscale = params.pop("grayscale", True)
            if grayscale:
                return VanillaCNNActorCritic(observation_space, action_space, **params)
            else:
                return VanillaColorCNNActorCritic(observation_space, action_space, **params)

        elif model_type == "lstm":
            try:
                from tmrl.custom.custom_models import RNNActorCritic

                return RNNActorCritic(observation_space, action_space, **params)
            except ImportError:
                logging.warning("RNNActorCritic not found, using MLP")
                return MLPActorCritic(observation_space, action_space, **params)

        elif model_type == "lstm_cnn":
            # LSTM + CNN hybrid (not yet implemented in original code)
            logging.warning("LSTM+CNN not implemented, using CNN")
            return VanillaCNNActorCritic(observation_space, action_space, **params)

        elif model_type == "custom":
            # Allow custom model via module path
            if "module" not in params:
                raise ValueError("Custom model type requires 'module' parameter")

            module_path = params.pop("module")
            module_name, class_name = module_path.rsplit(".", 1)

            import importlib

            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)

            return model_class(observation_space, action_space, **params)

        else:
            raise ValueError(f"Unknown model type: {model_type}")


class AlgorithmFactory:
    """Factory for creating training algorithm instances."""

    @staticmethod
    def create(config, observation_space, action_space, device="cuda"):
        """
        Create training algorithm from configuration.

        Args:
            config: ExperimentConfig instance
            observation_space: Gymnasium observation space
            action_space: Gymnasium action space
            device: Device for training ('cuda' or 'cpu')

        Returns:
            TrainingAgent instance
        """
        # Import here to avoid circular dependencies
        from tmrl.custom.custom_algorithms import REDQSACAgent, SpinupSacAgent

        params = config.training_params.copy()
        algorithm = config.algorithm.lower()

        # Get model from factory
        model_cls_instance = ModelFactory.create(config, observation_space, action_space)
        model_cls = type(model_cls_instance)

        # Common parameters
        common_params = {
            "observation_space": observation_space,
            "action_space": action_space,
            "device": device,
            "model_cls": model_cls,
        }

        # Merge with user params
        common_params.update(params)

        if algorithm == "sac":
            return SpinupSacAgent(**common_params)

        elif algorithm == "redq_sac":
            # REDQ requires additional parameters
            if "n" not in common_params:
                common_params["n"] = 10  # Default ensemble size
            if "m" not in common_params:
                common_params["m"] = 2  # Default subset size
            if "q_updates_per_policy_update" not in common_params:
                common_params["q_updates_per_policy_update"] = 20

            return REDQSACAgent(**common_params)

        elif algorithm == "ppo":
            # PPO not implemented in original code
            logging.warning("PPO not implemented, using SAC")
            return SpinupSacAgent(**common_params)

        elif algorithm == "td3":
            # TD3 not implemented in original code
            logging.warning("TD3 not implemented, using SAC")
            return SpinupSacAgent(**common_params)

        elif algorithm == "custom":
            # Allow custom algorithm via module path
            if "module" not in params:
                raise ValueError("Custom algorithm type requires 'module' parameter")

            module_path = params.pop("module")
            module_name, class_name = module_path.rsplit(".", 1)

            import importlib

            module = importlib.import_module(module_name)
            algorithm_class = getattr(module, class_name)

            return algorithm_class(**common_params)

        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")


class EnvironmentFactory:
    """Factory for creating environment instances."""

    @staticmethod
    def create(config):
        """
        Create environment from configuration.

        Args:
            config: ExperimentConfig instance

        Returns:
            Gymnasium environment instance
        """
        import rtgym

        import tmrl.config.config_constants as cfg
        from tmrl.custom.tm.tm_gym_interfaces import TM2020Interface, TM2020InterfaceLidar, TM2020InterfaceLidarProgress
        from tmrl.envs import GenericGymEnv
        from tmrl.util import partial

        env_params = config.env_params.copy()

        # Determine interface type
        interface_type = env_params.get("interface_type", "lidar")

        if interface_type == "lidar":
            interface = partial(
                TM2020InterfaceLidar,
                img_hist_len=env_params.get("img_hist_len", 4),
                gamepad=env_params.get("gamepad", True),
            )
        elif interface_type == "lidar_progress":
            interface = partial(
                TM2020InterfaceLidarProgress,
                img_hist_len=env_params.get("img_hist_len", 4),
                gamepad=env_params.get("gamepad", True),
            )
        elif interface_type == "full":
            interface = partial(
                TM2020Interface,
                img_hist_len=env_params.get("img_hist_len", 4),
                gamepad=env_params.get("gamepad", True),
                grayscale=env_params.get("grayscale", True),
                resize_to=(env_params.get("img_width", 64), env_params.get("img_height", 64)),
            )
        else:
            raise ValueError(f"Unknown interface type: {interface_type}")

        # Build rtgym config
        config_dict = rtgym.DEFAULT_CONFIG_DICT.copy()
        config_dict["interface"] = interface

        # Add rtgym-specific parameters
        rtgym_params = env_params.get("rtgym", {})
        for k, v in rtgym_params.items():
            config_dict[k] = v

        # Create environment
        return GenericGymEnv(id=cfg.RTGYM_VERSION, gym_kwargs={"config": config_dict})
