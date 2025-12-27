"""
Experiment Runner

Orchestrates training experiments using configuration-driven approach.
"""

from pathlib import Path
import logging
import time
import json
from typing import Optional
import pandas as pd

from .config import ExperimentConfig
from .factories import (
    RewardFactory,
    ModelFactory,
    AlgorithmFactory,
    EnvironmentFactory
)


class ExperimentRunner:
    """
    Manages the lifecycle of a training experiment.
    
    Handles:
    - Component creation from config
    - Experiment directory setup
    - Training execution
    - Results saving and tracking
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment runner.
        
        Args:
            config: ExperimentConfig instance
        """
        self.config = config
        self.config.validate()  # Validate config on init
        
        # Setup experiment directory
        self.experiment_dir = Path("experiments/runs") / config.name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Track experiment start time
        self.start_time = None
        self.end_time = None
        
        logging.info(f"Initialized experiment: {config.name}")
        logging.info(f"Reward: {config.reward_type}, Model: {config.model_type}, Algorithm: {config.algorithm}")
    
    def _setup_logging(self):
        """Setup experiment-specific logging."""
        log_file = self.experiment_dir / "experiment.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to root logger
        logging.getLogger().addHandler(file_handler)
    
    def _save_config(self):
        """Save experiment configuration."""
        config_path = self.experiment_dir / "config.yaml"
        self.config.to_yaml(str(config_path))
        logging.info(f"Saved config to {config_path}")
    
    def _create_components(self):
        """
        Create all components needed for training.
        
        Returns:
            Tuple of (env, trainer, worker) - components ready for training
        """
        logging.info("Creating environment...")
        env = EnvironmentFactory.create(self.config)
        
        obs_space = env.observation_space
        act_space = env.action_space
        
        logging.info(f"Observation space: {obs_space}")
        logging.info(f"Action space: {act_space}")
        
        logging.info(f"Creating model ({self.config.model_type})...")
        # Model is created inside AlgorithmFactory
        
        logging.info(f"Creating algorithm ({self.config.algorithm})...")
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Using device: {device}")
        
        algorithm = AlgorithmFactory.create(
            self.config,
            obs_space,
            act_space,
            device=device
        )
        
        logging.info("All components created successfully")
        
        return env, algorithm
    
    def _integrate_with_tmrl(self):
        """
        Integrate with existing TMRL training infrastructure.
        
        This creates the necessary Server, Trainer, and RolloutWorker
        using the existing TMRL architecture.
        
        Returns:
            Tuple of (server, trainer, worker)
        """
        from tmrl.networking import Server, Trainer, RolloutWorker
        from tmrl.training_offline import TorchTrainingOffline
        from tmrl.custom.custom_memories import MemoryTMLidar, MemoryTMFull
        from tmrl.util import partial
        import tmrl.config.config_constants as cfg
        
        logging.info("Integrating with TMRL infrastructure...")
        
        # Determine memory type based on model
        if self.config.model_type == "mlp":
            memory_cls = MemoryTMLidar
        else:
            memory_cls = MemoryTMFull
        
        # Create partial for environment
        env_cls = partial(EnvironmentFactory.create, self.config)
        
        # Create dummy env to get spaces
        dummy_env = env_cls()
        obs_space = dummy_env.observation_space
        act_space = dummy_env.action_space
        
        # Create memory
        memory = partial(
            memory_cls,
            memory_size=self.config.training_params.get('memory_size', 1000000),
            batch_size=self.config.training_params.get('batch_size', 256),
            dataset_path=cfg.DATASET_PATH,
            imgs_obs=self.config.env_params.get('img_hist_len', 4),
            act_buf_len=2,
            crc_debug=False
        )
        
        # Create algorithm partial
        device = 'cuda' if self.config.training_params.get('cuda', True) else 'cpu'
        algorithm_partial = partial(
            AlgorithmFactory.create,
            self.config,
            obs_space,
            act_space,
            device
        )
        
        # Create trainer class
        trainer_cls = partial(
            TorchTrainingOffline,
            env_cls=env_cls,
            memory_cls=memory,
            training_agent_cls=algorithm_partial,
            epochs=self.config.training_params.get('epochs', 10000),
            rounds=self.config.training_params.get('rounds', 10),
            steps=self.config.training_params.get('steps', 1000),
            update_model_interval=self.config.training_params.get('update_model_interval', 100),
            update_buffer_interval=self.config.training_params.get('update_buffer_interval', 100),
            max_training_steps_per_env_step=self.config.training_params.get('max_training_steps_per_env_step', 2.0),
            start_training=self.config.training_params.get('start_training', 400),
            device=device
        )
        
        # Setup paths
        model_path_worker = str(Path(cfg.WEIGHTS_FOLDER) / f"{self.config.name}.tmod")
        model_path_trainer = str(Path(cfg.WEIGHTS_FOLDER) / f"{self.config.name}_t.tmod")
        checkpoint_path = str(Path(cfg.CHECKPOINTS_FOLDER) / f"{self.config.name}_t.tcpt")
        
        logging.info(f"Model paths: worker={model_path_worker}, trainer={model_path_trainer}")
        logging.info(f"Checkpoint path: {checkpoint_path}")
        
        # Note: For simplicity in this implementation, we'll create a standalone training loop
        # rather than the full distributed Server/Trainer/Worker setup
        # This makes it easier to test and iterate
        
        return trainer_cls, model_path_trainer, checkpoint_path
    
    def run(self, test_mode: bool = False):
        """
        Run the training experiment.
        
        Args:
            test_mode: If True, run in test mode (shorter training for verification)
        """
        self.start_time = time.time()
        
        try:
            # Save configuration
            self._save_config()
            
            # Integrate with TMRL
            trainer_cls, model_path, checkpoint_path = self._integrate_with_tmrl()
            
            # Create trainer instance
            logging.info("Creating trainer instance...")
            trainer_instance = trainer_cls()
            
            if test_mode:
                logging.info("Running in TEST MODE - limited epochs")
                # Override epochs for quick test
                trainer_instance.epochs = 2
                trainer_instance.rounds = 2
                trainer_instance.steps = 10
            
            # Run training
            logging.info("Starting training...")
            logging.info(f"Epochs: {trainer_instance.epochs}, Rounds: {trainer_instance.rounds}, Steps: {trainer_instance.steps}")
            
            # For now, just validate that everything initializes correctly
            # Full training would require the complete Server/Worker setup
            logging.info("✓ All components initialized successfully")
            logging.info("✓ Configuration validated")
            logging.info("✓ Ready for training")
            
            # Save experiment metadata
            self._save_metadata(status="initialized")
            
            return True
            
        except Exception as e:
            logging.error(f"Experiment failed: {e}", exc_info=True)
            self._save_metadata(status="failed", error=str(e))
            raise
        
        finally:
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            logging.info(f"Experiment duration: {duration:.2f}s")
    
    def _save_metadata(self, status="completed", error=None):
        """Save experiment metadata."""
        metadata = {
            'name': self.config.name,
            'status': status,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': (self.end_time - self.start_time) if self.end_time else None,
            'config': self.config.to_dict(),
            'error': error
        }
        
        metadata_path = self.experiment_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"Saved metadata to {metadata_path}")
    
    def get_results(self) -> Optional[dict]:
        """
        Load results from completed experiment.
        
        Returns:
            Dictionary of results, or None if not available
        """
        metadata_path = self.experiment_dir / "metadata.json"
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def list_experiments(runs_dir: str = "experiments/runs") -> list:
        """
        List all experiments in runs directory.
        
        Args:
            runs_dir: Path to runs directory
            
        Returns:
            List of experiment names
        """
        runs_path = Path(runs_dir)
        if not runs_path.exists():
            return []
        
        return [d.name for d in runs_path.iterdir() if d.is_dir()]
    
    @staticmethod
    def load_experiment(name: str, runs_dir: str = "experiments/runs") -> Optional['ExperimentRunner']:
        """
        Load a previous experiment.
        
        Args:
            name: Experiment name
            runs_dir: Path to runs directory
            
        Returns:
            ExperimentRunner instance, or None if not found
        """
        config_path = Path(runs_dir) / name / "config.yaml"
        
        if not config_path.exists():
            return None
        
        config = ExperimentConfig.from_yaml(str(config_path))
        return ExperimentRunner(config)

