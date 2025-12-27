#!/usr/bin/env python3
"""
TMRL Experiment Training Script

Clean entry point for running training experiments with configuration-driven approach.

Usage:
    python train.py --preset baseline
    python train.py --preset time_optimal
    python train.py --config experiments/my_experiment.yaml
    python train.py --preset baseline --name my_exp_001
    python train.py --preset baseline --override reward_params.apex_weight=2.0
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple, Union

# Add repo root to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

from experiments.config import ExperimentConfig
from experiments.runner import ExperimentRunner


def parse_override(override_str: str) -> Tuple[str, Union[str, int, float, bool]]:
    """
    Parse override string in format key=value.

    Args:
        override_str: String like "reward_params.speed_weight=1.5"

    Returns:
        Tuple of (key, value)
    """
    if "=" not in override_str:
        raise ValueError(f"Invalid override format: {override_str}. Use key=value")

    key, value_str = override_str.split("=", 1)
    parsed_value: Union[str, int, float, bool] = value_str

    # Try to convert value to appropriate type
    try:
        # Try int
        parsed_value = int(value_str)
    except ValueError:
        try:
            # Try float
            parsed_value = float(value_str)
        except ValueError:
            # Try boolean
            if value_str.lower() in ("true", "false"):
                parsed_value = value_str.lower() == "true"
            # Otherwise keep as string

    return key, parsed_value


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run TMRL training experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use preset configuration
  python train.py --preset baseline
  python train.py --preset time_optimal
  python train.py --preset racing_line

  # Use custom config file
  python train.py --config experiments/my_config.yaml

  # Override specific parameters
  python train.py --preset baseline --name exp_001
  python train.py --preset baseline --override reward_params.speed_weight=1.5
  python train.py --preset baseline --override training_params.lr_actor=0.001

  # Test mode (quick verification)
  python train.py --preset baseline --test
        """,
    )

    # Config source (mutually exclusive, but not required if using list commands)
    config_group = parser.add_mutually_exclusive_group(required=False)
    config_group.add_argument(
        "--preset", type=str, help="Use preset configuration (baseline, time_optimal, racing_line)"
    )
    config_group.add_argument("--config", type=str, help="Path to custom YAML config file")

    # Overrides
    parser.add_argument("--name", type=str, help="Override experiment name")
    parser.add_argument(
        "--override",
        type=str,
        action="append",
        help="Override config parameter (can be used multiple times). Format: key=value",
    )

    # Options
    parser.add_argument("--test", action="store_true", help="Run in test mode (shortened training for verification)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--list-presets", action="store_true", help="List available presets and exit")
    parser.add_argument("--list-experiments", action="store_true", help="List previous experiments and exit")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Handle list commands
    if args.list_presets:
        presets_dir = Path("configs/presets")
        if presets_dir.exists():
            presets = [f.stem for f in presets_dir.glob("*.yaml")]
            print("Available presets:")
            for preset in presets:
                print(f"  - {preset}")
        else:
            print("No presets directory found")
        sys.exit(0)

    if args.list_experiments:
        experiments = ExperimentRunner.list_experiments()
        if experiments:
            print("Previous experiments:")
            for exp in experiments:
                print(f"  - {exp}")
        else:
            print("No previous experiments found")
        sys.exit(0)

    # Require preset or config for actual training
    if not args.preset and not args.config:
        parser.error("one of the arguments --preset --config is required (or use --list-presets / --list-experiments)")

    # Load configuration
    try:
        if args.preset:
            config_path = Path("configs/presets") / f"{args.preset}.yaml"
            if not config_path.exists():
                print(f"Error: Preset '{args.preset}' not found at {config_path}")
                print("Available presets:")
                presets_dir = Path("configs/presets")
                if presets_dir.exists():
                    for f in presets_dir.glob("*.yaml"):
                        print(f"  - {f.stem}")
                sys.exit(1)

            logging.info(f"Loading preset: {args.preset}")
            config = ExperimentConfig.from_yaml(str(config_path))
        else:
            logging.info(f"Loading config: {args.config}")
            config = ExperimentConfig.from_yaml(args.config)

        # Apply overrides
        if args.name:
            config.name = args.name
            logging.info(f"Overriding name: {args.name}")

        if args.override:
            for override_str in args.override:
                key, value = parse_override(override_str)
                config.update(**{key: value})
                logging.info(f"Override: {key} = {value}")

        # Validate config
        config.validate()

        # Create and run experiment
        runner = ExperimentRunner(config)

        logging.info("=" * 70)
        logging.info(f"Starting experiment: {config.name}")
        logging.info(f"  Reward: {config.reward_type}")
        logging.info(f"  Model: {config.model_type}")
        logging.info(f"  Algorithm: {config.algorithm}")
        if args.test:
            logging.info("  Mode: TEST (shortened)")
        logging.info("=" * 70)

        success = runner.run(test_mode=args.test)

        if success:
            logging.info("=" * 70)
            logging.info("Experiment completed successfully!")
            logging.info(f"Results saved to: {runner.experiment_dir}")
            logging.info("=" * 70)
            sys.exit(0)
        else:
            logging.error("Experiment failed")
            sys.exit(1)

    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logging.error(f"Configuration error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logging.info("\nExperiment interrupted by user")
        sys.exit(130)
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
