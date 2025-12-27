"""
Experiment Comparison Tools

Tools for comparing and analyzing multiple experiments.
"""

from pathlib import Path
import json
from typing import List, Optional, Dict
import logging
from datetime import datetime


class ExperimentComparison:
    """
    Compare multiple experiment results.
    """
    
    def __init__(self, experiment_names: List[str], runs_dir: str = "experiments/runs"):
        """
        Initialize comparison with list of experiments.
        
        Args:
            experiment_names: List of experiment names to compare
            runs_dir: Base directory for experiment runs
        """
        self.runs_dir = Path(runs_dir)
        self.experiments = []
        
        for name in experiment_names:
            exp_data = self._load_experiment(name)
            if exp_data:
                self.experiments.append(exp_data)
            else:
                logging.warning(f"Could not load experiment: {name}")
    
    def _load_experiment(self, name: str) -> Optional[Dict]:
        """
        Load experiment data from disk.
        
        Args:
            name: Experiment name
            
        Returns:
            Experiment data dictionary or None
        """
        exp_dir = self.runs_dir / name
        metadata_path = exp_dir / "metadata.json"
        config_path = exp_dir / "config.yaml"
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        data['name'] = name
        data['dir'] = str(exp_dir)
        
        return data
    
    def get_summary_table(self) -> str:
        """
        Generate summary table of experiments.
        
        Returns:
            Formatted string table
        """
        if not self.experiments:
            return "No experiments to compare"
        
        # Table header
        lines = [
            "=" * 80,
            "EXPERIMENT COMPARISON",
            "=" * 80,
            f"{'Name':<30} {'Status':<12} {'Reward Type':<15} {'Duration':<10}",
            "-" * 80
        ]
        
        # Table rows
        for exp in self.experiments:
            name = exp.get('name', 'unknown')[:30]
            status = exp.get('status', 'unknown')[:12]
            
            config = exp.get('config', {})
            reward_type = config.get('reward_type', 'unknown')[:15]
            
            duration = exp.get('duration')
            if duration:
                duration_str = f"{duration:.1f}s"
            else:
                duration_str = "N/A"
            
            lines.append(f"{name:<30} {status:<12} {reward_type:<15} {duration_str:<10}")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def get_config_comparison(self) -> str:
        """
        Compare configurations across experiments.
        
        Returns:
            Formatted string with config differences
        """
        if not self.experiments:
            return "No experiments to compare"
        
        lines = [
            "=" * 80,
            "CONFIGURATION COMPARISON",
            "=" * 80
        ]
        
        for exp in self.experiments:
            config = exp.get('config', {})
            lines.append(f"\n{exp['name']}:")
            lines.append(f"  reward_type: {config.get('reward_type', 'N/A')}")
            lines.append(f"  model_type: {config.get('model_type', 'N/A')}")
            lines.append(f"  algorithm: {config.get('algorithm', 'N/A')}")
            
            # Show key reward params
            reward_params = config.get('reward_params', {})
            if reward_params:
                lines.append(f"  reward_params:")
                for key, value in list(reward_params.items())[:5]:
                    lines.append(f"    {key}: {value}")
            
            # Show key training params
            training_params = config.get('training_params', {})
            if training_params:
                lines.append(f"  training_params:")
                for key in ['lr_actor', 'lr_critic', 'gamma', 'batch_size']:
                    if key in training_params:
                        lines.append(f"    {key}: {training_params[key]}")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive comparison report.
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Report string
        """
        report_lines = [
            "TMRL Experiment Comparison Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Experiments compared: {len(self.experiments)}",
            "",
            self.get_summary_table(),
            "",
            self.get_config_comparison()
        ]
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logging.info(f"Report saved to: {output_path}")
        
        return report
    
    @staticmethod
    def list_all_experiments(runs_dir: str = "experiments/runs") -> List[str]:
        """
        List all available experiments.
        
        Args:
            runs_dir: Base directory for experiment runs
            
        Returns:
            List of experiment names
        """
        runs_path = Path(runs_dir)
        if not runs_path.exists():
            return []
        
        experiments = []
        for exp_dir in runs_path.iterdir():
            if exp_dir.is_dir():
                metadata_path = exp_dir / "metadata.json"
                if metadata_path.exists():
                    experiments.append(exp_dir.name)
        
        return sorted(experiments)
    
    @staticmethod
    def get_last_n_experiments(n: int = 5, runs_dir: str = "experiments/runs") -> List[str]:
        """
        Get the last N experiments by modification time.
        
        Args:
            n: Number of experiments to return
            runs_dir: Base directory for experiment runs
            
        Returns:
            List of experiment names (most recent first)
        """
        runs_path = Path(runs_dir)
        if not runs_path.exists():
            return []
        
        exp_times = []
        for exp_dir in runs_path.iterdir():
            if exp_dir.is_dir():
                metadata_path = exp_dir / "metadata.json"
                if metadata_path.exists():
                    mtime = metadata_path.stat().st_mtime
                    exp_times.append((exp_dir.name, mtime))
        
        # Sort by modification time (most recent first)
        exp_times.sort(key=lambda x: x[1], reverse=True)
        
        return [name for name, _ in exp_times[:n]]


def compare_experiments_cli():
    """
    CLI interface for comparing experiments.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare TMRL experiments")
    
    parser.add_argument(
        '--experiments', '-e',
        nargs='+',
        help='Experiment names to compare'
    )
    parser.add_argument(
        '--last', '-n',
        type=int,
        help='Compare last N experiments'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all experiments'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for report'
    )
    
    args = parser.parse_args()
    
    if args.list:
        experiments = ExperimentComparison.list_all_experiments()
        if experiments:
            print("Available experiments:")
            for exp in experiments:
                print(f"  - {exp}")
        else:
            print("No experiments found")
        return
    
    if args.last:
        experiment_names = ExperimentComparison.get_last_n_experiments(args.last)
        if not experiment_names:
            print("No experiments found")
            return
    elif args.experiments:
        experiment_names = args.experiments
    else:
        print("Please specify experiments to compare (--experiments or --last)")
        return
    
    comparison = ExperimentComparison(experiment_names)
    report = comparison.generate_report(args.output)
    print(report)


if __name__ == "__main__":
    compare_experiments_cli()

