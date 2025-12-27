#!/usr/bin/env python3
"""
TMRL Experiment Comparison Script

Compare multiple training experiments.

Usage:
    python compare.py --list
    python compare.py --last 5
    python compare.py --experiments exp1 exp2 exp3
    python compare.py --last 3 --output report.txt
"""

import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

from experiments.comparison import compare_experiments_cli

if __name__ == "__main__":
    compare_experiments_cli()
