# TMRL Codebase Architecture & Refactoring Plan

**Branch:** `andrew/add-experiment-framework`  
**Goal:** Enable rapid experimentation with reward functions, neural architectures, and RL algorithms  
**Date:** December 27, 2025  
**Status:** ✅ IMPLEMENTATION COMPLETE

---

## Quick Start (New Experiment Framework)

```bash
# List available presets
python train.py --list-presets

# Run with baseline configuration
python train.py --preset baseline

# Run with time-optimal reward (speed focused)
python train.py --preset time_optimal

# Run with racing line optimization (apex detection)
python train.py --preset racing_line

# Custom name for experiment
python train.py --preset time_optimal --name exp_001

# Override specific parameters
python train.py --preset baseline --override reward_params.time_pressure_weight=0.8

# List previous experiments
python train.py --list-experiments

# Compare experiments
python compare.py --list
python compare.py --experiments exp1 exp2
```

---

## Table of Contents

1. [Current System Architecture](#current-system-architecture)
2. [Component Mapping](#component-mapping)
3. [Data Flow](#data-flow)
4. [Configuration System](#configuration-system)
5. [Critical Files](#critical-files)
6. [Refactoring Plan](#refactoring-plan)
7. [Verification Checklist](#verification-checklist)

---

## Current System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    TMRL Distributed System                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐         ┌──────────────┐                  │
│  │ RolloutWorker│◄───────►│    Server    │                  │
│  │ (Collects    │         │  (Relay Hub) │                  │
│  │  samples)    │         └──────────────┘                  │
│  └──────────────┘                 ▲                          │
│                                   │                          │
│  ┌──────────────┐                 │                          │
│  │ RolloutWorker│◄────────────────┘                          │
│  │  (Worker 2)  │                 │                          │
│  └──────────────┘                 │                          │
│                                   ▼                          │
│                          ┌──────────────┐                    │
│                          │   Trainer    │                    │
│                          │ (Updates NN) │                    │
│                          └──────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

#### 1. **RolloutWorker** (`tmrl/networking.py:450-895`)
- **Purpose**: Collect training samples by interacting with environment
- **Key Methods**:
  - `run()`: Main collection loop
  - `step()`: Take action, get observation/reward
  - `reset()`: Reset environment between episodes
  - `send_and_clear_buffer()`: Send samples to server
  - `update_actor_weights()`: Receive new policy from server

#### 2. **Server** (`tmrl/networking.py:113-245`)
- **Purpose**: Central relay between workers and trainer
- **Key Methods**:
  - Accepts connections from workers and trainer
  - Buffers samples from workers
  - Broadcasts policy updates to workers

#### 3. **Trainer** (`tmrl/networking.py:963-1027`)
- **Purpose**: Train neural network, send updates to server
- **Key Methods**:
  - `run()`: Main training loop
  - Receives samples from server
  - Updates policy via `TrainingAgent.train()`
  - Sends updated weights back to server

#### 4. **TrainingAgent** (e.g., SAC in `tmrl/custom/custom_algorithms.py:25-195`)
- **Purpose**: Implements RL algorithm (SAC, REDQ-SAC, etc.)
- **Key Methods**:
  - `train(batch)`: Execute one training step
  - `get_actor()`: Return policy to broadcast

#### 5. **Environment** (`tmrl/envs.py`, `tmrl/custom/tm/tm_gym_interfaces.py`)
- **Purpose**: Wrap TrackMania as a Gymnasium environment
- **Key Classes**:
  - `TM2020Interface`: Base interface for full images
  - `TM2020InterfaceLidar`: LIDAR-based observations
  - `GenericGymEnv`: Wrapper making it Gymnasium-compatible

#### 6. **RewardFunction** (`tmrl/custom/tm/utils/compute_reward.py`)
- **Purpose**: Compute rewards from game state
- **Current Implementation**: Waypoint-based progress tracking

---

## Component Mapping

### File Organization (Current)

```
tmrl/
├── Core Framework
│   ├── actor.py                    # ActorModule interface (policy wrapper)
│   ├── training.py                 # TrainingAgent interface
│   ├── training_offline.py         # TorchTrainingOffline (main training loop)
│   ├── memory.py                   # Memory (replay buffer) interface
│   ├── envs.py                     # GenericGymEnv wrapper
│   ├── networking.py               # Server, RolloutWorker, Trainer
│   └── util.py                     # Utilities (partial, collate_torch, etc.)
│
├── Configuration
│   └── config/
│       ├── config_constants.py     # Load config.json, define constants
│       └── config_objects.py       # Build actual objects from config
│
├── Custom Implementations (THIS IS WHERE EVERYTHING LIVES)
│   └── custom/
│       ├── custom_algorithms.py    # SAC, REDQ-SAC implementations
│       ├── custom_models.py        # All neural networks (MLP, CNN, etc.)
│       ├── custom_memories.py      # Memory implementations for TM
│       ├── custom_checkpoints.py   # Save/load logic
│       └── tm/                     # TrackMania-specific code
│           ├── tm_gym_interfaces.py    # Environment interfaces
│           ├── tm_preprocessors.py     # Observation preprocessing
│           └── utils/
│               ├── compute_reward.py   # REWARD FUNCTION (480 lines!)
│               ├── control_gamepad.py  # Virtual gamepad control
│               ├── control_keyboard.py # Keyboard control
│               ├── tools.py            # TM utilities (LIDAR, OpenPlanet)
│               └── window.py           # Window management
│
├── Tools
│   └── tools/
│       ├── check_environment.py    # Environment validation
│       ├── record.py               # Record demonstrations
│       └── init_package/           # Setup scripts
│
└── Entry Points
    ├── __main__.py                 # CLI entry (python -m tmrl)
    └── __init__.py                 # Library entry (from tmrl import ...)
```

### Key Imports and Dependencies

```python
# How components connect:

# 1. Configuration loads everything
config_constants.py
    └─> Reads ~/TmrlData/config/config.json
    └─> Defines RUN_NAME, CUDA_TRAINING, ENV_CONFIG, etc.

config_objects.py
    └─> Imports from custom_algorithms, custom_models, custom_memories
    └─> Creates partial() instances: AGENT, TRAINER, MEMORY, POLICY
    └─> Based on config.json settings (LIDAR vs images, SAC vs REDQ)

# 2. Main entry point assembles everything
__main__.py
    └─> Imports config_objects (TRAINER, POLICY, CONFIG_DICT, etc.)
    └─> Creates RolloutWorker or Trainer based on CLI args
    └─> Passes partial() objects that get instantiated at runtime

# 3. Training loop
networking.py: Trainer.run()
    └─> Calls iterate_epochs()
        └─> Instantiates run_cls (TorchTrainingOffline from config)
            └─> TorchTrainingOffline.run_epoch()
                └─> Calls agent.train(batch) repeatedly
                └─> Calls interface.broadcast_model(agent.get_actor())

# 4. Data collection loop  
networking.py: RolloutWorker.run()
    └─> Creates env_cls() (GenericGymEnv with TM interface)
    └─> Creates actor (SquashedGaussianMLPActor or CNN variant)
    └─> Loop: obs -> actor.act(obs) -> env.step(action) -> buffer.append()
    └─> Periodically: send buffer to server, receive new weights

# 5. Environment interaction
GenericGymEnv (envs.py)
    └─> Wraps rtgym environment
        └─> TM2020Interface/TM2020InterfaceLidar (tm_gym_interfaces.py)
            └─> get_obs_rew_terminated_info()
                └─> Captures screenshot/LIDAR
                └─> Calls RewardFunction.compute_reward()
                └─> Returns (obs, reward, terminated, info)
```

---

## Data Flow

### 1. Training Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│ RolloutWorker                                                 │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│ Environment ──┐                                                │
│    │          │ obs                                            │
│    │          ▼                                                │
│    │      Actor (policy)                                       │
│    │          │ action                                         │
│    │          ▼                                                │
│    │      Environment.step()                                   │
│    │          │                                                │
│    ├─────────►│ Screenshot/LIDAR                               │
│    │          │                                                │
│    │          ▼                                                │
│    │      RewardFunction.compute_reward()                      │
│    │          │                                                │
│    │          │ reward, terminated                             │
│    ▼          ▼                                                │
│  (obs, action, reward, next_obs, terminated) -> Local Buffer  │
│                                   │                            │
│                                   │ Send periodically          │
│                                   ▼                            │
└───────────────────────────────────┼────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────┐
│ Server                                                        │
├──────────────────────────────────────────────────────────────┤
│  Accumulate samples from all workers in central buffer        │
│                                   │                            │
│                                   │ Forward to trainer         │
│                                   ▼                            │
└───────────────────────────────────┼────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────┐
│ Trainer                                                        │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  Samples -> Memory (replay buffer)                             │
│                │                                               │
│                │ Sample batches                                │
│                ▼                                               │
│            TrainingAgent.train(batch)                          │
│                │                                               │
│                │ (SAC or REDQ algorithm)                       │
│                │                                               │
│                │ Update actor & critic networks                │
│                ▼                                               │
│            TrainingAgent.get_actor()                           │
│                │                                               │
│                │ Broadcast new weights                         │
│                ▼                                               │
│            Server -> RolloutWorkers                            │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 2. Configuration Flow

```
config.json (~/TmrlData/config/config.json)
    │
    ├─> RUN_NAME, CUDA_TRAINING, ENV settings
    ├─> ALG settings (lr, gamma, etc.)
    ├─> MEMORY_SIZE, BATCH_SIZE
    │
    ▼
config_constants.py (loads config.json)
    │
    ├─> Defines constants: RUN_NAME, CUDA_TRAINING, etc.
    ├─> Computes: PRAGMA_LIDAR, MODEL_PATH_WORKER, etc.
    │
    ▼
config_objects.py (builds objects)
    │
    ├─> Selects model: MLPActorCritic vs VanillaCNNActorCritic
    ├─> Selects interface: TM2020InterfaceLidar vs TM2020Interface
    ├─> Selects memory: MemoryTMLidar vs MemoryTMFull
    ├─> Selects algorithm: SAC_Agent vs REDQ_Agent
    │
    ├─> Creates partial() objects:
    │   ├─> POLICY = partial(SquashedGaussianMLPActor)
    │   ├─> TRAIN_MODEL = partial(MLPActorCritic)
    │   ├─> AGENT = partial(SAC_Agent, lr_actor=..., gamma=...)
    │   ├─> MEMORY = partial(MemoryTMLidar, batch_size=...)
    │   ├─> TRAINER = partial(TorchTrainingOffline, epochs=...)
    │   └─> ENV_CLS = partial(GenericGymEnv, config=...)
    │
    ▼
__main__.py or user script
    │
    ├─> RolloutWorker(actor_module_cls=POLICY, ...)
    │       └─> Instantiates policy when created
    │
    └─> Trainer(training_cls=TRAINER, ...)
            └─> Instantiates TRAINER which creates AGENT, MEMORY, etc.
```

---

## Configuration System

### Current Config Location

**File:** `~/TmrlData/config/config.json`

This file controls EVERYTHING. Changes require:
1. Edit JSON file
2. Restart training/worker
3. No validation until runtime

### Current Config Structure

```json
{
  "__VERSION__": "0.6.0",
  "RUN_NAME": "SAC_4_LIDAR_baseline",
  
  "ENV": {
    "RTGYM_INTERFACE": "TM20LIDAR",  // or "TM20FULL", "TM20IMAGES"
    "WINDOW_WIDTH": 958,
    "IMG_HIST_LEN": 4,
    "RTGYM_CONFIG": { ... },
    "REWARD_CONFIG": {
      "END_OF_TRACK": 100.0,
      "CHECK_FORWARD": 500,
      "FAILURE_COUNTDOWN": 10,
      ...
    }
  },
  
  "ALG": {
    "ALGORITHM": "SAC",  // or "REDQSAC"
    "LR_ACTOR": 0.0003,
    "LR_CRITIC": 0.00005,
    "GAMMA": 0.995,
    ...
  },
  
  "MEMORY_SIZE": 1000000,
  "BATCH_SIZE": 256,
  "MAX_EPOCHS": 10000,
  ...
}
```

### How Config is Used

```python
# Step 1: Load JSON
# File: tmrl/config/config_constants.py
CONFIG_FILE = Path.home() / "TmrlData" / "config" / "config.json"
with open(CONFIG_FILE) as f:
    TMRL_CONFIG = json.load(f)

# Step 2: Extract values
RUN_NAME = TMRL_CONFIG["RUN_NAME"]
ENV_CONFIG = TMRL_CONFIG["ENV"]
RTGYM_INTERFACE = ENV_CONFIG["RTGYM_INTERFACE"]
PRAGMA_LIDAR = RTGYM_INTERFACE.endswith("LIDAR")

# Step 3: Build objects based on config
# File: tmrl/config/config_objects.py
if PRAGMA_LIDAR:
    TRAIN_MODEL = MLPActorCritic
    POLICY = SquashedGaussianMLPActor
    INT = partial(TM2020InterfaceLidar, ...)
else:
    TRAIN_MODEL = VanillaCNNActorCritic
    POLICY = SquashedGaussianVanillaCNNActor
    INT = partial(TM2020Interface, ...)

# Step 4: Use in main
# File: tmrl/__main__.py
rw = RolloutWorker(
    env_cls=partial(GenericGymEnv, config=CONFIG_DICT),
    actor_module_cls=POLICY,  # From config_objects
    ...
)
```

---

## Critical Files

### Core Training Files (Must Understand)

| File | Lines | Purpose | Key Classes/Functions |
|------|-------|---------|----------------------|
| `tmrl/training_offline.py` | 163 | Main training loop | `TorchTrainingOffline`, `run_epoch()` |
| `tmrl/networking.py` | 1027 | Server/Worker/Trainer | `Server`, `RolloutWorker`, `Trainer`, `iterate_epochs()` |
| `tmrl/actor.py` | 153 | Actor interface | `ActorModule` (base class) |
| `tmrl/training.py` | 45 | Training interface | `TrainingAgent` (base class) |
| `tmrl/memory.py` | 218 | Memory interface | `Memory` (base class) |

### Custom Implementations (What We'll Refactor)

| File | Lines | Purpose | What's Inside |
|------|-------|---------|---------------|
| `custom/custom_algorithms.py` | 426 | RL algorithms | `SpinupSacAgent`, `REDQSACAgent` |
| `custom/custom_models.py` | 827 | Neural networks | MLP, CNN, RNN, LIDAR, Image models (ALL MIXED) |
| `custom/custom_memories.py` | 212 | Replay buffers | `MemoryTMLidar`, `MemoryTMFull` |
| `custom/tm/tm_gym_interfaces.py` | 494 | TM environments | `TM2020Interface`, `TM2020InterfaceLidar` |
| `custom/tm/utils/compute_reward.py` | 480 | **REWARD FUNCTION** | `RewardFunction.compute_reward()` |

### Configuration Files

| File | Lines | Purpose |
|------|-------|---------|
| `config/config_constants.py` | 131 | Load config.json, define constants |
| `config/config_objects.py` | 191 | Build objects from config |
| `~/TmrlData/config/config.json` | ~200 | **USER-FACING CONFIG** |

---

## Refactoring Plan

### Goals

1. **Modular reward functions** - Easy to create/swap variants
2. **Flexible model selection** - Simple config changes for architectures
3. **Rapid experimentation** - Change config, run, compare results
4. **No breaking changes** - Existing code continues to work
5. **Reproducibility** - Every experiment saves its exact config

### Phase 1: Core Infrastructure (Priority 1)

**Estimated Time:** 4-6 hours

#### 1.1 Create Experiment Framework

```
experiments/
├── __init__.py
├── config.py          # NEW - ExperimentConfig dataclass
├── factories.py       # NEW - RewardFactory, ModelFactory, AlgorithmFactory
├── runner.py          # NEW - ExperimentRunner
└── comparison.py      # NEW - Compare experiment results
```

**Files to Create:**

```python
# experiments/config.py (~80 lines)
@dataclass
class ExperimentConfig:
    name: str
    reward_type: str = "baseline"
    model_type: str = "mlp"
    algorithm: str = "sac"
    reward_params: dict = field(default_factory=dict)
    model_params: dict = field(default_factory=dict)
    training_params: dict = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, path): ...
    
    def to_yaml(self, path): ...
```

```python
# experiments/factories.py (~120 lines)
class RewardFactory:
    @staticmethod
    def create(config: ExperimentConfig) -> RewardFunction:
        if config.reward_type == "baseline":
            return RewardFunction(**config.reward_params)
        elif config.reward_type == "time_optimal":
            return TimeOptimalReward(**config.reward_params)
        elif config.reward_type == "racing_line":
            return RacingLineReward(**config.reward_params)
        # Add more as needed

class ModelFactory: ...
class AlgorithmFactory: ...
```

```python
# experiments/runner.py (~100 lines)
class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = Path(f"experiments/runs/{config.name}")
    
    def run(self):
        # 1. Save config
        # 2. Create components via factories
        # 3. Run training
        # 4. Save results
```

#### 1.2 Create Config System

```
configs/
├── base.yaml          # Default values (current system)
└── presets/
    ├── baseline.yaml      # Current system as-is
    ├── time_optimal.yaml  # Time-pressure variant
    └── racing_line.yaml   # Apex-aware variant
```

#### 1.3 Create New Entry Point

```python
# train.py (NEW - ~40 lines)
"""
Clean entry point for experiments
"""
import argparse
from experiments.config import ExperimentConfig
from experiments.runner import ExperimentRunner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to config YAML')
    parser.add_argument('--preset', help='Use preset config')
    parser.add_argument('--name', help='Experiment name')
    args = parser.parse_args()
    
    # Load config
    if args.preset:
        config = ExperimentConfig.from_yaml(f"configs/presets/{args.preset}.yaml")
    else:
        config = ExperimentConfig.from_yaml(args.config)
    
    # Run
    runner = ExperimentRunner(config)
    runner.run()
```

### Phase 2: Reward Function Refactoring (Priority 2)

**Estimated Time:** 3-4 hours

#### 2.1 Refactor Reward Functions

```
tmrl/custom/tm/utils/
├── compute_reward.py          # EXISTING - keep base class
└── reward_variants.py         # NEW - add subclasses
```

**Changes:**

```python
# In compute_reward.py (ADD ~30 lines, don't change existing)
class RewardFunction:  # EXISTING - unchanged
    def compute_reward(self, pos, data):
        # ... existing code
        pass

# NEW subclasses (in same file or separate file)
class TimeOptimalReward(RewardFunction):
    """Adds time pressure to encourage speed"""
    def compute_reward(self, pos, data):
        base_reward, terminated = super().compute_reward(pos, data)
        # Add time bonus/penalty
        time_pressure = (self.cur_idx - self.step_counter * 1.0) * 0.5
        return base_reward + time_pressure, terminated

class RacingLineReward(RewardFunction):
    """Adds apex detection and curvature-aware speed targets"""
    def __init__(self, *args, apex_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.apex_weight = apex_weight
        self._detect_apexes()  # Pre-compute apex locations
    
    def compute_reward(self, pos, data):
        base_reward, terminated = super().compute_reward(pos, data)
        # Add apex bonus
        apex_bonus = self._compute_apex_bonus(pos)
        # Add speed-for-curvature reward
        speed_reward = self._compute_speed_reward(data[0])
        return base_reward + apex_bonus + speed_reward, terminated
```

#### 2.2 Update Factory

```python
# experiments/factories.py
class RewardFactory:
    @staticmethod
    def create(config):
        # Import here to avoid circular deps
        from tmrl.custom.tm.utils.compute_reward import (
            RewardFunction, 
            TimeOptimalReward, 
            RacingLineReward
        )
        
        if config.reward_type == "baseline":
            return RewardFunction(**config.reward_params)
        elif config.reward_type == "time_optimal":
            return TimeOptimalReward(**config.reward_params)
        elif config.reward_type == "racing_line":
            return RacingLineReward(**config.reward_params)
```

### Phase 3: Integration (Priority 3)

**Estimated Time:** 2-3 hours

#### 3.1 Connect to Existing System

The new framework needs to integrate with existing `config_objects.py`:

```python
# Option A: Parallel systems (recommended for Phase 1)
# - Keep config_objects.py working as-is
# - New train.py uses experiment framework
# - Old python -m tmrl uses config_objects.py

# Option B: Unified system (later)
# - config_objects.py reads from experiment configs
# - Single entry point for everything
```

#### 3.2 Create Comparison Tools

```python
# experiments/comparison.py (~80 lines)
class ExperimentComparison:
    def __init__(self, experiment_dirs):
        self.experiments = [self._load_experiment(d) for d in experiment_dirs]
    
    def compare_lap_times(self):
        # Generate table of best/mean/std lap times
        pass
    
    def plot_learning_curves(self):
        # Plot episode returns over time
        pass
    
    def generate_report(self):
        # Create HTML report with all comparisons
        pass
```

### Phase 4: Polish & Documentation (Priority 4)

**Estimated Time:** 2-3 hours

- Add validation for configs
- Create example configs with comments
- Update README with new workflow
- Add experiment tracking (optional wandb integration)

---

## Verification Checklist

### After Phase 1 (Infrastructure)

- [ ] Can load experiment config from YAML
- [ ] Can create RewardFunction via factory
- [ ] Can create Model via factory
- [ ] Can create Algorithm via factory
- [ ] Experiment runner saves config to output dir
- [ ] Can run baseline config end-to-end

### After Phase 2 (Rewards)

- [ ] TimeOptimalReward runs without errors
- [ ] RacingLineReward runs without errors
- [ ] Rewards produce sensible values (not NaN, not infinite)
- [ ] Can switch between reward types via config
- [ ] Old reward function still works (backward compatibility)

### After Phase 3 (Integration)

- [ ] Can run experiment via `python train.py --preset baseline`
- [ ] Can run experiment via `python train.py --preset time_optimal`
- [ ] Results are saved in `experiments/runs/`
- [ ] Can compare two experiments
- [ ] Old `python -m tmrl --worker` still works

### After Phase 4 (Polish)

- [ ] Config validation catches errors early
- [ ] Example configs are well-documented
- [ ] README has new usage examples
- [ ] Can generate comparison reports

### Critical Tests (Run These)

```bash
# Test 1: Baseline still works
python -m tmrl --worker  # Old system

# Test 2: New system works
python train.py --preset baseline

# Test 3: New reward works
python train.py --preset time_optimal

# Test 4: Comparison works
python compare.py --experiments exp_001 exp_002

# Test 5: Can override params
python train.py --preset baseline --override reward.speed.weight=1.5
```

---

## Detailed Component Reference

### 1. Actor (Policy)

**Interface:** `tmrl/actor.py`

```python
class ActorModule(ABC):
    def act(self, obs, test=False):
        """Compute action from observation"""
        raise NotImplementedError
    
    def save(self, path):
        """Save to disk"""
    
    def load(self, path, device):
        """Load from disk"""
```

**Implementations:**
- `SquashedGaussianMLPActor` - For LIDAR (MLP)
- `SquashedGaussianVanillaCNNActor` - For images (CNN)
- `SquashedGaussianRNNActor` - With LSTM

**Location:** `tmrl/custom/custom_models.py` (lines 52-150 for MLP variant)

### 2. Training Agent (Algorithm)

**Interface:** `tmrl/training.py`

```python
class TrainingAgent(ABC):
    def train(self, batch):
        """Execute training step, return metrics dict"""
        raise NotImplementedError
    
    def get_actor(self):
        """Return ActorModule to broadcast"""
        raise NotImplementedError
```

**Implementations:**
- `SpinupSacAgent` - SAC algorithm
- `REDQSACAgent` - REDQ-SAC variant

**Location:** `tmrl/custom/custom_algorithms.py`

### 3. Memory (Replay Buffer)

**Interface:** `tmrl/memory.py`

```python
class Memory(ABC):
    def append_buffer(self, buffer):
        """Add samples to memory"""
        raise NotImplementedError
    
    def __iter__(self):
        """Sample batches for training"""
        for _ in range(self.nb_steps):
            yield self.sample()
```

**Implementations:**
- `MemoryTMLidar` - For LIDAR observations
- `MemoryTMFull` - For image observations

**Location:** `tmrl/custom/custom_memories.py`

### 4. Environment Interface

**Base:** `rtgym.RealTimeGymInterface`

**Implementations:**
- `TM2020Interface` - Full images + speed/gear/rpm
- `TM2020InterfaceLidar` - LIDAR + speed
- `TM2020InterfaceLidarProgress` - LIDAR + speed + track progress

**Location:** `tmrl/custom/tm/tm_gym_interfaces.py`

**Key Methods:**
```python
def get_obs_rew_terminated_info(self):
    # Capture screenshot or compute LIDAR
    # Call reward_function.compute_reward()
    # Return (obs, reward, terminated, info)
```

### 5. Reward Function

**Location:** `tmrl/custom/tm/utils/compute_reward.py`

**Current Structure:**

```python
class RewardFunction:
    def __init__(self, 
                 reward_data_path,  # Path to trajectory pickle
                 ws_client,         # WebSocket for graphing
                 nb_obs_forward=10,
                 nb_obs_backward=10,
                 nb_zero_rew_before_failure=10,
                 min_nb_steps_before_failure=70,
                 max_dist_from_traj=60.0,
                 ...):
        # Load trajectory waypoints
        with open(reward_data_path, 'rb') as f:
            self.pathdata = pickle.load(f)
    
    def compute_reward(self, pos, data):
        # pos: [x, y, z] position
        # data: [speed, distance, x, y, z, steer, gas, brake, finished, gear, rpm]
        
        # Find closest waypoint on trajectory
        # Reward = waypoints passed since last step
        # Terminate if stuck or too far from trajectory
        
        return reward, terminated
```

**Reward Calculation:**
1. Find best matching waypoint in trajectory
2. Reward = (best_index - cur_idx) / 100.0
3. If stuck (no progress), increment failure counter
4. If failure_counter > threshold, terminate
5. Optional: Add collision detection, speed bonuses

**Current Mode System (line 422-434):**
```python
mode = 2  # Currently set to mode 2

if mode == 1:    # All rewards
    reward = path + speed + collision
elif mode == 2:  # Path only (CURRENT)
    reward = path
elif mode == 3:  # Path + speed
    reward = path + speed
elif mode == 4:  # Path + collision
    reward = path + collision
```

---

## Implementation Order

### Week 1: Core Framework

**Day 1-2:**
1. Create `experiments/` directory structure
2. Implement `ExperimentConfig` class
3. Implement `RewardFactory`, `ModelFactory`, `AlgorithmFactory`
4. Create `ExperimentRunner` skeleton
5. Test: Can load config, create components

**Day 3:**
1. Create base YAML configs
2. Create 3 presets: baseline, time_optimal, racing_line
3. Create `train.py` entry point
4. Test: Can run baseline experiment

**Day 4:**
1. Implement reward variants (TimeOptimalReward, RacingLineReward)
2. Test reward functions produce sensible values
3. Run full training with new rewards

**Day 5:**
1. Create comparison tools
2. Test comparing multiple experiments
3. Documentation

### Week 2: Advanced Features (Optional)

- Hot-reload for reward weights
- Parallel experiment runner
- WandB integration
- Hyperparameter sweeps

---

## File Changes Summary

### New Files (~500 lines total)

```
experiments/
├── __init__.py               # ~10 lines
├── config.py                 # ~100 lines (ExperimentConfig)
├── factories.py              # ~150 lines (3 factories)
├── runner.py                 # ~120 lines (ExperimentRunner)
└── comparison.py             # ~120 lines (comparison tools)

configs/
├── base.yaml                 # ~100 lines
└── presets/
    ├── baseline.yaml         # ~30 lines
    ├── time_optimal.yaml     # ~30 lines
    └── racing_line.yaml      # ~30 lines

train.py                      # ~40 lines (new entry point)
```

### Modified Files (~50 lines added)

```
tmrl/custom/tm/utils/compute_reward.py
    +30 lines: Add TimeOptimalReward, RacingLineReward subclasses

tmrl/custom/custom_models.py (optional)
    +20 lines: Add LSTMActorCritic (if implementing LSTM)
```

### Unchanged Files (backward compatibility)

```
✓ tmrl/actor.py
✓ tmrl/training.py
✓ tmrl/training_offline.py
✓ tmrl/networking.py
✓ tmrl/envs.py
✓ tmrl/memory.py
✓ tmrl/config/config_constants.py
✓ tmrl/config/config_objects.py
✓ tmrl/__main__.py
✓ All other existing files
```

---

## Common Gotchas & Solutions

### 1. Import Cycles

**Problem:** Circular imports between factories and implementations

**Solution:** Import inside factory methods, not at module level

```python
# BAD
from tmrl.custom.custom_models import MLPActorCritic

class ModelFactory:
    @staticmethod
    def create(config):
        return MLPActorCritic(...)

# GOOD
class ModelFactory:
    @staticmethod
    def create(config):
        from tmrl.custom.custom_models import MLPActorCritic
        return MLPActorCritic(...)
```

### 2. Config Paths

**Problem:** Relative paths break when running from different directories

**Solution:** Always use absolute paths or paths relative to repo root

```python
# Use pathlib
from pathlib import Path
REPO_ROOT = Path(__file__).parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
```

### 3. Reward Function State

**Problem:** Reward function has internal state (cur_idx, step_counter)

**Solution:** Call `reset()` between episodes

```python
# In environment interface
def reset(self):
    self.reward_function.reset()
    # ... rest of reset logic
```

### 4. Device Placement

**Problem:** Model on GPU but data on CPU (or vice versa)

**Solution:** Ensure consistent device placement in factories

```python
class ModelFactory:
    @staticmethod
    def create(config, device):
        model = MLPActorCritic(...)
        return model.to(device)
```

---

## Success Criteria

The refactoring is successful if:

1. **Easy to experiment**: Change config YAML, run training, see results
2. **Fast iteration**: <5 minutes from idea to training start
3. **Reproducible**: Each experiment saves exact config used
4. **Comparable**: Can easily compare lap times across experiments
5. **Extensible**: Adding new reward = add 1 class + 1 config
6. **Backward compatible**: Old code still works
7. **Well documented**: README explains new workflow
8. **Tested**: All verification checklist items pass

---

## Next Steps

1. Review this document
2. Approve refactoring plan
3. Create branch: `andrew/add-experiment-framework`
4. Implement Phase 1 (core infrastructure)
5. Test with baseline config
6. Implement Phase 2 (reward variants)
7. Run experiments, compare results
8. Iterate and improve

---

**Document Version:** 1.0  
**Last Updated:** December 27, 2025  
**Author:** AI Assistant  
**Reviewer:** Andrew Kent

