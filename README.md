# AMBRCT

## 🚁 Overview
A high-fidelity drone tracking simulation environment based on Blender rendering, featuring advanced camera control and multiple tracking scenarios. This repository extends the original [cranfield-drone-tracking-gym](https://github.com/mazqtpopx/cranfield-drone-tracking-gym).
This enhanced Drone Tracking Gym is modified to introduce dynamic illumination variations as the primary domain shift, serving as a dedicated testbed for evaluating tracking algorithm robustness and generalization performance under changing lighting conditions in domain generalization research.
And this repo provides the proposed method AM-BR-CT for color transfer in RL for UAV tracking.

## 📋 Requirements

This project requires two separate Python environments:

### 1. Code Runtime Environment
The main Python environment for running the training and testing scripts:
```bash
conda env create -f environment.yml
conda activate <env_name>
# Or using pip
pip install -r requirements.txt
```

### 2. Blender Python Environment
Blender comes with its own Python interpreter that needs additional packages:
```bash
# Install blender dependencies
pip install -r blender_requirements.txt
```

**Note:** Make sure Blender is properly installed and accessible in your system PATH.

## 🎯 Key Features

### Scenario Configuration
Configure different tracking scenarios by modifying `Scenario.py`:

```python
# Available scenarios:
# "basic_tracking"      - Simple tracking with static drone
# "dynamic_tracking"    - Dynamic drone movement patterns
# "obstacle_tracking"   - Tracking with environmental obstacles

DRONE_SCENARIO = "dynamic_tracking"  # Choose your scenario
```

### Environment Configuration
Select different environment variants in `env_id.py`:

```python
# Available variants:
# "ori"       - Original environment
# "hue60" or "hue60m"    - Hue-shifted environment walm(hue60) and cool(hue60m)

# Current version: AM-BR-CT
ENV_NAME = "hue60"  # Choose your environment variant
```

### Advanced Parameters
Configure performance parameters:

```python
# Parameter options:
# NUM_REF: 1, 100, 500, 5000
# ALPHA: 0, 0.1, 0.3, 0.5, 0.7, 0.9

NUM_REF = 5000  # Number of reference samples
ALPHA = 0.7     # Mixing parameter
```

## 🚀 Quick Start

### Generate Reference Images
```bash
python generate_reference_images.py
```

### Test Model
```bash
# Standard testing
python test_model_step.py

# Testing with MKL optimization
python test_model_step_w_mkl.py
```

### Run with Batch Scripts
```bash
# Without MKL
run_wo.bat

# With MKL optimization
run_w.bat
```

**Important:** Before running tests, update `MODEL_PATH` in both `test_model_step.py` and `test_model_step_w_mkl.py` to point to your trained model.

## 📁 Project Structure

```
env_code_git_public/
├── blender_requirements.txt     # Blender Python dependencies
├── drone_tracking_gym/          # Core environment package
│   ├── envs/                    # Environment implementations
│   │   ├── CameraControl.py     # Camera control logic
│   │   ├── DroneControl.py      # Drone control logic
│   │   ├── drone_tracking.blend.py
│   │   ├── drone_tracking_env.py
│   │   ├── drone_tracking_ptz.blend.py
│   │   ├── drone_tracking_ptz_env.py
│   │   ├── lighting_variants.py # Lighting configurations
│   │   ├── OffscreenRenderer.py # Blender offscreen rendering
│   │   ├── Reward.py           # Reward functions
│   │   ├── Scenario.py         # Scenario configurations
│   │   ├── SecondOrderSystem.py
│   │   ├── Viz_System_Response.py
│   │   └── __init__.py
│   ├── register_extended_envs.py
│   └── __init__.py
├── environment.yml              # Conda environment file
├── env_id.py                    # Environment selection
├── generate_reference_images.py # Reference image generation
├── object_tracking_PPO_LSTM.py  # PPO-LSTM training script
├── requirements.txt             # Python dependencies
├── run_w.bat                    # Run with MKL
├── run_wo.bat                   # Run without MKL
├── test_model_step.py           # Model testing
└── test_model_step_w_mkl.py     # Model testing with MKL
```

## 📚 References

- [Cranfield Drone Tracking Gym](https://github.com/mazqtpopx/cranfield-drone-tracking-gym) - Original implementation
- [PyTorch Blender Integration](https://github.com/cheind/pytorch-blender) - Blender rendering utilities
