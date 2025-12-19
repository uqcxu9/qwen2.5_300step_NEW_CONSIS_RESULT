# Qwen2.5 Economic Agent GRPO Training Results

This repository contains the code, training results, and simulation outputs for the Qwen2.5-7B economic agent trained with GRPO (Generalized Advantage Estimation with Proximal Policy Optimization).

## Directory Structure

```
├── RL/                          # RL training code
│   ├── reward.py               # Reward function
│   ├── prepare_verl_data.py    # Data preparation script
│   └── config/                 # Training configs
│       └── econ_grpo_small.yaml
├── data/
│   ├── verl_dataset_small/     # Training/validation data (parquet)
│   └── gpt-3-noperception-reflection-1-100agents-240months/
│       ├── dense_log.pkl       # Final simulation log
│       ├── env_*.pkl           # Environment states
│       ├── result_analysis/    # Analysis results (CSV)
│       └── constraint_check/   # Economic constraint checks
├── checkpoints_v11/
│   └── global_step_350/
│       └── actor/lora_adapter/ # LoRA adapter weights
├── analysis_plots/             # Training & analysis plots
├── logs/                       # Training & simulation logs
├── ai_economist/               # AI Economist framework
├── simulate.py                 # Simulation script
├── simulate_utils.py           # Simulation utilities
├── merge_lora.py              # LoRA merge script
└── config.yaml                # Environment config
```

## Training Configuration

- **Base Model**: Qwen2.5-7B-Instruct
- **Training Steps**: 700 (checkpoint at step 350)
- **LoRA**: r=64, alpha=16
- **Batch Size**: 2
- **Rollout.n**: 4
- **Entropy Coeff**: 0.10

## Key Files

### Reward Function (`RL/reward.py`)
- Poor-lazy penalty for work
- Regime-adjusted consumption targets
- Anti-plateau penalty for consumption diversity

### Simulation Results
- 100 agents, 240 months (20 years)
- Full dialog logs and dense state logs included

## Usage

### 1. Merge LoRA to Base Model
```bash
python merge_lora.py \
  --base_model /path/to/Qwen2.5-7B-Instruct \
  --lora_path checkpoints_v11/global_step_350/actor/lora_adapter \
  --output_path /path/to/merged_model
```

### 2. Run Simulation
```bash
python simulate.py \
  --policy_model gpt \
  --model_type qwen \
  --num_agents 100 \
  --episode_length 240 \
  --seed 42
```

### 3. Continue Training
```bash
cd RL
python -m verl.trainer.main_ppo \
  --config-path config \
  --config-name econ_grpo_small
```

## Requirements

See `GRPO_requirements.txt` for training dependencies.
See `requirements.txt` for simulation dependencies.

## Notes

- Large files (*.pkl, *.parquet, *.safetensors) are tracked with Git LFS
- Checkpoint step 350 is the best performing checkpoint
- Step 700 checkpoint was corrupted during save

