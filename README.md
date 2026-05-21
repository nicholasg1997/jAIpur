# jAIpur

jAIpur is a reinforcement learning (RL) environment and research toolkit for the board game Jaipur. It features a custom OpenAI Gymnasium-compatible environment, Stable Baselines3-style agent training (with MIMO support via [@adysonmaia/sb3-plus](https://github.com/adysonmaia/sb3-plus)), and is organized for future agent evaluation and human play.

---

## Features

- **Jaipur Gym Environment:**  
  Implements all rules, tokens, and actions for Jaipur with multi-discrete action spaces and rich observations.
- **Agent Training:**  
  Train agents using Stable Baselines3-style APIs and [sb3-plus](https://github.com/adysonmaia/sb3-plus) for multi-output (MIMO) policy support.  
  Training is highly configurable via YAML config files.
- **Extensible:**  
  Designed for easy addition of:
  - Agent-vs-agent or self-play evaluation scripts (coming soon)
  - Human-vs-agent CLI or GUI play (coming soon)

## Installation

1. **Clone this repository:**
    ```bash
    git clone https://github.com/nicholasg1997/jAIpur.git
    cd jAIpur
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    - Requires [Gymnasium](https://github.com/Farama-Foundation/Gymnasium), [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3), and [sb3-plus](https://github.com/adysonmaia/sb3-plus) (by [@adysonmaia](https://github.com/adysonmaia)), which provides MIMO policy support.

## Configuration

- All core environment and game rules are defined in the `configs/` folder.
- The default RL hyperparameters are set in `configs/ppo_configs.yaml`.
- You can specify a different config directory at runtime to override defaults.

## Usage

### Training an Agent

To train an agent with the default configuration:

```bash
python scripts/train_agent.py
```

- The training script loads the default config from `configs/ppo_configs.yaml`.
- To use a custom config, pass the directory or file location as an argument (see script for details).

Trained models are saved to the `models/` directory by default.

## Evaluation Framework

We provide a rollout-based evaluation pipeline for trained agents.

### Evaluate a trained model

```bash
python scripts/evaluate_agent.py --model path/to/model.zip --episodes 50

## Project Structure

```
jaipur_rl/
  envs/         # Gymnasium environment
  game/         # Jaipur game logic
  training/     # Training utilities, callbacks, schedulers, trainer
configs/        # Game and RL configs (ppo_configs.yaml, game_configs.py, etc)
scripts/        # Training entry points
models/         # Saved agents
```

## Acknowledgements

- Jaipur board game by [Space Cowboys](https://www.spacecowboys.fr/jaipur)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- [sb3-plus](https://github.com/adysonmaia/sb3-plus) by [@adysonmaia](https://github.com/adysonmaia) for multi-output policy support

---

Questions, suggestions, or want to contribute?  
Open an issue or pull request on GitHub!

