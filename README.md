# Tower Defense Reinforcement Learning

This repository contains a reinforcement learning implementation for a Tower Defense game using Deep Q-Networks (DQN) with prioritized experience replay. The game environment and the agent are built using Python, Pygame, NumPy, and TensorFlow.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training](#training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Tower Defense is a classic strategy game where the player must defend against waves of enemies by strategically placing defensive towers. This project implements a reinforcement learning agent that learns to play the game effectively by placing and upgrading towers, and using special abilities.

## Features

- **Tower Defense Environment**: A custom environment built with Pygame.
- **Deep Q-Network (DQN) Agent**: An agent that uses DQN with prioritized experience replay to learn optimal strategies.
- **Prioritized Experience Replay**: Enhances the learning process by focusing on the most significant experiences.
- **Visualization**: Real-time rendering of the game state and agent actions.
- **Hyperparameter Tuning**: Tools to experiment with different hyperparameters to find the best configuration.

## Installation

To run this project, you need to have Python installed on your machine. You can install the required dependencies using pip:

```bash
pip install numpy pygame tensorflow matplotlib
```

## Usage

1. **Test the Environment**: Run the environment to ensure everything is set up correctly.

```bash
python tower_defense_rl.py
```

2. **Train the Agent**: Start training the agent with default or custom hyperparameters.

```bash
python tower_defense_rl.py --train --alpha 0.6 --beta_start 0.4
```

3. **Visualize Training**: The environment renders the game state and agent actions during training. You can adjust the rendering frequency by changing the `RENDER_EVERY` parameter.

## Configuration

The configuration parameters are defined at the beginning of the script. You can adjust these parameters to customize the game environment and the training process.

- **Game Configuration**:
  - `WIDTH`, `HEIGHT`: Dimensions of the game window.
  - `GRID_SIZE`: Size of the game grid.
  - `CELL_SIZE`: Size of each cell in the grid.
  - `NUM_EPISODES`: Number of training episodes.
  - `BATCH_SIZE`: Batch size for experience replay.
  - `LEARNING_RATE`: Learning rate for the neural network.
  - `GAMMA`: Discount factor for future rewards.
  - `EPSILON_START`, `EPSILON_MIN`, `EPSILON_DECAY_RATE`: Parameters for epsilon-greedy exploration.
  - `TARGET_UPDATE_FREQ`: Frequency of updating the target network.
  - `MODEL_PATH`: Path to save the trained model.
  - `RENDER_EVERY`: Frequency of rendering the game state.
  - `PRIORITIZED_REPLAY_EPS`: Small constant for prioritized experience replay.
  - `MEMORY_SIZE_START`, `MEMORY_SIZE_MAX`: Initial and maximum sizes of the replay buffer.

- **Tower and Enemy Types**:
  - `TOWER_TYPES`: List of available tower types.
  - `ENEMY_TYPES`: List of available enemy types.

- **Actions**:
  - `ACTIONS`: List of available actions for the agent.

- **Rewards and Penalties**:
  - `REWARD_ENEMY_KILL`: Reward for killing an enemy.
  - `REWARD_WAVE_COMPLETE`: Reward for completing a wave.
  - `REWARD_BASE_SURVIVAL`: Reward for base survival.
  - `PENALTY_ENEMY_HIT_BASE`: Penalty for an enemy hitting the base.
  - `PENALTY_WASTED_RESOURCES`: Penalty for wasting resources.
  - `REWARD_STEP`: Reward for each step.

## Training

The agent is trained using the `train_agent` function. You can specify the alpha and beta_start parameters for prioritized experience replay. The training process includes real-time rendering of the game state and agent actions.

```python
def train_agent(alpha=0.6, beta_start=0.4):
    # Training logic here
```

## Results

After training, the agent's performance is evaluated based on the average reward obtained during the episodes. The best hyperparameters are determined through a grid search over different alpha and beta_start values.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- Inspired by classic Tower Defense games.
- Built using Python, Pygame, NumPy, and TensorFlow.

Feel free to customize and extend this project for your own purposes. Happy coding!
