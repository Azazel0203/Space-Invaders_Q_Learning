
```markdown
# Deep Q-Network for Space Invaders

## Overview

This project implements a Deep Q-Network (DQN) to play the Space Invaders game using the OpenAI Gym environment. The DQN is implemented using PyTorch and learns to make decisions based on the visual input of the game frames.

## Dependencies

- Python 3.x
- PyTorch
- NumPy
- Gym

Install the required dependencies using:

```bash
pip install torch numpy gym
```

## Project Structure

- `main.py`: The main script to train and play the Space Invaders game using the implemented DQN.
- `model.py`: Contains the definition of the neural network architecture (`DeepQNetwork`) and the agent (`Agent`).
- `utils.py`: Utility functions, including frame preprocessing and stacking.

## How to Use

1. Run the training script:

   ```bash
   python main.py
   ```
2. Adjust hyperparameters, such as learning rate, discount factor, and exploration rate, in the `Agent` class or as command-line arguments in the `main.py` script.

## Code Explanation

### Preprocessing Functions (`preprocess` and `stack_frames`)

```python
def preprocess(frame):  # 210, 160, 3 -> 185, 95, 1
    return np.mean(frame[15:200, 30:125], axis=2).reshape(185, 95, 1)

def stack_frames(stacked_frames, frame, stack_size=4):
    if stacked_frames is None:
        stacked_frames = np.zeros(shape=(stack_size, *frame.shape))  # 4, 185, 95, 1
        stacked_frames[0] = frame.copy()
    else:
        stacked_frames[:-1] = stacked_frames[1:]
        stacked_frames[-1] = frame.copy()
    return stacked_frames  # (4, 185, 95, 1)
```

### Main Loop (`main` function)

The `main` function orchestrates the training and playing of the Space Invaders game. It includes:

- Initializing the Gym environment and the DQN agent.
- Filling the agent's memory with random samples.
- Playing and learning for a specified number of episodes.
- Printing scores and rewards obtained during gameplay.

### DeepQNetwork Class

```python
class DeepQNetwork(nn.Module):
    # ... (constructor and forward method)
```

This class defines the neural network architecture for the Deep Q-Network. It includes convolutional layers (`conv1`, `conv2`, `conv3`) followed by fully connected layers (`fc1`, `fc2`). The `forward` method defines the forward pass through the network.

### Agent Class

```python
class Agent():
    # ... (constructor, store_transitions, chooseAction, learn methods)
```

The `Agent` class encapsulates the DQN agent. It includes methods for storing transitions, choosing actions based on epsilon-greedy policy (`chooseAction`), and learning from the stored experiences (`learn`).

### Training Loop

The training loop within the `main` function involves:

1. Initializing the Gym environment and the agent.
2. Filling the agent's memory with random samples.
3. Playing the game for a specified number of episodes.
4. Choosing actions, observing rewards, and updating the agent's Q-values through the DQN.

### Learning Process

The learning process in the `Agent` class involves:

1. Storing transitions in the replay memory.
2. Choosing actions based on an epsilon-greedy policy during gameplay.
3. Calculating the temporal difference error and updating the Q-values through backpropagation.

### Miscellaneous

- The `replace_freq` parameter in the agent's constructor controls how often the target network is updated with the parameters of the evaluation network.
- The `eps_history` list records the epsilon values over episodes, and `scores` and `rewards` track the scores and rewards obtained during gameplay.

These are the main components and functionalities of your code. You can further refine and extend it based on your specific requirements and experiment with different hyperparameters to improve the performance of your DQN agent.
