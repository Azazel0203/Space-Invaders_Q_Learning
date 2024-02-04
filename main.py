import gym
import numpy as np
from tqdm import tqdm
from model import Agent, DeepQNetwork

def preprocess(frame): # 210, 160, 3 -> 185, 95, 1
    return np.mean(frame[15:200, 30:125], axis=2).reshape(185, 95, 1)  # 184, 95, 1

def stack_frames(stacked_frames, frame, stack_size=4):
    if stacked_frames is None:
        stacked_frames = np.zeros(shape=(stack_size, *frame.shape))  # 4, 185, 95, 1
        stacked_frames[0] = frame.copy()
    else:
        stacked_frames[:-1] = stacked_frames[1:]
        stacked_frames[-1] = frame.copy()
    # stacked_frames = stacked_frames.view(-1, 1, 185, 95, 4)
    return stacked_frames # (4, 185, 95, 1)

def main():
    env = gym.make('SpaceInvaders-v4', render_mode='rgb_array')
    env.metadata['render_fps'] = 144
    n_actions = env.action_space.n
    agent = Agent(gamma=0.95, epsilon=1.0, alpha=0.003, replace_freq=None, min_epsilon=0.001, mem_size=5000)

    # Fill Memory with random Samples
    while agent.mem_cntr < agent.mem_size:
        state, _ = env.reset()
        stacked_frames = None
        observation = stack_frames(stacked_frames=stacked_frames, frame=preprocess(state))
        done = False
        while not done:
            action = env.action_space.sample()
            new_state, reward, done, truncated, _ = env.step(action)
            new_observation = stack_frames(stacked_frames=stacked_frames, frame=preprocess(new_state))
            agent.store_transitions(observation=observation, action=action, reward=reward, new_observation=new_observation, done=done)
            observation = new_observation
    print("initialized the memory")

    # Play and Learn
    max_steps = 10
    episodes = 2
    rewards = []
    batch_size = 32
    eps_history = []
    scores = []
    
    env = gym.make('SpaceInvaders-v4', render_mode='rgb_array')
    env.metadata['render_fps'] = 144
    for episode in range(episodes):
        print(f'Game: {episode} | epsilon: {agent.EPSILON:.4f}')
        eps_history.append(agent.EPSILON)
        done = False
        state, _ = env.reset()
        # state -> (210, 160, 3)
        stacked_frames = None
        observation = stack_frames(stacked_frames=stacked_frames, frame=preprocess(state))
        # observation -> (4, 185, 95, 1)
        reward_current_episode = 0
        score = 0
        while not done:
            action = agent.chooseAction(observation=observation)
            # env.render()
            state_, reward, done, truncated, info = env.step(action)
            new_observation = stack_frames(stacked_frames=stacked_frames, frame=preprocess(state_))
            reward_current_episode += reward
            score += 1
            if done and info['ale.lives'] == 0:
                reward -= 100
            agent.store_transitions(observation=observation, action=action, reward=reward, new_observation=new_observation, done=done)
            observation = new_observation
            agent.learn(batch_size)
        scores.append(score)
        rewards.append(reward_current_episode)
        print(f"Score: {score} | Reward: {reward_current_episode}")

if __name__ == "__main__":
    main()
