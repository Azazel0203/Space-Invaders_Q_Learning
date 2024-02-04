import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module):
    def __init__(self, ALPHA, n_actions):
        super().__init__()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4, padding=1, device=self.device) 
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, device=self.device)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, device=self.device)
        self.fc1 = nn.Linear(in_features=4*128*19*8, out_features=521, device=self.device)
        self.fc2 = nn.Linear(in_features=521, out_features=n_actions, device=self.device)
        self.optimizer = optim.RMSprop(params=self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss()
        self.to(self.device)
    
    def forward(self, observation):
        observation = T.Tensor(observation).to(self.device)
        observation = observation.view(-1, 1, 185, 95) # however many batch is possible
        observation = F.relu(self.conv1(observation))
        observation = F.relu(self.conv2(observation))
        observation = F.relu(self.conv3(observation))
        # print(observation.shape)
        observation = observation.view(-1, 128*19*8*4) # make sure its of desired shape
        # print(observation.shape)
        observation = F.relu(self.fc1(observation))
        # print(observation.shape)
        q_values_for_all_possibel_actions = self.fc2(observation)
        return q_values_for_all_possibel_actions


class Agent():
    def __init__(self, gamma, epsilon, alpha, mem_size, replace_freq, min_epsilon, action_space=[0, 1, 2, 3, 4, 5]):
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.ALPHA = alpha
        self.mem_size = mem_size
        self.replace_freq = replace_freq
        self.min_epsilon = min_epsilon
        self.action_space = action_space
        self.mem_cntr = 0
        self.learned_step_counter = 0
        self.steps = 0
        self.memory = []
        self.q_eval = DeepQNetwork(self.ALPHA, len(self.action_space))
        self.q_next = DeepQNetwork(self.ALPHA, len(self.action_space))
    
    def store_transitions(self, observation, action, reward, new_observation, done):
        if self.mem_cntr < self.mem_size:
            self.memory.append([observation, action, reward, new_observation, done])
        else:
            index = self.mem_cntr%self.mem_size
            self.memory[index] = [observation, action, reward, new_observation, done]
        self.mem_cntr += 1
    
    
    def chooseAction(self, observation):
        rand = np.random.random()
        # print(self.action_space)
        # exit()
        # print("random", rand)
        if rand < self.EPSILON: # random action
            action = np.random.randint(len(self.action_space))
            print(action)
        else:
            q_values_for_all_possible_actions = self.q_eval.forward(observation=observation)
            action = T.argmax(q_values_for_all_possible_actions)
        self.steps += 1
        return action

    def learn(self, batch_size):
        self.q_eval.optimizer.zero_grad() # zero_grad the model
        
        # if its the time to copy to target!!
        if self.replace_freq is not None and self.learned_step_counter%self.replace_freq == 0: # replace the q_next
            self.q_next.load_state_dict(self.q_eval.state_dict())
        
        if self.mem_cntr + batch_size < self.mem_size:
            batch_start = int(np.random.choice(range(self.mem_cntr)))
        else:
            batch_start = int(np.random.choice(range(self.mem_size-batch_size-1)))
        
        mini_batch = self.memory[batch_start:batch_start+batch_size]
        memory = np.array(mini_batch, dtype=object)
        
        q_values_for_all_possible_actions = self.q_eval.forward(list(memory[:,0][:])).to(self.q_eval.device)
        q_values_for_all_possible_next_actions = self.q_next.forward(list(memory[:,3][:])).to(self.q_eval.device)
        
        # print(q_values_for_all_possible_next_actions)
        maxA_next_actions = T.argmax(q_values_for_all_possible_next_actions, dim=1).to(self.q_eval.device)
        rewards = T.Tensor(list(memory[:,2])).to(self.q_eval.device)
        terminal = T.Tensor(list(memory[:,4])).to(self.q_eval.device)
        # Now the loss function for everything else should be 0
        # so Q_target should be almost equal to q_values_for_all_possible_actions
        # except the maxA_next_actions qvalues
        Q_target = q_values_for_all_possible_actions.clone()
        indices = np.arange(batch_size)
        Q_target[indices, maxA_next_actions] = rewards + (1-terminal)*self.GAMMA*T.max(q_values_for_all_possible_next_actions)
        
        # decay the epsilon
        if self.steps > 500:
            if self.EPSILON - 1e4 > self.min_epsilon:
                self.EPSILON -= 1e4
            else:
                self.EPSILON = self.min_epsilon
        
        # get loss and backprop
        loss = self.q_eval.loss(Q_target, q_values_for_all_possible_actions).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learned_step_counter+=1
        
        
        
        