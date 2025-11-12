# import flappy_bird_gymnasium
from dqn import DQN
from experience_replay import ReplayMemory
import gymnasium
import torch 
import itertools
import yaml
# env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

class Agent:
    def __init__(self,hyperparameters_set):
        with open('hyperparameters.yaml', 'r') as file:
            all_hyperparameters = yaml.safe_load(file)
            hyperparameters = all_hyperparameters[hyperparameters_set]
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']

    def run(self, is_training=True, render=False):
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n 
        rewards_per_episode = []
        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size,seed=42)
        
        for episode in itertools.count(1000):
            state, _ = env.reset()
            terminated = False
            episode_reward = 0.0
            while not terminated:
                # Next action:
                # (feed the observation to your agent here)
                action = env.action_space.sample()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action)
                
                episode_reward += reward

                if is_training:
                    memory.append((state,action,new_state,reward, terminated))

                # Move to new state
                state = new_state
            rewards_per_episode.append(episode_reward)