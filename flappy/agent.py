# import flappy_bird_gymnasium
from dqn import DQN
from experience_replay import ReplayMemory
import gymnasium
import torch 
import itertools 
# env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

class Agent:
    def run(self, is_training=True, render=False):
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n 
        rewards_per_episode = []
        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemory(10000,seed=42)
        
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