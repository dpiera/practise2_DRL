from collections import defaultdict
import random
from typing import List, DefaultDict

import numpy as np
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatdim


class IQL:
    """
    Agent using the Independent Q-Learning algorithm
    """
    def __init__(
        self,
        num_agents: int,
        action_spaces: List[Space],
        gamma: float,
        learning_rate: float = 0.5,
        epsilon: float = 1.0,
        **kwargs,
    ):
        """
        Constructor of IQL

        Initializes variables for independent Q-learning agents

        :param num_agents (int): number of agents
        :param action_spaces (List[Space]): action spaces of the environment for each agent
        :param gamma (float): discount factor (gamma)
        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for all agents

        :attr n_acts (List[int]): number of actions for each agent
        :attr q_tables (List[DefaultDict]): tables for Q-values mapping actions ACTs
            to respective Q-values for all agents
        """
        self.num_agents = num_agents
        self.action_spaces = action_spaces
        self.n_acts = [flatdim(action_space) for action_space in action_spaces]

        self.gamma: float = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        # STORE INITIAL EPSILON (Crucial for exponential decay)
        self.start_epsilon = epsilon
        self.epsilon = epsilon

        # initialise Q-tables for all agents
        # access value of Q_i(o, a) with self.q_tables[i][str((o, a))] (str conversion for hashable obs)
        self.q_tables: List[DefaultDict] = [
            defaultdict(lambda: 0) for _ in range(self.num_agents)
        ]

    def act(self, obss) -> List[int]:# decides what move the agent makes next.
        """
        Implement the epsilon-greedy action selection here for stateless task

        **IMPLEMENT THIS FUNCTION**

        :param obss (List): list of observations for each agent
        :return (List[int]): index of selected action for each agent
        """
        actions = []

        ### MY CODE STARTS HERE ###

        for i in range(self.num_agents):
            # 1. Exploration (Random Action)
            if random.random() < self.epsilon:
                # Pick a random action index from 0 to n_acts-1
                action = random.randint(0, self.n_acts[i] - 1)
            
            # 2. Exploitation (Greedy Action)
            else:
                # We need to find the action with the highest Q-value for the current observation.
                # Since the Q-table uses string keys str((obs, action)), we must 
                # check the value for every possible action.
                q_values = []
                for a in range(self.n_acts[i]):
                    key = str((obss[i], a))
                    q_values.append(self.q_tables[i][key])
                
                # Pick the action index with the highest Q-value
                # We use np.argmax. If all are 0 (start), it defaults to 0.
                action = int(np.argmax(q_values))
                
            actions.append(action)

        #raise NotImplementedError("Need to implement the act() function of IQL")
        return actions

    def learn(
        self,
        obss: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        n_obss: List[np.ndarray],
        done: bool,
    ):
        """
        Updates the Q-tables based on agents' experience

        **IMPLEMENT THIS FUNCTION**

        :param obss (List[np.ndarray]): list of observations for each agent
        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param n_obss (List[np.ndarray]): list of observations after taking the action for each agent
        :param done (bool): flag indicating whether a terminal state has been reached
        :return (List[float]): updated Q-values for current actions of each agent
        """
        ### MY CODE STRATS HERE ###

        updated_q_values = []

        for i in range(self.num_agents):
            obs = obss[i]
            action = actions[i]
            reward = rewards[i]
            next_obs = n_obss[i]

            # 1. Retrieve Current Q-value
            # Construct the key string as per the __init__ instructions
            current_key = str((obs, action))
            current_q = self.q_tables[i][current_key]

            # 2. Calculate Target
            if done:
                # If the game is over, there is no future reward
                target = reward
            else:
                # We need the max Q-value for the NEXT state
                next_q_values = []
                for a in range(self.n_acts[i]):
                    next_key = str((next_obs, a))
                    next_q_values.append(self.q_tables[i][next_key])
                
                max_next_q = max(next_q_values)
                target = reward + self.gamma * max_next_q

            # 3. Update Q-value
            # New Q = Old Q + Learning Rate * (Target - Old Q)
            new_q = current_q + self.learning_rate * (target - current_q)
            self.q_tables[i][current_key] = new_q
            
            updated_q_values.append(new_q)

        #raise NotImplementedError("Need to implement the learn() function of IQL")

        return updated_q_values
    

    def schedule_hyperparameters(self, timestep: int, max_timestep: int, method='exponential'):
        """
        Updates the hyperparameters using Linear or Exponential decay.
        
        :param timestep (int): current timestep
        :param max_timestep (int): total steps
        :param method (str): 'linear' or 'exponential'
        """
        if method == 'linear':
            # Your original simple formula
            decay_progress = timestep / (0.8 * max_timestep)
            self.epsilon = 1.0 - min(1.0, decay_progress) * 0.99
            
        elif method == 'exponential':
            # New Complex Method:
            # We decay faster at the start and slower at the end.
            # Formula: epsilon = start * e^(-decay_rate * progress)
            
            decay_rate = 5.0  # Adjust this: Higher = faster decay
            progress = timestep / max_timestep
            
            # Calculate exponential decay
            self.epsilon = self.start_epsilon * np.exp(-decay_rate * progress)
            
            # Safety Clamp: Ensure we always explore at least 5% (0.05)
            # This prevents the agent from getting permanently stuck in a bad habit.
            self.epsilon = max(0.05, self.epsilon)

        

    # def schedule_hyperparameters(self, timestep: int, max_timestep: int):
    #     """
    #     Updates the hyperparameters

    #     This function is called before every episode and allows you to schedule your
    #     hyperparameters.

    #     :param timestep (int): current timestep at the beginning of the episode
    #     :param max_timestep (int): maximum timesteps that the training loop will run for
    #     """
    #     self.epsilon = 1.0 - (min(1.0, timestep / (0.8 * max_timestep))) * 0.99
