from collections import defaultdict
import random
from typing import List

import numpy as np
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatdim

class CQL:
    """
    Centralized Q-Learning Agent.
    It learns a single policy for the joint action of all agents to maximize the SUM of rewards.
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
        self.num_agents = num_agents
        self.action_spaces = action_spaces
        self.n_acts = [flatdim(action_space) for action_space in action_spaces]
        
        # Calculate total joint actions (Product of all individual actions)
        # For Prisoner's Dilemma: 2 * 2 = 4 joint actions
        self.total_joint_actions = int(np.prod(self.n_acts))

        self.gamma = gamma
        self.learning_rate = learning_rate
        
        # Store initial epsilon for decay
        self.start_epsilon = epsilon
        self.epsilon = epsilon

        # Single Central Q-table
        # Key: str((joint_obs, joint_action_index))
        self.q_table = defaultdict(lambda: 0)

    def _encode_joint_action(self, actions: List[int]) -> int:
        """Helper: Converts individual actions [a1, a2] -> single index (0-3)"""
        index = 0
        multiplier = 1
        # We iterate in reverse to treat the list like a number (e.g. binary)
        for i in reversed(range(self.num_agents)):
            index += actions[i] * multiplier
            multiplier *= self.n_acts[i]
        return index

    def _decode_joint_action(self, index: int) -> List[int]:
        """Helper: Converts single index (0-3) -> individual actions [a1, a2]"""
        actions = []
        for num_actions in reversed(self.n_acts):
            actions.append(index % num_actions)
            index //= num_actions
        return list(reversed(actions))

    def act(self, obss) -> List[int]:
        """
        Selects a Joint Action using Epsilon-Greedy.
        Returns a list of actions for the environment.
        """
        # 1. Exploration: Random Joint Action
        if random.random() < self.epsilon:
            joint_action_index = random.randint(0, self.total_joint_actions - 1)
            return self._decode_joint_action(joint_action_index)
        
        # 2. Exploitation: Best Joint Action
        else:
            joint_obs_str = str(obss)
            q_values = []
            
            # Check Q-value for every possible joint action (0, 1, 2, 3)
            for i in range(self.total_joint_actions):
                key = str((joint_obs_str, i))
                q_values.append(self.q_table[key])
            
            # Pick the joint action index with the highest value
            best_joint_index = int(np.argmax(q_values))
            return self._decode_joint_action(best_joint_index)

    def learn(
        self,
        obss: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        n_obss: List[np.ndarray],
        done: bool,
    ):
        """
        Updates the Central Q-table.
        CRITICAL: Uses sum(rewards) to encourage cooperation.
        """
        # Prepare Keys
        joint_obs_str = str(obss)
        next_joint_obs_str = str(n_obss)
        joint_action_index = self._encode_joint_action(actions)
        
        # --- THE MAGIC FIX ---
        # The central brain cares about the TEAM score, not individual scores.
        total_reward = sum(rewards)

        # 1. Get current Q
        current_key = str((joint_obs_str, joint_action_index))
        current_q = self.q_table[current_key]

        # 2. Calculate Target
        if done:
            target = total_reward
        else:
            # Find max Q for NEXT joint state over all possible joint actions
            next_q_values = []
            for i in range(self.total_joint_actions):
                next_key = str((next_joint_obs_str, i))
                next_q_values.append(self.q_table[next_key])
            
            max_next_q = max(next_q_values)
            target = total_reward + self.gamma * max_next_q

        # 3. Update Q-value
        self.q_table[current_key] += self.learning_rate * (target - current_q)

    def schedule_hyperparameters(self, timestep: int, max_timestep: int, method='linear'):
        """Same scheduler as IQL"""
        if method == 'linear':
            decay_progress = timestep / (0.8 * max_timestep)
            self.epsilon = 1.0 - min(1.0, decay_progress) * 0.99
        elif method == 'exponential':
            decay_rate = 5.0
            progress = timestep / max_timestep
            self.epsilon = self.start_epsilon * np.exp(-decay_rate * progress)
            self.epsilon = max(0.05, self.epsilon)
