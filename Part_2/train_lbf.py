import os
# Force software rendering to prevent "Segmentation Fault" on WSL
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

import gymnasium as gym
import lbforaging
import numpy as np
import matplotlib.pyplot as plt
import imageio
import warnings
from iql import IQL
from cql import CQL

# Suppress warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
CONFIG = {
    "gamma": 0.99,          
    "lr": 0.1,              
    "initial_epsilon": 1.0,
    "min_epsilon": 0.05,
    "decay_steps": 90000,   
    "total_episodes": 100000, 
    "eval_freq": 1000,      
    "window": 500           
}

def get_env_id(coop: bool):
    return "Foraging-5x5-2p-1f-coop-v3" if coop else "Foraging-5x5-2p-1f-v3"

def moving_average(data, window_size):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def get_render_frame(env):
    # Try Standard Gymnasium Render
    try:
        frame = env.render()
        if frame is not None and isinstance(frame, np.ndarray) and frame.ndim >= 3:
            return frame
    except:
        pass
    # Try Unwrapped Legacy Render
    try:
        frame = env.unwrapped.render(mode='rgb_array')
        if frame is not None and isinstance(frame, np.ndarray) and frame.ndim >= 3:
            return frame
    except:
        pass
        
    return None

def record_video_imageio(agent, agent_type, coop_mode):
    env_id = get_env_id(coop_mode)
    print(f"Recording video for {agent_type} (Coop={coop_mode})...")
    
    # Create Env
    try:
        env = gym.make(env_id, render_mode="rgb_array")
    except:
        env = gym.make(env_id)

    frames = [] 
    
    # Force Greedy Policy
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    # 30 Seconds at 5 FPS = 150 Frames
    FPS = 5
    MIN_FRAMES = FPS * 30  
    
    # Start the first episode
    obs, _ = env.reset()
    episode_reward = 0
    
    while len(frames) < MIN_FRAMES:
        # 1. Capture Frame
        frame = get_render_frame(env)
        if frame is not None:
            frames.append(frame)
        
        # 2. Act
        if agent_type == "IQL":
            obs_keys = [str(o) for o in obs]
            actions = agent.act(obs_keys)
        else:
            actions = agent.act(obs)
            
        # 3. Step
        obs, rewards, terminated, truncated, _ = env.step(actions)
        episode_reward += np.sum(rewards)
        
        # Check if this specific game is over
        done = terminated or truncated
        
        if done:
            #We Won! (Ate the apple)
            if episode_reward > 0:
                print(f"  -> Agent WON! Stopping video early.")
                break # Stop recording,victory.
            
            #We Failed/Timed Out so RESET and KEEP RECORDING
            #ensures the video reaches 30 seconds
            else:
                print(f"  -> Agent failed/timed out. Resetting to fill video time...")
                obs, _ = env.reset()
                episode_reward = 0
                

    # Add a Victory Pause of 2 seconds
    last_frame = get_render_frame(env)
    if last_frame is not None:
        for _ in range(FPS * 2): 
            frames.append(last_frame)

    env.close()
    agent.epsilon = old_epsilon 
    
    if len(frames) == 0:
        print("ERROR: No frames captured.")
        return

    video_filename = f"video_{agent_type}_{'coop' if coop_mode else 'std'}.mp4"
    
    try:
        imageio.mimsave(video_filename, frames, fps=FPS) 
        print(f"Success! Video saved as: {video_filename}")
    except Exception as e:
        print(f"Error saving video file: {e}")


def train_agent(agent_type, coop_mode):
    env_id = get_env_id(coop_mode)
    print(f"\n--- STARTING TRAINING: {agent_type} | Co-op Mode: {coop_mode} ---")
    
    env = gym.make(env_id)
    
    if agent_type == "IQL":
        agent = IQL(2, env.action_space, CONFIG["gamma"], CONFIG["lr"], CONFIG["initial_epsilon"])
    else:
        agent = CQL(2, env.action_space, CONFIG["gamma"], CONFIG["lr"], CONFIG["initial_epsilon"])

    total_steps = 0
    all_rewards = [] 
    
    for episode in range(CONFIG["total_episodes"]):
        obs, _ = env.reset()
        done = False
        episode_reward = np.zeros(2)
        
        while not done:
            decay_progress = episode / CONFIG["decay_steps"]
            agent.epsilon = max(CONFIG["min_epsilon"], 1.0 - decay_progress)
            
            if agent_type == "IQL":
                obs_keys = [str(o) for o in obs]
                actions = agent.act(obs_keys)
            else:
                actions = agent.act(obs)

            n_obs, rewards, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated
            
            if agent_type == "IQL":
                obs_keys = [str(o) for o in obs]
                n_obs_keys = [str(o) for o in n_obs]
                agent.learn(obs_keys, actions, rewards, n_obs_keys, done)
            else:
                agent.learn(obs, actions, rewards, n_obs, done)
            
            obs = n_obs
            episode_reward += rewards
            total_steps += 1
            
        all_rewards.append(np.sum(episode_reward))

        if episode % CONFIG["eval_freq"] == 0:
            recent_avg = np.mean(all_rewards[-CONFIG["eval_freq"]:])
            print(f"Ep {episode}: Avg Reward {recent_avg:.2f} | Epsilon {agent.epsilon:.2f}")

    env.close()
    return agent, all_rewards

def plot_results(iql_rewards, cql_rewards, mode_label):
    plt.figure(figsize=(12, 6))
    iql_smooth = moving_average(iql_rewards, CONFIG["window"])
    cql_smooth = moving_average(cql_rewards, CONFIG["window"])
    plt.plot(iql_smooth, label="IQL (Independent)", color='blue', alpha=0.9)
    plt.plot(cql_smooth, label="CQL (Centralized)", color='red', alpha=0.9)
    plt.title(f"Training Dynamics ({mode_label}) - {CONFIG['total_episodes']} Episodes")
    plt.xlabel(f"Episodes (Moving Average Window: {CONFIG['window']})")
    plt.ylabel("Total Team Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = f"results_{mode_label}.png"
    plt.savefig(filename)
    print(f"\nPlot saved to {filename}")

if __name__ == "__main__":
    print("=== CONFIGURATION ===")
    print(CONFIG)
    
    # 1. Standard Mode
    print("\n=== EXPERIMENT 1: STANDARD MODE ===")
    iql_agent_std, iql_data_std = train_agent("IQL", coop_mode=False)
    record_video_imageio(iql_agent_std, "IQL", coop_mode=False)
    
    cql_agent_std, cql_data_std = train_agent("CQL", coop_mode=False)
    record_video_imageio(cql_agent_std, "CQL", coop_mode=False)
    
    plot_results(iql_data_std, cql_data_std, "Standard")

    # 2. Cooperative Mode
    print("\n=== EXPERIMENT 2: COOPERATIVE MODE ===")
    iql_agent_coop, iql_data_coop = train_agent("IQL", coop_mode=True)
    record_video_imageio(iql_agent_coop, "IQL", coop_mode=True)
    
    cql_agent_coop, cql_data_coop = train_agent("CQL", coop_mode=True)
    record_video_imageio(cql_agent_coop, "CQL", coop_mode=True)
    
    plot_results(iql_data_coop, cql_data_coop, "Cooperative")
