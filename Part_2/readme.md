# Multi-Agent Reinforcement Learning - Part 2: Level-Based Foraging

This directory contains the second phase of the project, extending the **Independent Q-Learning (IQL)** and **Centralized Q-Learning (CQL)** algorithms to a complex grid-world environment: **Level-Based Foraging (LBF)**.

In this environment, agents must navigate a grid to collect food. The challenge lies in coordination: food items have "levels," and can only be collected if the sum of the levels of the agents involved equals or exceeds the food's level.

## 📂 File Structure

* **`train_lbf.py`**: The main training script. It runs two major experiments:
    1.  **Standard Mode**: Agents are competitive/individualistic (`coop=False`).
    2.  **Cooperative Mode**: Agents share rewards (`coop=True`).
    It also handles data logging, plotting, and recording video replays of the trained agents.
* **`iql.py`**: The **Independent Q-Learning** agent, adapted for high-dimensional grid states. It treats the other agent as part of the environment and learns a private policy.
* **`cql.py`**: The **Centralized Q-Learning** agent. It learns a joint policy based on the global state and optimizes the team's total reward, enabling superior coordination.
* **`results_Standard.png`**: Plot showing training performance in the standard (competitive) setting. Both agents learn to forage, but CQL converges slightly faster.
* **`results_Cooperative.png`**: Plot showing training performance in the cooperative setting. Shared rewards significantly accelerate learning for both algorithms.
* **`video_*.mp4`**: Video recordings of the trained agents demonstrating their learned behaviors in both modes (e.g., `video_CQL_coop.mp4` shows coordinated foraging).

## 🚀 How to Run

### 1. Install Dependencies
In addition to the standard libraries, you must install the Level-Based Foraging environment:
```bash
pip install gymnasium lbforaging imageio[ffmpeg]
```
(Note: imageio[ffmpeg] is required to save the video files)

### 2. Run the Experiment
This single script will train both IQL and CQL agents in both Standard and Cooperative modes, generate the plots, and save the video replays.

```bash
python train_lbf.py
```
Note: If you are running on a headless server (like WSL), you may need to use xvfb-run python train_lbf.py to enable video recording

### 📊 Results Summary
Standard Mode: Agents learn to forage effectively. CQL reaches optimal performance faster (approx. episode 40k) compared to IQL (approx. episode 50k) because it avoids the "moving target" problem of decentralized learning.

Cooperative Mode: The shared reward signal drastically improves learning speed. CQL solves the environment almost perfectly by episode 30k, demonstrating highly coordinated movement.

Visual Analysis: The generated videos confirm that cooperative agents (especially CQL) learn to move in sync to maximize efficiency, whereas independent agents often exhibit jittery movement before converging.
