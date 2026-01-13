# Collaboration Metrics for Heterogeneous Cooperative Destructive Foraging
This repository contains the simulation environment, algorithms, and evaluation tools used to study heterogeneous cooperative foraging under a destructive foraging setting. The framework is designed to analyze cooperation dynamics between teams with different roles (e.g., search and service). **This work presents a unified and extensible set of metrics for evaluating performance, cooperation, and fairness of inter-team and intra-team work in foraging scenarios.**
The codebase supports reproducible experiments and algorithm comparison.


## Featured files
### Environment - `Environment/` 
File `CleanupEnvironment.py` implements the destructive foraging environment.
Key characteristics:

- Discrete grid-based environment.
- Items (“waste”) are randomly distributed at the beginning of each episode.
- Items are destroyed (removed from the environment) when serviced — no homing or return task is required.
- Two heterogeneous teams with different roles:
    - Scouts: enhanced sensing, responsible for discovering items. 
    - Foragers: limited sensing, service-oriented agents responsible for removing known items. The framework makes the simplifying assumptions that there is no limit to the number of items removed by each agent, which do not restrict the validity of the proposed metrics.
- Partial observability and dynamic internal models (e.g., idleness, belief maps).
- The environment is independent of the specific algorithm used and can be reused with different control policies.

Different maps are available in `Environment/Maps` folder.

### Algorithms - Folder `Algorithms/`

This folder contains the implementations of all evaluated control strategies, including:

- Learning-based approach: Double Deep Q-Learning (DDQL)
- Heuristic baselines: lawn mower and random walker (NRRA)
- Greedy strategies (instantaneous reward maximization)
- Particle Swarm Optimization
- Lévy Walk–based exploration + Dijkstra-based servicing

Each algorithm interacts with the environment through the same interface, enabling fair comparison.

### Evaluation and Metrics
`Evaluation/AlgorithmSaveToLegendarium.py`

Responsible for:

- Logging episode-level and step-level data.
- Storing trajectories, actions, and internal state variables.
- Saving raw experimental results in a structured format (“Legendarium”) for post-processing.

This module is intentionally lightweight to avoid coupling evaluation logic with execution.

`Evaluation/AlgorithmAnalyzerLegendarium.py`

Implements the metric computation and analysis, and figure drawing.

Includes:

- Primary performance metrics (Porcenatge Target Achieved, RMSE, Normalized Time To x%, Throughput, Idleness Metrics...).
- Inter-team metrics (Dicovery-to-service Latency, Inter-team Temporal Lag, Cooperative Success Ratio, Cooperation Sensitivity under Stochastic Corruption...).
- Intra-team metrics (Gini Coefficient, Coverage Overlap, Marginal Contribution...).

### Typical Workflow
1. Configure and run an algorithm in the `CleanupEnvironment.py`.
2. Save execution data using `AlgorithmSaveToLegendarium.py`.
3. Analyze results and compute metrics using `AlgorithmAnalyzerLegendarium.py`.
4. Compare algorithms across multiple dimensions of cooperation and performance.

