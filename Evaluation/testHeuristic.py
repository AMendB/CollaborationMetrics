import sys
import json
sys.path.append('.')

from Environment.CleanupEnvironment import MultiAgentCleanupEnvironment
from Algorithms.LawnMower import LawnMowerAgent
from Algorithms.NRRA import WanderingAgent
from Algorithms.PSO import ParticleSwarmOptimizationFleet
from Algorithms.Greedy import OneStepGreedyFleet
from Algorithms.DRL.ActionMasking.ActionMaskingUtils import ConsensusSafeActionMasking
import numpy as np
from tqdm import trange

algorithms = [
	# 'WanderingAgent', 
    # 'LawnMower', 
    'PSO', 
    # 'Greedy',
	]

SEED = 3
SHOW_RENDER = False
PRINT_INFO = False
RUNS = 100


# Set config #
# scenario_map_name = 'acoruna_port'
# scenario_map_name = 'marinapalamos'
scenario_map_name = 'comb_port'
# scenario_map_name = 'challenging_map_big'
n_actions_explorers = 8
n_actions_cleaners = 8
if 'big' in scenario_map_name:
    n_explorers = 3
    n_cleaners = 3
else:
    n_explorers = 2
    n_cleaners = 2
if 'challenging' in scenario_map_name:
    obstacles = True
else:
    obstacles = False
n_agents = n_explorers + n_cleaners
movement_length_explorers = 2
movement_length_cleaners = 1
movement_length_of_each_agent = np.repeat((movement_length_explorers, movement_length_cleaners), (n_explorers, n_cleaners))
vision_length_explorers = 4
vision_length_cleaners = 1
max_distance_travelled_explorers = 400
max_distance_travelled_cleaners = 200
if 'big' in scenario_map_name:
    max_steps_per_episode = 170
else:
    max_steps_per_episode = 150

reward_function = 'negativedijkstra'
# reward_function = 'negativeastar'
# reward_function = 'negativedistance'
# reward_weights=(1, 50, 2, 0)
reward_weights=(2.626225214357622, 14.33181947501113, 5.826858678174348, 1.7319722255470185)

# Set initial positions #
random_initial_positions = True
if random_initial_positions:
    initial_positions = 'fixed'
else:
    initial_positions = np.array([[32, 7], [30, 7], [28, 7], [26, 7]])[:n_agents, :] # acoruna_port
    # initial_positions = None

# Create environment # 
env = MultiAgentCleanupEnvironment(scenario_map_name = scenario_map_name,
                        number_of_agents_by_team=(n_explorers,n_cleaners),
                        n_actions_by_team=(n_actions_explorers, n_actions_cleaners),
                        max_distance_travelled_by_team = (max_distance_travelled_explorers, max_distance_travelled_cleaners),
                        max_steps_per_episode = max_steps_per_episode,
                        fleet_initial_positions = initial_positions, # None, 'area', 'fixed' or positions array
                        seed = SEED,
                        movement_length_by_team =  (movement_length_explorers, movement_length_cleaners),
                        vision_length_by_team = (vision_length_explorers, vision_length_cleaners),
                        flag_to_check_collisions_within = False,
                        max_collisions = 1000,
                        reward_function = reward_function,
                        reward_weights = reward_weights,
                        dynamic = True,
                        obstacles = obstacles,
                        show_plot_graphics = SHOW_RENDER,
                        )

for algorithm in algorithms:
    if algorithm == 'LawnMower':
        lawn_mower_rng = np.random.default_rng(seed=100)
        agents = [LawnMowerAgent(world=env.scenario_map, number_of_actions=8, movement_length=movement_length_of_each_agent[i], forward_direction=int(lawn_mower_rng.uniform(0,8)), seed=SEED+i, agent_is_cleaner=env.team_id_of_each_agent[i]==env.cleaners_team_id) for i in range(n_agents)]
        consensus_safe_masking_module = ConsensusSafeActionMasking(navigation_map = env.scenario_map, angle_set_of_each_agent=env.angle_set_of_each_agent, movement_length_of_each_agent = env.movement_length_of_each_agent)
    elif algorithm == 'WanderingAgent':
        agents = [WanderingAgent(world=env.scenario_map, number_of_actions=8, movement_length=movement_length_of_each_agent[i], seed=SEED+i, agent_is_cleaner=env.team_id_of_each_agent[i]==env.cleaners_team_id) for i in range(n_agents)]
        consensus_safe_masking_module = ConsensusSafeActionMasking(navigation_map = env.scenario_map, angle_set_of_each_agent=env.angle_set_of_each_agent, movement_length_of_each_agent = env.movement_length_of_each_agent)
    elif algorithm == 'PSO':
        agents = ParticleSwarmOptimizationFleet(env)
        consensus_safe_masking_module = ConsensusSafeActionMasking(navigation_map = env.scenario_map, angle_set_of_each_agent=env.angle_set_of_each_agent, movement_length_of_each_agent = env.movement_length_of_each_agent)
    elif algorithm == 'Greedy':
        agents = OneStepGreedyFleet(env)
        consensus_safe_masking_module = ConsensusSafeActionMasking(navigation_map = env.scenario_map, angle_set_of_each_agent=env.angle_set_of_each_agent, movement_length_of_each_agent = env.movement_length_of_each_agent)

    mean_cleaned_percentage = 0
    mse_error_accumulated = 0
    average_reward = [0 for _ in range(env.n_teams)]
    average_episode_length = [0 for _ in range(env.n_teams)]
    env.reset_seeds()

    # Start episodes #
    for run in trange(RUNS):
        
        done = {i: False for i in range(n_agents)}
        states = env.reset_env()

        # runtime = 0
        step = 0

        # Reset algorithms #
        if algorithm in ['LawnMower']:
            for i in range(n_agents):
                agents[i].reset(int(lawn_mower_rng.uniform(0,8)) if algorithm == 'LawnMower' else None, env.scenario_map)
            consensus_safe_masking_module.update_map(env.scenario_map)
        elif algorithm in ['WanderingAgent']:
            for i in range(n_agents):
                agents[i].reset(env.scenario_map)
            consensus_safe_masking_module.update_map(env.scenario_map)
        elif algorithm in ['PSO']:
            agents.reset()
            consensus_safe_masking_module.update_map(env.scenario_map)
        
        acc_rw_episode = [0 for _ in range(n_agents)]
        ep_length_per_teams = [0 for _ in range(env.n_teams)]

        while any([not value for value in done.values()]):  # while at least 1 active

            # Add step #
            step += 1
            
            # Take new actions #
            if algorithm  in ['WanderingAgent', 'LawnMower']:
                actions = {agent_id: agents[agent_id].move(actual_position=position, trash_in_pixel=env.model_trash_map[position[0], position[1]]) for agent_id, position in env.get_active_agents_positions_dict().items()}
                q_values = {agent_id: np.array([1 if i == actions[agent_id] else 0 for i in range(8)]).astype(float) for agent_id in range(n_agents)}
                actions = consensus_safe_masking_module.query_actions(q_values=q_values, agents_positions=env.get_active_agents_positions_dict(), model_trash_map=env.model_trash_map, team_id_of_each_agent=env.team_id_of_each_agent)
            elif algorithm == 'PSO':
                q_values = agents.get_agents_q_values()
                actions = consensus_safe_masking_module.query_actions(q_values=q_values, agents_positions=env.get_active_agents_positions_dict(), model_trash_map=env.model_trash_map, team_id_of_each_agent=env.team_id_of_each_agent)
            elif algorithm == 'Greedy':
                # actions = agents.get_agents_actions()
                q_values = agents.get_agents_q_values()
                actions = consensus_safe_masking_module.query_actions(q_values=q_values, agents_positions=env.get_active_agents_positions_dict(), model_trash_map=env.model_trash_map, team_id_of_each_agent=env.team_id_of_each_agent)

            # t0 = time.time()
            states, new_reward, done = env.step(actions, dont_calculate_rewards=False)
            acc_rw_episode = [acc_rw_episode[i] + new_reward[i] for i in range(n_agents)]
            mse_error_accumulated += env.get_model_mse()
            ep_length_per_teams = [ep_length_per_teams[team_id] + 1 if not env.dones_by_teams[team_id] else ep_length_per_teams[team_id] for team_id in env.teams_ids]
            # t1 = time.time()
            # runtime += t1-t0
            
            if PRINT_INFO:
                print(f"Step {env.steps}")
                print(f"Actions: {dict(sorted(actions.items()))}")
                print(f"Rewards: {new_reward}")
                trashes_agents_pixels = {agent_id: env.model_trash_map[position[0], position[1]] for agent_id, position in env.get_active_agents_positions_dict().items()}
                print(f"Trashes in agents pixels: {trashes_agents_pixels}")
                print(f"Trashes removed: {env.trashes_removed_per_agent}")
                print(f"Trashes remaining: {len(env.trash_positions_yx)}")
                print()

        # Print accumulated reward of episode #
        if PRINT_INFO:
            print('Total reward: ', acc_rw_episode)
            # print('Total runtime: ', runtime)
        
        # Calculate mean metrics all episodes #
        mean_cleaned_percentage += env.get_percentage_cleaned_trash()
        for team in range(env.n_teams):
            mean_team_acc_reward_ep = np.mean([acc_rw_episode[i] for i in range(n_agents) if env.team_id_of_each_agent[i] == team])
            average_reward[team] += mean_team_acc_reward_ep
            average_episode_length[team] += ep_length_per_teams[team]
    
    # Print algorithm results #
    print(f'Algorithm: {algorithm}. Scenario: {env.scenario_map_name}, with {n_explorers} explorers and {n_cleaners} cleaners. Dynamic: {env.dynamic}. Obstacles: {env.obstacles}. Reward function: {env.reward_function}, Reward weights: {env.reward_weights}.')

    for team in range(env.n_teams):
        print(f'Average reward for {algorithm} team {team} with {n_explorers if team==0 else n_cleaners} agents: {average_reward[team]/RUNS}. Episode average length of {average_episode_length[team]/RUNS}. Cleaned percentage: {round(mean_cleaned_percentage/RUNS*100, 2)}%. Accumulated MSE: Mean MSE accumulated: {round(mse_error_accumulated / RUNS, 4)}')
    print()