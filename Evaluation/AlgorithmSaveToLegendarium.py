import sys
sys.path.append('.')
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import trange
import pandas as pd
from cycler import cycler
from datetime import datetime
from itertools import product

from Environment.CleanupEnvironment import MultiAgentCleanupEnvironment
from Evaluation.Utils.metrics_wrapper import MetricsDataCreator

from sylegendarium.legendarium import Legendarium
import numbers
import copy

class AlgorithmRecorder:
    def __init__(self, savepath, map_name, teams_name, team_id_of_each_agent, max_distance_by_team, max_steps_per_episode, objectives, n_agents, runs, experiment_name='legendarium'):
        ''' Initialize the storage of metrics with legendarium. '''

        # Inicializamos legendarium
        self.data_saver = Legendarium(experiment_name=experiment_name,
                                                    experiment_description=f'{map_name} experiment',
                                                    path=savepath)

        # Create a parameter
        self.data_saver.create_parameter("map_name", map_name)
        self.data_saver.create_parameter("teams_name", teams_name)
        self.data_saver.create_parameter("team_id_of_each_agent", team_id_of_each_agent)
        self.data_saver.create_parameter("max_distance_by_team", max_distance_by_team)
        self.data_saver.create_parameter("max_steps_per_episode", max_steps_per_episode)
        self.data_saver.create_parameter("n_objectives", len(objectives))
        self.data_saver.create_parameter("n_agents", n_agents)
        self.data_saver.create_parameter("runs", runs)
        
        # Create a metric for date
        self.data_saver.create_metric("date", str, "Date of the experiment", "-")

        # Create a metric related to the algorithm
        self.data_saver.create_metric("algorithm", str, "Algorithm used in the experiment", "-")
        self.data_saver.create_metric("reward_function", str, "Reward function used in the experiment", "-")
        self.data_saver.create_metric("reward_weights", tuple, "Weights of the reward function", "-")
        self.data_saver.create_metric("epsilon", numbers.Real, "Epsilon value", "-")
        self.data_saver.create_metric("epsilon_team", numbers.Real, "Team which is taking epsilon actions", "-")

        # Create metrics related to the agents
        self.data_saver.create_metric("agents_positions", dict, "Agents positions in the map", "Map units")
        self.data_saver.create_metric("actions", dict, "Actions taken by the agents", "Action units")
        self.data_saver.create_metric("dones", dict, "Dones of the agents", "-")
        self.data_saver.create_metric("rewards", dict, "Rewards obtained by the agents", "points")
        self.data_saver.create_metric("reward_components", dict, "Individual components of the reward obtained by the agents", "points")
        self.data_saver.create_metric("traveled_distances", dict, "Distances travelled by the agents", "Distance units")
        self.data_saver.create_metric("trashes_at_sight", dict, "Trashes at sight for each agent", "Trashes")
        self.data_saver.create_metric("cleaned_trashes", dict, "Cleaned trashes by each agent", "Trashes")
        self.data_saver.create_metric("history_cleaned_trashes", dict, "History of cleaned trashes by each agent", "Trashes")
        self.data_saver.create_metric("trash_remaining_info", np.ndarray, "Information about the remaining trash", "step_discover, vehicle_discover")
        self.data_saver.create_metric("trash_removed_info", np.ndarray, "Information about the removed trashes", "step_discover, vehicle_discover, step_remove, vehicle_remove")
        self.data_saver.create_metric("ponderated_distances_from_cleaners_to_known_trash", dict, "Distances from each cleaner to known trash positions ponderated by the amount of trash", "Distance units")
        self.data_saver.create_metric("coverage_overlap_ratio", dict, "Ratio of area covered by more than one agent of the same team from the total covered area by that team", "-")

        # Create metrics related to the model
        self.data_saver.create_metric("objective", str, "Name of the objective", "-")
        self.data_saver.create_metric("ground_truth", np.ndarray, "Ground truth", "Objective units")
        self.data_saver.create_metric("model", np.ndarray, "Model", "Objective units")
        self.data_saver.create_metric("idleness_map", np.ndarray, "Idleness map", "Idleness units")
        self.data_saver.create_metric("trash_remaining", numbers.Real, "Number of trashes remaining in the map", "Trashes")
        self.data_saver.create_metric("percentage_of_trash_collected", numbers.Real, "Percentage of trash collected", "Percentage")
        self.data_saver.create_metric("percentage_of_trash_discovered", numbers.Real, "Percentage of trash discovered", "Percentage")
        self.data_saver.create_metric("model_mse", numbers.Real, "MSE of the model", "Objective units")
        self.data_saver.create_metric("model_rmse", numbers.Real, "RMSE of the model", "Objective units")
        self.data_saver.create_metric("model_r2", numbers.Real, "R2 of the model", "Objective units")
        self.data_saver.create_metric("sum_model_changes", numbers.Real, "Sum of model changes", "Objective units")

    def save_registers(self, run = None, step = None, env = None, algorithm = None, rewards=None, reward_components=None, actions=None, objective = None, dones = None, epsilon = 0, epsilon_team = None):

        # Warning: Make a copy of all mutable objects to be saved: lists, dicts, numpy arrays, etc.
        self.data_saver.write(run=run, 
                        step=step, 
                        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        algorithm=algorithm,
                        reward_function=env.reward_function,
                        reward_weights=env.reward_weights,
                        epsilon=epsilon,
                        epsilon_team=epsilon_team,
                        agents_positions=copy.deepcopy(env.get_active_agents_positions_dict()),
                        actions=copy.deepcopy(actions),
                        dones=copy.deepcopy(dones),
                        rewards=copy.deepcopy(rewards),
                        reward_components=copy.deepcopy(reward_components),
                        traveled_distances=copy.deepcopy(env.get_traveled_distances()),
                        trashes_at_sight=copy.deepcopy(env.get_trashes_at_sight()),
                        cleaned_trashes=copy.deepcopy(env.trashes_removed_per_agent),
                        history_cleaned_trashes=copy.deepcopy(env.history_trashes_removed_per_agent),
                        trash_remaining_info=copy.deepcopy(env.trash_remaining_info),
                        trash_removed_info=copy.deepcopy(env.trash_removed_info),
                        ponderated_distances_from_cleaners_to_known_trash=copy.deepcopy(env.get_ponderated_distances_from_cleaners_to_known_trash()),
                        coverage_overlap_ratio=copy.deepcopy(env.get_coverage_overlap_ratio()),
                        objective=copy.deepcopy(objective),
                        ground_truth=copy.deepcopy(env.real_trash_map),
                        model=copy.deepcopy(env.model_trash_map),
                        idleness_map=copy.deepcopy(env.idleness_map),
                        trash_remaining=len(env.trash_positions_yx),
                        percentage_of_trash_collected=copy.deepcopy(env.get_percentage_cleaned_trash()),
                        percentage_of_trash_discovered=copy.deepcopy(env.get_percentage_discovered_trash()),
                        model_mse=copy.deepcopy(env.get_model_mse()),
                        model_rmse=copy.deepcopy(env.get_model_rmse()),
                        model_r2=copy.deepcopy(env.get_model_r2()),
                        sum_model_changes=copy.deepcopy(env.get_changes_in_model()),
                        )
        
    def save(self):
        ''' Save the data in the legendarium format. '''
        self.data_saver.save()

if __name__ == '__main__':

    import time
    from Algorithms.LawnMower import LawnMowerAgent
    from Algorithms.NRRA import WanderingAgent
    from Algorithms.PSO import ParticleSwarmOptimizationFleet
    from Algorithms.Greedy import OneStepGreedyFleet
    from Algorithms.DRL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
    from Algorithms.DRL.ActionMasking.ActionMaskingUtils import ConsensusSafeActionMasking
    from Algorithms.LevyWalks import LevyWalksFleet

    def evaluate_algorithms(algorithms, show_render, save_data, save_path, runs, seed, seed_epsilon_study, extra_name=''):
        random_decision_counter = 0
        total_decision_counter = 0

        runtimes_dict = {}

        first_iteration = True
        for path_to_algorithm in algorithms:
            
            print(f"Evaluating algorithm: {path_to_algorithm}")
            
            # Find if there is epsilon information in the path "Eps_{epsilon}_EosTm_{epsilon_team}" #
            if 'Eps_' in path_to_algorithm and '_EpsTm_' in path_to_algorithm:
                epsilon = float(path_to_algorithm.split('Eps_')[1].split('_EpsTm_')[0])
                epsilon_team = int(path_to_algorithm.split('_EpsTm_')[1].split('_')[0])
                eps_info = path_to_algorithm[path_to_algorithm.find("Eps_"):path_to_algorithm.find("_EpsTm_") +  len(f"_EpsTm_{epsilon_team}_")]
                path_to_algorithm = path_to_algorithm.replace(eps_info, '')
            else:
                epsilon = -1
                epsilon_team = -1

            rng_q_epsilon = np.random.default_rng(seed=seed_epsilon_study)
            
            ## LITERATURE ALGORITHMS ##
            if any(alg in path_to_algorithm.lower() for alg in ['wanderingagent', 'lawnmower', 'pso', 'greedy', 'greedyastar', 'greedydijkstra', 'levywalks', 'levywalksdijkstra']):
                selected_algorithm = path_to_algorithm

                # If any of the algorithms contains 'Training', extract the scenario_map_name and reward function and weights from the environment_config.json #
                if np.any([alg.find('Training') != -1 for alg in algorithms]):
                    # Get one that contains 'Training' #
                    path = [alg for alg in algorithms if alg.find('Training') != -1][0]
                    path = '/'.join(path.split('/')[:-1]) + '/'
                    f = open(path + 'environment_config.json',)
                    config = json.load(f)
                    f.close()
                    scenario_map_name = config['scenario_map_name']
                    reward_weights= tuple(config['reward_weights'])
                    reward_function = config['reward_function']
                    n_explorers = env_config['number_of_agents_by_team'][0]
                    n_cleaners = env_config['number_of_agents_by_team'][1]
                    obstacles = env_config['obstacles']
                else:
                    scenario_map_name = 'comb_port' # 'ypacarai_lake', 'acoruna_port', 'marinapalamos', 'comb_port'
                    reward_weights=(2.12, 5.91, 2.23, 1.7)
                    if 'astar' in path_to_algorithm.lower():
                        reward_function = 'negativeastar'
                    elif 'dijkstra' in path_to_algorithm.lower():
                        reward_function = 'negativedijkstra'
                    else:
                        reward_function = 'negativedistance' 
                    n_explorers = 2
                    n_cleaners = 2
                    obstacles = False
                # Set config #
                n_actions_explorers = 8
                n_actions_cleaners = 8
                n_agents = n_explorers + n_cleaners
                movement_length_explorers = 2
                movement_length_cleaners = 1
                movement_length_of_each_agent = np.repeat((movement_length_explorers, movement_length_cleaners), (n_explorers, n_cleaners))
                vision_length_explorers = 4
                vision_length_cleaners = 1
                max_distance_travelled_explorers = 400
                max_distance_travelled_cleaners = 200
                max_steps_per_episode = 150

                # Set initial positions #
                random_initial_positions = True
                if random_initial_positions:
                    initial_positions = 'fixed'
                else:
                    # initial_positions = np.array([[30, 20], [40, 25], [40, 20], [30, 28]])[:n_agents, :] # ypacarai lake
                    initial_positions = np.array([[32, 7], [30, 7], [28, 7], [26, 7]])[:n_agents, :] # acoruna_port
                    # initial_positions = None

                # Create environment # 
                env = MultiAgentCleanupEnvironment(scenario_map_name = scenario_map_name,
                                        number_of_agents_by_team=(n_explorers,n_cleaners),
                                        n_actions_by_team=(n_actions_explorers, n_actions_cleaners),
                                        max_distance_travelled_by_team = (max_distance_travelled_explorers, max_distance_travelled_cleaners),
                                        max_steps_per_episode = max_steps_per_episode,
                                        fleet_initial_positions = initial_positions, # None, 'area', 'fixed' or positions array
                                        seed = seed,
                                        movement_length_by_team =  (movement_length_explorers, movement_length_cleaners),
                                        vision_length_by_team = (vision_length_explorers, vision_length_cleaners),
                                        flag_to_check_collisions_within = False,
                                        max_collisions = 1000,
                                        reward_function = reward_function,
                                        reward_weights = reward_weights,
                                        dynamic = True,
                                        obstacles = obstacles,
                                        show_plot_graphics = show_render,
                                        )
                scenario_map = env.scenario_map
                scenario_map_name = env.scenario_map_name

                if "lawnmower" in selected_algorithm.lower():
                    lawn_mower_rng = np.random.default_rng(seed=100)
                    selected_algorithm_agents = [LawnMowerAgent(world=scenario_map, number_of_actions=8, movement_length=movement_length_of_each_agent[i], forward_direction=int(lawn_mower_rng.uniform(0,8)), seed=seed+i, agent_is_cleaner=env.team_id_of_each_agent[i]==env.cleaners_team_id) for i in range(n_agents)]
                elif "wanderingagent" in selected_algorithm.lower():
                    selected_algorithm_agents = [WanderingAgent(world=scenario_map, number_of_actions=8, movement_length=movement_length_of_each_agent[i], seed=seed+i, agent_is_cleaner=env.team_id_of_each_agent[i]==env.cleaners_team_id) for i in range(n_agents)]
                elif "pso" in selected_algorithm.lower():
                    selected_algorithm_agents = ParticleSwarmOptimizationFleet(env)
                elif "greedy" in selected_algorithm.lower():
                    selected_algorithm_agents = OneStepGreedyFleet(env)
                elif "levywalks" in selected_algorithm.lower():
                    selected_algorithm_agents = LevyWalksFleet(env, seed=seed)
                
                consensus_safe_masking_module = ConsensusSafeActionMasking(navigation_map = env.scenario_map, angle_set_of_each_agent=env.angle_set_of_each_agent, movement_length_of_each_agent = env.movement_length_of_each_agent)

            ## DEEP REINFORCEMENT LEARNING ALGORITHMS ##
            else:
                # Load env config #
                model_policy = path_to_algorithm.split('/')[-1]
                path_to_algorithm = '/'.join(path_to_algorithm.split('/')[:-1]) + '/'
                f = open(path_to_algorithm + 'environment_config.json',)
                env_config = json.load(f)
                f.close()
                
                env = MultiAgentCleanupEnvironment(scenario_map_name = env_config['scenario_map_name'],
                                        number_of_agents_by_team=env_config['number_of_agents_by_team'],
                                        n_actions_by_team=env_config['n_actions'],
                                        max_distance_travelled_by_team = env_config['max_distance_travelled_by_team'],
                                        max_steps_per_episode=env_config['max_steps_per_episode'],
                                        fleet_initial_positions = env_config['fleet_initial_positions'], # np.array(env_config['fleet_initial_positions']), #
                                        seed = seed,
                                        movement_length_by_team =  env_config['movement_length_by_team'],
                                        vision_length_by_team = env_config['vision_length_by_team'],
                                        flag_to_check_collisions_within = env_config['flag_to_check_collisions_within'],
                                        max_collisions = env_config['max_collisions'],
                                        reward_function = env_config['reward_function'],
                                        reward_weights = tuple(env_config['reward_weights']),
                                        dynamic = env_config['dynamic'],
                                        obstacles = env_config['obstacles'],
                                        show_plot_graphics = show_render,
                                        )
                scenario_map = env.scenario_map
                scenario_map_name = env.scenario_map_name
                n_agents = env.n_agents
                reward_function = env.reward_function
                reward_weights = env.reward_weights
                
                # Load exp config #
                f = open(path_to_algorithm + 'experiment_config.json',)
                exp_config = json.load(f)
                f.close()

                independent_networks_per_team = exp_config['independent_networks_per_team']
                greedy_training = exp_config['greedy_training']
                pso_training = exp_config['pso_training']

                if independent_networks_per_team and not greedy_training and not pso_training:
                    selected_algorithm = "DRLIndNets"
                elif independent_networks_per_team and greedy_training and not pso_training:
                    selected_algorithm = "DRLIndNetsGreedy"
                elif independent_networks_per_team and not greedy_training and pso_training:
                    selected_algorithm = "DRLIndNetsPSO"
                else:
                    selected_algorithm = "DRLOneNetwork"
                    # raise NotImplementedError("This algorithm is not implemented. Choose one that is.")

                if epsilon > 0:
                    consensus_safe_masking_module = ConsensusSafeActionMasking(navigation_map = env.scenario_map, angle_set_of_each_agent=env.angle_set_of_each_agent, movement_length_of_each_agent = env.movement_length_of_each_agent)


                network = MultiAgentDuelingDQNAgent(env=env,
                                        memory_size=int(1E3),  #int(1E6), 1E5
                                        batch_size=exp_config['batch_size'],
                                        target_update=1000,
                                        seed = seed,
                                        consensus_actions=exp_config['consensus_actions'],
                                        device='cuda:0',
                                        independent_networks_per_team = independent_networks_per_team,
                                        )
                network.load_model(path_to_algorithm + f'{model_policy}.pth')
                network.epsilon = 0

            if save_data and first_iteration:
                # Reward function and create path to save #
                if not save_path.endswith('/'):
                    save_path += '/'
                save_path = f'{save_path}{extra_name}' + scenario_map_name + '.' + str(n_agents) + '.' + reward_function.split('_')[0] + '_' + '_'.join(map(str, reward_weights))
                if not(os.path.exists(save_path)): # create the directory if not exists
                    os.mkdir(save_path)

                data_saver = AlgorithmRecorder(experiment_name=save_name,
                                            savepath=save_path,
                                                map_name=scenario_map_name,
                                                teams_name={0: 'Scouts', 1: 'Foragers'},
                                                team_id_of_each_agent={idx: team_id for idx, team_id in enumerate(env.team_id_of_each_agent)},
                                                max_distance_by_team={idx: dist for idx, dist in enumerate(env.max_distance_travelled_by_team)},
                                                max_steps_per_episode=env.max_steps_per_episode,
                                                objectives=['Trash'],
                                                n_agents=n_agents,
                                                runs=runs)
                env.save_environment_configuration(save_path)
                first_iteration = False

            runtimes_dict[selected_algorithm] = []

            env.reset_seeds()

            # START EPISODES #
            for run in trange(runs):
                
                done = {i: False for i in range(n_agents)}
                states = env.reset_env()

                runtime = 0
                step = 0

                # Save data #
                if save_data:
                    data_saver.save_registers(env=env,
                                            run=run, 
                                            step=step, 
                                            algorithm=selected_algorithm, 
                                            rewards={i: 0 for i in range(n_agents)},
                                            reward_components={i: {} for i in range(n_agents)},
                                            actions={i: None for i in range(n_agents)}, 
                                            objective="Trash",
                                            dones=done,
                                            epsilon=epsilon,
                                            epsilon_team=epsilon_team)

                # Reset algorithms #
                if 'drl' in selected_algorithm.lower():
                    network.nogobackfleet_masking_module.reset()
                elif 'lawnmower' in selected_algorithm.lower():
                    for i in range(n_agents):
                        selected_algorithm_agents[i].reset(int(lawn_mower_rng.uniform(0,8)), env.scenario_map)
                    consensus_safe_masking_module.update_map(env.scenario_map)
                elif 'wanderingagent' in selected_algorithm.lower():
                    for i in range(n_agents):
                        selected_algorithm_agents[i].reset(env.scenario_map)
                    consensus_safe_masking_module.update_map(env.scenario_map)
                elif 'pso' in selected_algorithm.lower():
                    selected_algorithm_agents.reset()
                    consensus_safe_masking_module.update_map(env.scenario_map)
                
                acc_rw_episode = [0 for _ in range(n_agents)]

                while any([not value for value in done.values()]):  # while at least 1 active

                    # Add step #
                    step += 1
                    
                    # Take new actions #
                    t0 = time.perf_counter()

                    if 'drl' in selected_algorithm.lower():
                        states = {agent_id: np.float16(np.uint8(state * 255)/255) for agent_id, state in states.items()} # Get the same format as training
                        actions = network.select_consensus_actions(states=states, positions=env.get_active_agents_positions_dict(), n_actions_of_each_agent=env.n_actions_of_each_agent, done = done, deterministic=True)
                    elif 'wanderingagent' in selected_algorithm.lower() or 'lawnmower' in selected_algorithm.lower():
                        actions = {agent_id: selected_algorithm_agents[agent_id].move(actual_position=position, trash_in_pixel=env.model_trash_map[position[0], position[1]]) for agent_id, position in env.get_active_agents_positions_dict().items()}
                        q_values = {agent_id: np.array([1 if i == actions[agent_id] else 0 for i in range(8)]).astype(float) for agent_id in range(n_agents)}
                        actions = consensus_safe_masking_module.query_actions(q_values=q_values, agents_positions=env.get_active_agents_positions_dict(), model_trash_map=env.model_trash_map, team_id_of_each_agent=env.team_id_of_each_agent)
                    elif 'pso' in selected_algorithm.lower():
                        q_values = selected_algorithm_agents.get_agents_q_values()
                        actions = consensus_safe_masking_module.query_actions(q_values=q_values, agents_positions=env.get_active_agents_positions_dict(), model_trash_map=env.model_trash_map, team_id_of_each_agent=env.team_id_of_each_agent)
                    elif 'greedy' in selected_algorithm.lower():
                        q_values = selected_algorithm_agents.get_agents_q_values()
                        actions = consensus_safe_masking_module.query_actions(q_values=q_values, agents_positions=env.get_active_agents_positions_dict(), model_trash_map=env.model_trash_map, team_id_of_each_agent=env.team_id_of_each_agent)
                    elif 'levywalks' in selected_algorithm.lower():
                        actions = selected_algorithm_agents.get_agents_actions()
                        q_values = {agent_id: np.array([1 if i == actions[agent_id] else 0 for i in range(8)]).astype(float) for agent_id in range(n_agents)}
                        actions = consensus_safe_masking_module.query_actions(q_values=q_values, agents_positions=env.get_active_agents_positions_dict(), model_trash_map=env.model_trash_map, team_id_of_each_agent=env.team_id_of_each_agent)

                    total_decision_counter += 1
                    if epsilon > rng_q_epsilon.random():
                        random_decision_counter += 1
                        q_values_algorithm = {agent_id: np.array([100000 if i == actions[agent_id] else 0 for i in range(env.n_actions_of_each_agent[agent_id])]).astype(float) for agent_id in range(n_agents) if env.team_id_of_each_agent[agent_id]!=epsilon_team}
                        q_values_epsilon = {agent_id: np.random.rand(env.n_actions_of_each_agent[agent_id]) for agent_id in range(n_agents) if env.team_id_of_each_agent[agent_id]==epsilon_team} # random q values
                        q_values = {**q_values_algorithm, **q_values_epsilon}
                        actions = consensus_safe_masking_module.query_actions(q_values=q_values, agents_positions=env.get_active_agents_positions_dict(), model_trash_map=env.model_trash_map, team_id_of_each_agent=env.team_id_of_each_agent)

                    t1 = time.perf_counter()
                    runtimes_dict[selected_algorithm].append(t1-t0)
                    
                    # states, rewards, done = env.step(actions)
                    states, rewards, reward_components, done = env.step(actions, return_reward_components=True)
                    acc_rw_episode = [acc_rw_episode[i] + rewards[i] for i in range(n_agents)]

                    # print(f"Step {env.steps}")
                    # print(f"Actions: {dict(sorted(actions.items()))}")
                    # print(f"Rewards: {rewards}")
                    # trashes_agents_pixels = {agent_id: env.model_trash_map[position[0], position[1]] for agent_id, position in env.get_active_agents_positions_dict().items()}
                    # print(f"Trashes in agents pixels: {trashes_agents_pixels}")
                    # print(f"Trashes removed: {env.trashes_removed_per_agent}")
                    # print(f"Trashes remaining: {len(env.trash_positions_yx)}")
                    # print()

                    # Save data #
                    if save_data:
                        data_saver.save_registers(env=env,
                                                run=run, 
                                                step=step, 
                                                algorithm=selected_algorithm, 
                                                rewards=rewards,
                                                reward_components=reward_components,
                                                actions=actions, 
                                                objective="Trash",
                                                dones=done,
                                                epsilon=epsilon,
                                                epsilon_team=epsilon_team)

                # print('Total runtime: ', runtime)
                # print('Total reward: ', acc_rw_episode)
                
            print(f"Random actions taken: {random_decision_counter} out of {total_decision_counter} decisions: {100*random_decision_counter/total_decision_counter}%")
            if save_data:
                data_saver.save()  
                
        if save_data:
            # Save runtimes_dict as json #
            with open(os.path.join(save_path, 'runtimes.json'), 'w') as f:
                json.dump(runtimes_dict, f)


    ## MAIN CONFIGURATION ##
    save_name = 'evaluation'

    EPSILON_TEST = False

    SHOW_RENDER = False
    SAVE_DATA = True
    savepath = 'Evaluation/Results/'

    RUNS = 100
    SEED = 3
    SEED_EPSILON_STUDY = 3


    if EPSILON_TEST:
        # definitive_algs = [
        #     'Training/T/T_RW_negativedijkstra_2.12_5.91_2.23_1.7_20k_ep0.75_hu4k_te5_prewarm0_comb_port/policy',
        #     'Greedy',
        #     'LevyWalks'
        #     ]
        # algorithms= [f'Eps_{round(eps, 2)}_EpsTm_{eps_team}_{alg}' for eps, eps_team in product(np.arange(0.0, 1 + 0.0001, 0.05), [0, 1]) for alg in definitive_algs]
        # evaluate_algorithms(algorithms, SHOW_RENDER, SAVE_DATA, savepath, RUNS, SEED, SEED_EPSILON_STUDY)
        for eps, eps_team in product(np.arange(0.0, 1 + 0.0001, 0.05), [0, 1]):
            eps_name = f'Eps_{round(eps, 2)}_EpsTm_{eps_team}_'

            # algorithms= [f'Training/T/{eps_name}T_RW_timenegativelogdijkstra_2.30_2.19_1.86_4.10_10k_ep0.55_hu7k_te15_prewarm0_comb_port/policy']
            algorithms= [f'Training/T/{eps_name}T_RW_negativedijkstra_2.12_5.91_2.23_1.7_20k_ep0.75_hu4k_te5_prewarm0_comb_port/policy']
            # algorithms= [f'{eps_name}PSOdijkstra']
            # algorithms= [f'{eps_name}Greedydijkstra']
            # algorithms= [f'{eps_name}LevyWalksdijkstra']
            evaluate_algorithms(algorithms, SHOW_RENDER, SAVE_DATA, savepath, RUNS, SEED, SEED_EPSILON_STUDY, extra_name=eps_name)
    else:
        EXTRA_NAME = f''
        algorithms = [
            # 'Training/T/T_RW_timenegativelogdijkstra_2.30_2.19_1.86_4.10_10k_ep0.55_hu7k_te15_prewarm0_comb_port/policy',
            'Training/T/T_RW_negativedijkstra_2.12_5.91_2.23_1.7_20k_ep0.75_hu4k_te5_prewarm0_comb_port/policy',
            # 'WanderingAgent', 
            # 'LawnMower', 
            # 'PSO', 
            'Greedy',
            'LevyWalks'
            # 'Training/T//',
            # 'Training/T/Eps_0.7_EpsTm_0_T_RW_timenegativelogdijkstra_2.30_2.19_1.86_4.10_10k_ep0.55_hu7k_te15_prewarm0_comb_port/policy',
            ]
        evaluate_algorithms(algorithms, SHOW_RENDER, SAVE_DATA, savepath, RUNS, SEED, SEED_EPSILON_STUDY, extra_name=EXTRA_NAME)






