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

from Environment.CleanupEnvironment import MultiAgentCleanupEnvironment
from Evaluation.Utils.metrics_wrapper import MetricsDataCreator

from sylegendarium.legendarium import Legendarium
import numbers

class AlgorithmRecorder:
    def __init__(self, savepath, map_name, teams_name, team_id_of_each_agent, max_distance_by_team, max_steps_per_episode, objectives, n_agents, runs, experiment_name='legendarium'):
        ''' Inicializa el almacenamiento de métricas con legendarium. '''

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

        # Create a metric rekated to the algorithm
        self.data_saver.create_metric("algorithm", str, "Algorithm used in the experiment", "-")
        self.data_saver.create_metric("reward_function", str, "Reward function used in the experiment", "-")
        self.data_saver.create_metric("reward_weights", tuple, "Weights of the reward function", "-")

        # Create metrics related to the agents
        self.data_saver.create_metric("agents_positions", dict, "Agents positions in the map", "Map units")
        self.data_saver.create_metric("actions", dict, "Actions taken by the agents", "Action units")
        self.data_saver.create_metric("dones", dict, "Dones of the agents", "-")
        self.data_saver.create_metric("rewards", dict, "Rewards obtained by the agents", "points")
        self.data_saver.create_metric("traveled_distances", dict, "Distances travelled by the agents", "Distance units")
        self.data_saver.create_metric("trashes_at_sight", dict, "Trashes at sight for each agent", "Trashes")
        self.data_saver.create_metric("cleaned_trashes", dict, "Cleaned trashes by each agent", "Trashes")
        self.data_saver.create_metric("history_cleaned_trashes", dict, "History of cleaned trashes by each agent", "Trashes")
        self.data_saver.create_metric("trash_remaining_info", np.ndarray, "Information about the remaining trash", "step_discover, vehicle_discover")
        self.data_saver.create_metric("trash_removed_info", np.ndarray, "Information about the removed trashes", "step_discover, vehicle_discover, step_remove, vehicle_remove")

        # Create metrics related to the model
        self.data_saver.create_metric("objective", str, "Name of the objective", "-")
        self.data_saver.create_metric("ground_truth", np.ndarray, "Ground truth", "Objective units")
        self.data_saver.create_metric("model", np.ndarray, "Model", "Objective units")
        self.data_saver.create_metric("trash_remaining", numbers.Real, "Number of trashes remaining in the map", "Trashes")
        self.data_saver.create_metric("percentage_of_trash_collected", numbers.Real, "Percentage of trash collected", "Percentage")
        self.data_saver.create_metric("percentage_of_trash_discovered", numbers.Real, "Percentage of trash discovered", "Percentage")
        self.data_saver.create_metric("model_mse", numbers.Real, "MSE of the model", "Objective units")
        self.data_saver.create_metric("sum_model_changes", numbers.Real, "Sum of model changes", "Objective units")
        
    def save_registers(self, run = None, step = None, env = None, algorithm = None, rewards=None, actions=None, objective = None, dones = None):

        self.data_saver.write(run=run, 
                        step=step, 
                        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        algorithm=algorithm,
                        reward_function=env.reward_function,
                        reward_weights=env.reward_weights,
                        agents_positions=env.get_active_agents_positions_dict(),
                        actions=actions,
                        dones=dones,
                        rewards=rewards,
                        traveled_distances=env.get_traveled_distances(),
                        trashes_at_sight=env.get_trashes_at_sight(),
                        cleaned_trashes=env.trashes_removed_per_agent,
                        history_cleaned_trashes=env.history_trashes_removed_per_agent,
                        trash_remaining_info=env.trash_remaining_info,
                        trash_removed_info=env.trash_removed_info,
                        objective=objective,
                        ground_truth=env.real_trash_map,
                        model=env.model_trash_map,
                        trash_remaining=len(env.trash_positions_yx),
                        percentage_of_trash_collected=env.get_percentage_cleaned_trash(),
                        percentage_of_trash_discovered=env.get_percentage_discovered_trash(),
                        model_mse=env.get_model_mse(),
                        sum_model_changes=env.get_changes_in_model(),
                        )
        
    def save(self):
        ''' Save the data in the legendarium format. '''
        self.data_saver.save()

    def plot_and_tables_metrics_average(self, metrics_path, table, wilcoxon_dict, show_plot = True , save_plot = False):
 
        metrics_df = MetricsDataCreator.load_csv_as_df(metrics_path)
        self.runs = metrics_df['Run'].unique()

        # Obtain dataframes #
        numeric_columns = metrics_df.select_dtypes(include=[np.number])
        # Padding each episode with less steps than the max_steps_per_episode with the last value in the episode #
        numeric_columns = numeric_columns.groupby('Run').apply(lambda group: group.set_index('Step').reindex(range(self.env.max_steps_per_episode+1), method='ffill').reset_index()).reset_index(drop=True)
        # Calculate mean and std #
        self.results_mean = numeric_columns.groupby('Step').agg('mean')
        self.results_std = numeric_columns.groupby('Step').agg('std')

        # Extract data to plot or save fig metrics #
        if show_plot or save_plot:
            first_accreward_agent_index = self.results_mean.columns.get_loc('AccRw0')
            self.reward_agents_acc = self.results_mean.iloc[:, first_accreward_agent_index:first_accreward_agent_index + self.n_agents].values.tolist()
            self.reward_acc = self.results_mean['R_acc'].values.tolist()
            self.mse = self.results_mean['MSE'].values.tolist()
            self.sum_model_changes = self.results_mean['Sum_model_changes'].values.tolist()
            self.trash_remaining = self.results_mean['Trash_remaining'].values.tolist()
            self.percentage_of_trash_collected = self.results_mean['Percentage_of_trash_collected'].values.tolist()
            self.traveled_distance = self.results_mean['Traveled_distance'].values.tolist()
            self.max_redundancy = self.results_mean['Max_Redundancy'].values.tolist()
            first_traveldist_agent_index = self.results_mean.columns.get_loc('TravelDist0')
            self.traveled_distance_agents = self.results_mean.iloc[:, first_traveldist_agent_index:first_traveldist_agent_index + self.n_agents].values.tolist()
            if self.n_agents > 1:
                first_distbetween_index = self.results_mean.columns.get_loc([*env.fleet.get_distances_between_agents().keys()][0])
                self.distances_between_agents = self.results_mean.iloc[:, first_distbetween_index:first_distbetween_index + int((self.n_agents*(self.n_agents-1))/2)].values.tolist()

            self.plot_metrics(show_plot=show_plot, save_plot=save_plot, plot_std=True)
            plt.close('all')

        # TABLE OF METRICS #
        import warnings
        warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

        name_alg = self.algorithm.split('_')[0].capitalize()
        name_rw = self.reward_funct.split('_')[0].capitalize() + '-' + '-'.join(map(str, reward_weights))

        if not name_alg.capitalize() in table:
            new_df = pd.DataFrame(
                    columns=pd.MultiIndex.from_product([[name_alg], ["Mean33", "CI33 95%", "Mean66", "CI66 95%", "Mean100", "CI100 95%"]]),
                    index=table.index)
            table = pd.concat([table, new_df], axis=1)

        # Calculate the some metrics at 33%, 66% and 100% of each episode and add to table #
        mse = {f'{percentage}%': np.array(numeric_columns[numeric_columns['Step']==round(self.env.max_steps_per_episode*percentage/100)]['MSE']) for percentage in [33, 66, 100]}
        smc = {f'{percentage}%': np.array(numeric_columns[numeric_columns['Step']==round(self.env.max_steps_per_episode*percentage/100)]['Sum_model_changes']) for percentage in [33, 66, 100]}
        ptc = {f'{percentage}%': np.array(numeric_columns[numeric_columns['Step']==round(self.env.max_steps_per_episode*percentage/100)]['Percentage_of_trash_collected']) for percentage in [33, 66, 100]}
        accrw0 = {f'{percentage}%': np.array(numeric_columns[numeric_columns['Step']==round(self.env.max_steps_per_episode*percentage/100)]['AccRw0']) for percentage in [33, 66, 100]}
        accrw1 = {f'{percentage}%': np.array(numeric_columns[numeric_columns['Step']==round(self.env.max_steps_per_episode*percentage/100)]['AccRw1']) for percentage in [33, 66, 100]}
        accrw2 = {f'{percentage}%': np.array(numeric_columns[numeric_columns['Step']==round(self.env.max_steps_per_episode*percentage/100)]['AccRw2']) for percentage in [33, 66, 100]}
        accrw3 = {f'{percentage}%': np.array(numeric_columns[numeric_columns['Step']==round(self.env.max_steps_per_episode*percentage/100)]['AccRw3']) for percentage in [33, 66, 100]}

        table.loc['MSE-'+name_rw, name_alg] = [np.mean(mse['33%']), 1.96*np.std(mse['33%'])/np.sqrt(len(self.runs)), np.mean(mse['66%']),1.96*np.std(mse['66%'])/np.sqrt(len(self.runs)), np.mean(mse['100%']), 1.96*np.std(mse['100%'])/np.sqrt(len(self.runs))]
        table.loc['SumModelChanges-'+name_rw, name_alg] = [np.mean(smc['33%']), 1.96*np.std(smc['33%'])/np.sqrt(len(self.runs)), np.mean(smc['66%']),1.96*np.std(smc['66%'])/np.sqrt(len(self.runs)), np.mean(smc['100%']), 1.96*np.std(smc['100%'])/np.sqrt(len(self.runs))]
        table.loc['PercentageOfTrashCollected-'+name_rw, name_alg] = [np.mean(ptc['33%']), 1.96*np.std(ptc['33%'])/np.sqrt(len(self.runs)), np.mean(ptc['66%']),1.96*np.std(ptc['66%'])/np.sqrt(len(self.runs)), np.mean(ptc['100%']), 1.96*np.std(ptc['100%'])/np.sqrt(len(self.runs))]
        table.loc['AccumulatedReward0-'+name_rw, name_alg] = [np.mean(accrw0['33%']), 1.96*np.std(accrw0['33%'])/np.sqrt(len(self.runs)), np.mean(accrw0['66%']),1.96*np.std(accrw0['66%'])/np.sqrt(len(self.runs)), np.mean(accrw0['100%']), 1.96*np.std(accrw0['100%'])/np.sqrt(len(self.runs))]
        table.loc['AccumulatedReward1-'+name_rw, name_alg] = [np.mean(accrw1['33%']), 1.96*np.std(accrw1['33%'])/np.sqrt(len(self.runs)), np.mean(accrw1['66%']),1.96*np.std(accrw1['66%'])/np.sqrt(len(self.runs)), np.mean(accrw1['100%']), 1.96*np.std(accrw1['100%'])/np.sqrt(len(self.runs))]
        table.loc['AccumulatedReward2-'+name_rw, name_alg] = [np.mean(accrw2['33%']), 1.96*np.std(accrw2['33%'])/np.sqrt(len(self.runs)), np.mean(accrw2['66%']),1.96*np.std(accrw2['66%'])/np.sqrt(len(self.runs)), np.mean(accrw2['100%']), 1.96*np.std(accrw2['100%'])/np.sqrt(len(self.runs))]
        table.loc['AccumulatedReward3-'+name_rw, name_alg] = [np.mean(accrw3['33%']), 1.96*np.std(accrw3['33%'])/np.sqrt(len(self.runs)), np.mean(accrw3['66%']),1.96*np.std(accrw3['66%'])/np.sqrt(len(self.runs)), np.mean(accrw3['100%']), 1.96*np.std(accrw3['100%'])/np.sqrt(len(self.runs))]

        return table

        

if __name__ == '__main__':

    import time
    from Algorithms.LawnMower import LawnMowerAgent
    from Algorithms.NRRA import WanderingAgent
    from Algorithms.PSO import ParticleSwarmOptimizationFleet
    from Algorithms.Greedy import OneStepGreedyFleet
    from Algorithms.DRL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
    from Algorithms.DRL.ActionMasking.ActionMaskingUtils import ConsensusSafeActionMasking

    algorithms = [
        # 'Training/T/Definitivos/acoruna_drl_alone/policy',
        # 'Training/T/Definitivos/acoruna_greedy_training/policy',
        'WanderingAgent', 
        'LawnMower', 
        'PSO', 
        'Greedy',
        # 'Training/T/Definitivos/combport_drl_alone/policy',
        # 'Training/T/Definitivos/combport_greedy_training/policy',
        # 'Training/T//',
        ]

    SHOW_RENDER = False
    SAVE_DATA = True

    RUNS = 5
    SEED = 3

    EXTRA_NAME = ''
    # EXTRA_NAME = 'Final_Policy'
    # EXTRA_NAME = 'BestPolicy'
    # EXTRA_NAME = 'BestEvalPolicy'
    # EXTRA_NAME = 'BestEvalCleanPolicy'
    # EXTRA_NAME = 'BestEvalMSEPolicy'
    
    runtimes_dict = {}

    first_iteration = True
    for path_to_training_folder in algorithms:
        print(f"Evaluating algorithm: {path_to_training_folder}")
        ## LITERATURE ALGORITHMS ##
        if path_to_training_folder in ['WanderingAgent', 'LawnMower', 'PSO', 'Greedy']:
            selected_algorithm = path_to_training_folder

            # If any of the algorithms contains 'Training', extract the scenario_map_name and reward weights from the environment_config.json #
            if np.any([alg.find('Training') != -1 for alg in algorithms]):
                # Get one that contains 'Training' #
                path = [alg for alg in algorithms if alg.find('Training') != -1][0]
                path = '/'.join(path.split('/')[:-1]) + '/'
                f = open(path + 'environment_config.json',)
                config = json.load(f)
                f.close()
                scenario_map_name = config['scenario_map_name']
                reward_weights= tuple(config['reward_weights'])
                if 'greedyastar' in path_to_training_folder.lower():
                    reward_function = 'negativeastar'
                elif 'greedy' in path_to_training_folder.lower():
                    reward_function = 'negativedistance'
                else:
                    reward_function = config['reward_function']
                n_explorers = env_config['number_of_agents_by_team'][0]
                n_cleaners = env_config['number_of_agents_by_team'][1]
                obstacles = env_config['obstacles']
            else:
                scenario_map_name = 'acoruna_port' # 'ypacarai_lake', 'acoruna_port', 'marinapalamos', 'comb_port'
                reward_weights=(1, 50, 2, 0)
                if 'greedyastar' in path_to_training_folder.lower():
                    reward_function = 'negativeastar'
                elif 'greedy' in path_to_training_folder.lower():
                    reward_function = 'negativedistance' 
                else:
                    reward_function = 'negativedijkstra' # 'negativedistance', 'negativedijkstra', 'negativeastar'
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
                                    seed = SEED,
                                    movement_length_by_team =  (movement_length_explorers, movement_length_cleaners),
                                    vision_length_by_team = (vision_length_explorers, vision_length_cleaners),
                                    flag_to_check_collisions_within = False,
                                    max_collisions = 1000,
                                    reward_function = reward_function,
                                    reward_weights = reward_weights,
                                    dynamic = True,
                                    obstacles = False,
                                    show_plot_graphics = SHOW_RENDER,
                                    )
            scenario_map = env.scenario_map
            scenario_map_name = env.scenario_map_name
            
            if selected_algorithm == "LawnMower":
                lawn_mower_rng = np.random.default_rng(seed=100)
                selected_algorithm_agents = [LawnMowerAgent(world=scenario_map, number_of_actions=8, movement_length=movement_length_of_each_agent[i], forward_direction=int(lawn_mower_rng.uniform(0,8)), seed=SEED+i, agent_is_cleaner=env.team_id_of_each_agent[i]==env.cleaners_team_id) for i in range(n_agents)]
            elif selected_algorithm == "WanderingAgent":
                selected_algorithm_agents = [WanderingAgent(world=scenario_map, number_of_actions=8, movement_length=movement_length_of_each_agent[i], seed=SEED+i, agent_is_cleaner=env.team_id_of_each_agent[i]==env.cleaners_team_id) for i in range(n_agents)]
            elif selected_algorithm == "PSO":
                selected_algorithm_agents = ParticleSwarmOptimizationFleet(env)
                consensus_safe_masking_module = ConsensusSafeActionMasking(navigation_map = scenario_map, angle_set_of_each_agent=env.angle_set_of_each_agent, movement_length_of_each_agent = env.movement_length_of_each_agent)
            elif selected_algorithm == "Greedy":
                selected_algorithm_agents = OneStepGreedyFleet(env)

        ## DEEP REINFORCEMENT LEARNING ALGORITHMS ##
        else:
            # Load env config #
            model_policy = path_to_training_folder.split('/')[-1]
            path_to_training_folder = '/'.join(path_to_training_folder.split('/')[:-1]) + '/'
            f = open(path_to_training_folder + 'environment_config.json',)
            env_config = json.load(f)
            f.close()
            
            env = MultiAgentCleanupEnvironment(scenario_map_name = env_config['scenario_map_name'],
                                    number_of_agents_by_team=env_config['number_of_agents_by_team'],
                                    n_actions_by_team=env_config['n_actions'],
                                    max_distance_travelled_by_team = env_config['max_distance_travelled_by_team'],
                                    max_steps_per_episode=env_config['max_steps_per_episode'],
                                    fleet_initial_positions = env_config['fleet_initial_positions'], # np.array(env_config['fleet_initial_positions']), #
                                    seed = SEED,
                                    movement_length_by_team =  env_config['movement_length_by_team'],
                                    vision_length_by_team = env_config['vision_length_by_team'],
                                    flag_to_check_collisions_within = env_config['flag_to_check_collisions_within'],
                                    max_collisions = env_config['max_collisions'],
                                    reward_function = env_config['reward_function'],
                                    reward_weights = tuple(env_config['reward_weights']),
                                    dynamic = env_config['dynamic'],
                                    obstacles = env_config['obstacles'],
                                    show_plot_graphics = SHOW_RENDER,
                                    )
            scenario_map = env.scenario_map
            scenario_map_name = env.scenario_map_name
            n_agents = env.n_agents
            reward_function = env.reward_function
            reward_weights = env.reward_weights
            
            # Load exp config #
            f = open(path_to_training_folder + 'experiment_config.json',)
            exp_config = json.load(f)
            f.close()

            independent_networks_per_team = exp_config['independent_networks_per_team']
            greedy_training = exp_config['greedy_training']

            if independent_networks_per_team and not greedy_training:
                selected_algorithm = "DRLIndependent_Networks_Per_Team"
            elif independent_networks_per_team and greedy_training:
                selected_algorithm = "DRLIndependentgreedy"
            else:
                selected_algorithm = "DRLNetwork"
                # raise NotImplementedError("This algorithm is not implemented. Choose one that is.")

            network = MultiAgentDuelingDQNAgent(env=env,
                                    memory_size=int(1E3),  #int(1E6), 1E5
                                    batch_size=exp_config['batch_size'],
                                    target_update=1000,
                                    seed = SEED,
                                    consensus_actions=exp_config['consensus_actions'],
                                    device='cuda:0',
                                    independent_networks_per_team = independent_networks_per_team,
                                    )
            network.load_model(path_to_training_folder + f'{model_policy}.pth')
            network.epsilon = 0

        if SAVE_DATA and first_iteration:
            # Reward function and create path to save #
            savepath = f'Evaluation/Results/{EXTRA_NAME}' + scenario_map_name + '.' + str(n_agents) + '.' + reward_function.split('_')[0] + '_' + '_'.join(map(str, reward_weights))
            if not(os.path.exists(savepath)): # create the directory if not exists
                os.mkdir(savepath)

            data_saver = AlgorithmRecorder(experiment_name=experiment_name,
                                           savepath=savepath,
                                            map_name=scenario_map_name,
                                            teams_name={0: 'Explorers', 1: 'Cleaners'},
                                            team_id_of_each_agent={idx: team_id for idx, team_id in enumerate(env.team_id_of_each_agent)},
                                            max_distance_by_team={idx: dist for idx, dist in enumerate(env.max_distance_travelled_by_team)},
                                            max_steps_per_episode=env.max_steps_per_episode,
                                            objectives=['Trash'],
                                            n_agents=n_agents,
                                            runs=RUNS)
            env.save_environment_configuration(savepath)
            first_iteration = False

        runtimes_dict[selected_algorithm] = []

        env.reset_seeds()

        # START EPISODES #
        for run in trange(RUNS):
            
            done = {i: False for i in range(n_agents)}
            states = env.reset_env()

            runtime = 0
            step = 0

            # Save data #
            if SAVE_DATA:
                data_saver.save_registers(env=env,
                                        run=run, 
                                        step=step, 
                                        algorithm=selected_algorithm, 
                                        rewards={i: 0 for i in range(n_agents)}, 
                                        actions={i: None for i in range(n_agents)}, 
                                        objective="Trash",
                                        dones=done)

            # Reset algorithms #
            if 'DRL' in selected_algorithm:
                network.nogobackfleet_masking_module.reset()
            elif selected_algorithm in ['LawnMower']:
                for i in range(n_agents):
                    selected_algorithm_agents[i].reset(int(lawn_mower_rng.uniform(0,8)) if selected_algorithm == 'LawnMower' else None)
            elif selected_algorithm in ['PSO']:
                selected_algorithm_agents.reset()
            
            acc_rw_episode = [0 for _ in range(n_agents)]

            while any([not value for value in done.values()]):  # while at least 1 active

                # Add step #
                step += 1
                
                # Take new actions #
                t0 = time.perf_counter()
                if 'DRL' in selected_algorithm:
                    actions = network.select_consensus_actions(states=states, positions=env.get_active_agents_positions_dict(), n_actions_of_each_agent=env.n_actions_of_each_agent, done = done, deterministic=True)
                elif selected_algorithm in ['WanderingAgent', 'LawnMower']:
                    actions = {agent_id: selected_algorithm_agents[agent_id].move(actual_position=position, trash_in_pixel=env.model_trash_map[position[0], position[1]]) for agent_id, position in env.get_active_agents_positions_dict().items()}
                elif selected_algorithm == 'PSO':
                    q_values = selected_algorithm_agents.get_agents_q_values()
                    actions = consensus_safe_masking_module.query_actions(q_values=q_values, agents_positions=env.get_active_agents_positions_dict(), model_trash_map=env.model_trash_map, team_id_of_each_agent=env.team_id_of_each_agent)
                elif selected_algorithm == 'Greedy':
                    actions = selected_algorithm_agents.get_agents_actions()
                    # q_values = selected_algorithm_agents.get_agents_q_values()
                    # actions_qs = {agent_id: max(q_values[agent_id], key=q_values[agent_id].get) for agent_id in q_values.keys()}
                t1 = time.perf_counter()
                runtimes_dict[selected_algorithm].append(t1-t0)
                
                states, rewards, done = env.step(actions)
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
                if SAVE_DATA:
                    data_saver.save_registers(env=env,
                                            run=run, 
                                            step=step, 
                                            algorithm=selected_algorithm, 
                                            rewards=rewards, 
                                            actions=actions, 
                                            objective="Trash",
                                            dones=done)

            # print('Total runtime: ', runtime)
            # print('Total reward: ', acc_rw_episode)
            

        if SAVE_DATA:
            data_saver.save()  
            
    if SAVE_DATA:
        # Save runtimes_dict as json #
        with open(os.path.join(savepath, 'runtimes.json'), 'w') as f:
            json.dump(runtimes_dict, f)