import sys
sys.path.append('.')

from Environment.CleanupEnvironment import MultiAgentCleanupEnvironment
from Algorithms.DRL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
import numpy as np
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--scenario_map_name', type=str, default='acoruna_port', help='Name of the scenario map.')
parser.add_argument('--n_explorers', type=int, default=2, help='Number of explorers agents.')
parser.add_argument('--n_cleaners', type=int, default=2, help='Number of cleaners agents.')
parser.add_argument('--max_steps_per_episode', type=int, default=150, help='Max steps per episode.')
parser.add_argument('--dynamic_env', type=str, default='True', help='Dynamic environment.')
parser.add_argument('--obstacles', type=str, default='False', help='Obstacles in the environment.')
parser.add_argument('-rw', '--reward_function', type=str, default='negativedistance', help='Reward function to use: basic_reward, extended_reward, backtosimple')
parser.add_argument('-w', '--reward_weights', type=float, nargs='+', default=[1, 50, 2, 0], help='Reward weights for the reward function.')
parser.add_argument('-net', '--network_type', type=str, default='independent_networks_per_team', help='Type of network to use: independent_networks_per_team, shared_network')
parser.add_argument('-dev', '--device', type=str, default='cuda:0', help='Device to use: cuda:x, cpu')
parser.add_argument('--epsilon', type=float, default=0.5, help='Epsilon value for epsilon-greedy training.')
parser.add_argument('-eps', '--episodes', type=int, default=60000, help='Number of episodes to train the network.')
parser.add_argument('--extra_episodes', type=int, default=0, help='Extra episodes to keep training after the first training.')
parser.add_argument('-gt', '--greedy_training', type=str, default="False", help='Use greedy training instead of epsilon-greedy training.')
parser.add_argument('--pso_training', type=str, default="False", help='Use PSO training instead of epsilon-greedy training.')
parser.add_argument('-t', '--target_update', type=int, default=6000, help='Number of steps to update the target network.')
parser.add_argument('--train_every', type=int, default=15, help='Number of steps to train the network.')
parser.add_argument('--extra_name', type=str, default='', help='Extra name to add to the logdir.')
parser.add_argument('--preload_path', type=str, default='', help='Path to preload a model.')
parser.add_argument('--prewarm_percentage', type=float, default=0, help='Percentage of memory to prewarm with Greedy actions.')
parser.add_argument('--path_optuna_params', type=str, default='', help='Load parameters from Optuna.')
args = parser.parse_args()

# Selection of PARAMETERS TO TRAIN #
memory_size = int(1E6)
reward_function = args.reward_function
reward_weights = tuple(args.reward_weights) 
network_type = args.network_type
device = args.device
epsilon = args.epsilon
episodes = args.episodes
extra_episodes = args.extra_episodes
greedy_training = True if args.greedy_training.capitalize() == "True" else False
pso_training = True if args.pso_training.capitalize() == "True" else False
target_update = args.target_update
train_every = args.train_every
preload_path = args.preload_path
prewarm_percentage = args.prewarm_percentage

# Load parameters from Optuna #
if args.path_optuna_params:
	path_optuna_params = args.path_optuna_params
	if path_optuna_params[-1] != '/':
		path_optuna_params += '/'
	f = open(path_optuna_params + 'best_trial_optuna.json',)
	best_params = json.load(f)
	f.close()
	best_params = best_params['params']
	reward_weights = tuple([best_params['reward_weight_1'], best_params['reward_weight_2'], best_params['reward_weight_3'], best_params['reward_weight_4']])
	epsilon = best_params['epsilon']
	gamma = best_params['gamma']
	lr = best_params['learning_rate']
	target_update = best_params['target_update']
	train_every = best_params['train_every']
	batch_size = best_params['batch_size']




SHOW_PLOT_GRAPHICS = False
seed = 0

# Agents info #
n_actions_explorers = 8
n_actions_cleaners = 8
n_explorers = args.n_explorers
n_cleaners = args.n_cleaners
n_agents = n_explorers + n_cleaners
movement_length_explorers = 2
movement_length_cleaners = 1
movement_length_of_each_agent = np.repeat((movement_length_explorers, movement_length_cleaners), (n_explorers, n_cleaners))
vision_length_explorers = 4
vision_length_cleaners = 1
max_distance_travelled_explorers = 400
max_distance_travelled_cleaners = 200
max_steps_per_episode = args.max_steps_per_episode
scenario_map_name = args.scenario_map_name


# Set initial positions #
random_initial_positions = True 
if random_initial_positions:
	initial_positions = 'fixed'
else:
	# initial_positions = np.array([[46, 28], [46, 31], [49, 28], [49, 31]])[:n_agents, :] #ypacarai_map
	initial_positions = np.array([[32, 7], [30, 7], [28, 7], [26, 7]])[:n_agents, :] #coruna_port

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
							max_collisions = 10,
							reward_function = reward_function, 
							reward_weights = reward_weights,
							dynamic = True if args.dynamic_env.capitalize() == "True" else False,
							obstacles = True if args.obstacles.capitalize() == "True" else False,
							show_plot_graphics = SHOW_PLOT_GRAPHICS,
							)

# Network config:
if network_type == 'independent_networks_per_team':
	independent_networks_per_team = True
else:
	independent_networks_per_team = False

if memory_size == int(1E3):
	logdir = f'testing/Training_{network_type.split("_")[0]}_RW_{reward_function.split("_")[0]}_' + '_'.join(map(str, reward_weights)) + args.extra_name
else:
	if greedy_training:
		training_type = "_greedy"
	elif pso_training:
		training_type = "_pso"
	else:
		training_type = ""

	if n_cleaners == 0:
		logdir = f'Training/T{training_type}_curriculum_explorers_RW_{reward_function.split("_")[0]}_' + '_'.join(map(str, reward_weights)) + f'_{int(episodes/1000)}k{f"+{int(extra_episodes/1000)}k" if extra_episodes>0 else ""}_ep{epsilon}_hu{int(target_update/1000)}k_te{train_every}_prewarm{prewarm_percentage}_{scenario_map_name}' + args.extra_name
	elif n_explorers == 0:
		logdir = f'Training/T{training_type}_curriculum_cleaners_RW_{reward_function.split("_")[0]}_' + '_'.join(map(str, reward_weights)) + f'_{int(episodes/1000)}k{f"+{int(extra_episodes/1000)}k" if extra_episodes>0 else ""}_ep{epsilon}_hu{int(target_update/1000)}k_te{train_every}_prewarm{prewarm_percentage}_{scenario_map_name}' + args.extra_name
	else:
		logdir = f'Training/T{training_type}_RW_{reward_function.split("_")[0]}_' + '_'.join(map(str, reward_weights)) + f'_{int(episodes/1000)}k{f"+{int(extra_episodes/1000)}k" if extra_episodes>0 else ""}_ep{epsilon}_hu{int(target_update/1000)}k_te{train_every}_prewarm{prewarm_percentage}_{scenario_map_name}' + args.extra_name

network = MultiAgentDuelingDQNAgent(env=env,
									memory_size=memory_size, 
									batch_size=128,
									target_update=target_update,
									soft_update=False,
									tau=0.001, 
									epsilon_values=[1.0, 0.05],
									epsilon_interval=[0.0, epsilon], #0.5
									greedy_training=greedy_training, # epsilon is used to take to take greedy actions policy during training instead of random
									pso_training=pso_training, # epsilon is used to take to take greedy or pso actions policy during training instead of random
									learning_starts=100, 
									gamma=0.99,
									lr=1e-4,
									save_every=10000, # 5000
									train_every=train_every, #15 (steps)
									masked_actions=False,
									consensus_actions=True,
									device=device,
									logdir=logdir,
									eval_every=100, #1000
									eval_episodes=100, # 10
									prewarm_percentage=prewarm_percentage,
									noisy=False,
									distributional=False,
									independent_networks_per_team = independent_networks_per_team,
									curriculum_learning_team=None, # env.cleaners_team_id,
)
if preload_path:
	network.load_model(preload_path)
network.train(episodes=episodes, extra_episodes=extra_episodes) 