import sys
sys.path.append('.')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import colorcet

import torch

from sklearn.metrics import mean_squared_error, root_mean_squared_error
import json

from scipy.ndimage import gaussian_filter
import heapq

from Algorithms.a_star2 import a_star_find_path


class DiscreteVehicle: # class for single vehicle

	def __init__(self, initial_position, team_id, n_actions, movement_length, vision_length, navigation_map):
		
		""" Initial positions of the drones """
		self.initial_position = initial_position
		self.actual_agent_position = np.copy(initial_position)
		self.previous_agent_position = np.copy(self.actual_agent_position)

		""" Initialize the waypoints """
		self.waypoints = np.expand_dims(np.copy(initial_position), 0)

		""" Set other variables """
		self.team_id = team_id
		self.navigation_map = navigation_map 
		self.distance_traveled = 0.0 
		self.num_of_collisions = 0 # 
		self.movement_length = movement_length 
		self.vision_length = vision_length 
		self.influence_mask = self.compute_influence_mask()
		if n_actions == 8:
			self.angle_set = np.linspace(0, 2 * np.pi, n_actions, endpoint=False) # array with the 8 cardinal points in RADIANS, dividing a circle in 8 directions: [0. , 0.78539816, 1.57079633, 2.35619449, 3.14159265, 3.92699082, 4.71238898, 5.49778714]
			self.movement_set = np.array([np.round([np.cos(angle), np.sin(angle)]) * self.movement_length for angle in self.angle_set]).astype(int)
		elif n_actions == 9: # there is an action to stay in the same position (loiter mode)
			self.angle_set = np.linspace(0, 2 * np.pi, n_actions-1, endpoint=False)
			self.angle_set = np.append(self.angle_set, -1) # add the action to loter mode (-1)
		elif n_actions == 10: # there is an action to stay in the same position (loiter mode) and another to clean
			self.angle_set = np.linspace(0, 2 * np.pi, n_actions-2, endpoint=False)
			self.angle_set = np.append(self.angle_set, [-1, -2]) # loiter mode = -1, clean = -2
		

	def move_agent(self, action, valid=True):
		""" Move a vehicle in the direction of the action. If valid is False, the action is not performed. """

		self.previous_agent_position = np.copy(self.actual_agent_position)
		next_position = self.calculate_next_position(action)
		if action == 9: # if action is to clean, the agent stays in the same position and consume distance traveled
			self.distance_traveled += self.movement_length
		elif action == 8: # if action is to stay in the same position, the agent stays in the same position and consume just a little distance (loiter mode)
			self.distance_traveled += self.movement_length/5
		else:  # add to the total traveled distance the distance of the actual movement
			self.distance_traveled += np.linalg.norm(self.actual_agent_position - next_position)

		if self.check_agent_collision_with_obstacle(next_position) or not valid: # if next positions is a collision (with ground or between agents):
			collide = True
			self.num_of_collisions += 1 # add a collision to the count, but not move the vehicle
		else:
			collide = False
			self.actual_agent_position = next_position # set next position to actual
			self.waypoints = np.vstack((self.waypoints, [self.actual_agent_position])) # add actual position to visited locations array

		self.influence_mask = self.compute_influence_mask() # update influence mask after movement

		return collide # returns if the agent collide
	
	def calculate_next_position(self, action):
		angle = self.angle_set[action] # takes as the angle of movement the angle associated with the action taken, the action serves as the index of the array of cardinal points
		if angle < 0: # if angle is negative, the action is to stay in the same position or clean
			next_position = self.actual_agent_position
		else:
			movement = (np.round([np.cos(angle), np.sin(angle)]) * self.movement_length).astype(int) # converts the angle into cartesian motion (how many cells are moved in x-axis and how many in y-axis).
			next_position = self.actual_agent_position + movement # next position, adds the movement to the current one
			next_position = np.clip(next_position, (0,0), np.array(self.navigation_map.shape)-1) # saturate movement if out of indexes values (map edges)

		return next_position
		
	def check_agent_collision_with_obstacle(self, next_position):
		""" Return True if the next position leads to a collision """

		# Check limits of the map #
		if (next_position[0] < 0) or (next_position[0] >= self.navigation_map.shape[0]) or (next_position[1] < 0) or (next_position[1] >= self.navigation_map.shape[1]):
			return True
		# Check if position is navigable or reachable directly #
		if self.navigation_map[int(next_position[0]), int(next_position[1])] == 0: # if 0 in map, there's obstacle
			return True
		if not self.is_reachable(next_position):
			return True

		return False
	
	def check_agent_action_with_obstacle(self, test_action):
		""" Return True if the action leads to a collision """
# 
		next_position = self.calculate_next_position(test_action)

		return self.check_agent_collision_with_obstacle(next_position) 
	
	def is_reachable(self, next_position):
		""" Check if the next position is reachable """
		next_x, next_y = next_position
		dx = next_x - self.actual_agent_position[0]
		dy = next_y - self.actual_agent_position[1]
		steps = max(abs(dx), abs(dy))
		dx = dx / steps if steps != 0 else 0
		dy = dy / steps if steps != 0 else 0
		reachable = True
		for step in range(1, steps + 1):
			px = round(self.actual_agent_position[0] + dx * step)
			py = round(self.actual_agent_position[1] + dy * step)
			if self.navigation_map[px, py] != 1: # if 0 in map, there's obstacle
				reachable = False
				break

		return reachable		

	def compute_influence_mask(self):
		""" Compute influence area around actual position. It is what the agent can see. """

		influence_mask = np.zeros_like(self.navigation_map)

		pose_x, pose_y = self.actual_agent_position.astype(int)

		# State - coverage area #
		range_x_axis = np.arange(0, self.navigation_map.shape[0])
		range_y_axis = np.arange(0, self.navigation_map.shape[1])

		# Compute the circular mask (area) #
		mask = (range_x_axis[np.newaxis, :] - pose_x) ** 2 + (range_y_axis[:, np.newaxis] - pose_y) ** 2 <= self.vision_length ** 2

		influence_mask[mask.T] = 1.0 # converts True values to 1 and False values to 0

		# Remove unreachable positions #
		influence_mask = influence_mask * self.navigation_map
		for pose_x, pose_y in np.argwhere(influence_mask == 1):
			if not self.is_reachable([pose_x, pose_y]):
				influence_mask[pose_x, pose_y] = 0
		return influence_mask	
	
	def reset_agent(self, initial_position):
		""" Reset the agent: Position, waypoints, influence mask, etc. """

		self.initial_position = initial_position
		self.actual_agent_position = np.copy(initial_position)
		self.previous_agent_position = np.copy(self.actual_agent_position)
		self.waypoints = np.expand_dims(np.copy(initial_position), 0)
		self.distance_traveled = 0.0
		self.num_of_collisions = 0
		self.influence_mask = self.compute_influence_mask()


class DiscreteFleet: # class to create FLEETS of class DiscreteVehicle
	""" Coordinator of the movements of the fleet. """

	def __init__(self,
				 number_of_vehicles,
				 team_id_of_each_agent,
				 fleet_initial_positions,
				 n_actions_of_each_agent,
				 movement_length_of_each_agent,
				 vision_length_of_each_agent,
				 navigation_map,
				 check_collisions_within):

		""" Set init variables """
		self.number_of_vehicles = number_of_vehicles
		self.team_id_of_each_agent = team_id_of_each_agent
		self.initial_position_of_each_agent = fleet_initial_positions
		self.n_actions_of_each_agent = n_actions_of_each_agent 
		self.movement_length_of_each_agent = movement_length_of_each_agent 
		self.vision_length_of_each_agent = vision_length_of_each_agent 
		self.check_collisions_within = check_collisions_within


		""" Create the vehicles object array """
		self.vehicles = [DiscreteVehicle(initial_position = self.initial_position_of_each_agent[idx],
										 team_id = self.team_id_of_each_agent[idx],
										 n_actions = self.n_actions_of_each_agent[idx],
										 movement_length = self.movement_length_of_each_agent[idx],
										 vision_length = self.vision_length_of_each_agent[idx],
										 navigation_map = navigation_map) for idx in range(self.number_of_vehicles)]

		self.fleet_positions = np.asarray([veh.actual_agent_position for veh in self.vehicles])

		# Reset fleet number of collisions #
		self.fleet_collisions = 0
									
	def check_collision_within_fleet(self, veh_actions):
		""" Check if there is any collision between agents. Returns boolean array with True to the vehicles with unique new position, i.e., valid actions. """
		
		new_positions = []

		for idx, veh_action in veh_actions.items():
			# Calculate next positions #
			new_positions.append(self.vehicles[idx].calculate_next_position(veh_action))

		_, inverse_index, counts = np.unique(np.asarray(new_positions), return_inverse=True, return_counts=True, axis=0) # check if unique

		# True if NOT repeated #
		valid_actions_within_fleet = counts[inverse_index] == 1 

		return valid_actions_within_fleet

	def check_fleet_actions_with_obstacle(self, test_actions):
		""" Returns array of bools. True if the action leads to a collision """

		return [self.vehicles[k].check_agent_action_with_obstacle(test_actions[k]) for k in range(self.number_of_vehicles)] 

	def move_fleet(self, fleet_actions):

		if self.check_collisions_within:
			# Check if there are collisions between vehicles #
			valid_actions_within_fleet_mask = self.check_collision_within_fleet(fleet_actions)
			# Process the fleet actions and move the vehicles # 
			collisions_dict = {k: self.vehicles[k].move_agent(fleet_actions[k], valid=valid) for k, valid in zip(list(fleet_actions.keys()), valid_actions_within_fleet_mask)}
		else: 
			collisions_dict = {k: self.vehicles[k].move_agent(fleet_actions[k], valid=True) for k in fleet_actions.keys()}

		# Update vector with agents positions #
		self.fleet_positions = np.asarray([veh.actual_agent_position for veh in self.vehicles])
		# Sum up the collisions for termination #
		self.fleet_collisions = np.sum([self.vehicles[k].num_of_collisions for k in range(self.number_of_vehicles)])

		return collisions_dict # return dict with number of collisions of each vehicle 
		
	def get_fleet_distances_traveled(self):

		return [self.vehicles[k].distance_traveled for k in range(self.number_of_vehicles)]

	def get_fleet_positions(self):

		return np.array([veh.actual_agent_position for veh in self.vehicles])

	def get_distances_between_agents(self):

		distances_dict = {}

		# Calculate the euclidean distances between each pair of agents #
		for i in range(self.number_of_vehicles-1):
			for j in range(i + 1, self.number_of_vehicles):
				x1, y1 = self.vehicles[i].actual_agent_position
				x2, y2 = self.vehicles[j].actual_agent_position
				distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
				
				distances_dict[f'Distance_{i}{j}'] = distance
	

		return distances_dict
	
	def reset_fleet(self, initial_positions=None):
		""" Reset the fleet """

		if initial_positions is None:
			initial_positions = self.initial_position_of_each_agent

		# Reset each agent #
		for k in range(self.number_of_vehicles):
			self.vehicles[k].reset_agent(initial_position=initial_positions[k])

		# Assign initial positions to each agent #
		self.fleet_positions = np.asarray([veh.actual_agent_position for veh in self.vehicles])

		# Reset number of collisions #
		self.fleet_collisions = 0

	def get_positions(self): 
		return np.array([veh.actual_agent_position for veh in self.vehicles])

class MultiAgentCleanupEnvironment:

	def __init__(self, 
	      		 scenario_map_name,
				 max_distance_travelled_by_team,
				 max_steps_per_episode,
				 max_trashes_per_agent = None,
				 number_of_agents_by_team = (2, 2),
				 n_actions_by_team = 8,
				 fleet_initial_positions = None,
				 seed = 0,
				 movement_length_by_team = (2,1),
				 vision_length_by_team = (6,6),
				 flag_to_check_collisions_within = False,
				 max_collisions = 5,
				 reward_function = 'basic_reward',
				 dynamic = False, 
				 obstacles = False,
				 reward_weights = (1.0, 0.1),
				 show_plot_graphics = True,
				 ):

		""" The gym environment """

		# Random generators declaration #
		self.seed = seed
		np.random.seed(self.seed)
		self.rng_initial_agents_positions = np.random.default_rng(seed=self.seed)
		self.rng_wind_direction = np.random.default_rng(seed=self.seed)
		self.rng_trash_elements_number = np.random.default_rng(seed=self.seed)
		self.rng_trash_positions_MVN = np.random.default_rng(seed=self.seed)
		self.rng_pollution_spots_number = np.random.default_rng(seed=self.seed)
		self.rng_pollution_spots_locations_indexes = np.random.default_rng(seed=self.seed)
		self.rng_obstacle_number = np.random.default_rng(seed=self.seed)
		self.rng_obstacle_positions = np.random.default_rng(seed=self.seed)
		
		# Load the scenario config and other useful variables #
		self.scenario_map_name = scenario_map_name
		self.scenario_map = np.genfromtxt(f'Environment/Maps/{self.scenario_map_name}.csv', delimiter=',')
		self.scenario_map_backup = self.scenario_map.copy()
		self.number_of_agents_by_team = number_of_agents_by_team
		self.n_agents = np.sum(self.number_of_agents_by_team)
		self.n_teams = len(self.number_of_agents_by_team)
		self.n_actions_by_team = n_actions_by_team
		self.n_actions_of_each_agent = np.repeat(n_actions_by_team, number_of_agents_by_team)
		self.movement_length_by_team = movement_length_by_team
		self.movement_length_of_each_agent = np.repeat(movement_length_by_team, number_of_agents_by_team)
		self.vision_length_by_team = vision_length_by_team
		self.vision_length_of_each_agent = np.repeat(vision_length_by_team, number_of_agents_by_team)
		self.reward_function = reward_function
		self.reward_weights = reward_weights
		self.dynamic = dynamic
		self.obstacles = obstacles
		self.visitable_locations = np.vstack(np.where(self.scenario_map != 0)).T # coords of visitable cells
		self.flag_to_check_collisions_within = flag_to_check_collisions_within

		# Visualization #
		self.activate_plot_graphics = show_plot_graphics
		self.states = None
		self.state_to_render_first_active_agent = None
		self.render_fig = None
		self.colored_agents = True
		
		# Graph for Dijkstra #
		if 'dijkstra' in self.reward_function:
			self.graph = self.grid_to_graph()
			self.dijkstra_distance_map, self.predecessor_map = self.calculate_distance_and_predecessor_maps()

		# Initial positions #
		self.backup_fleet_initial_positions_entry = fleet_initial_positions
		if isinstance(fleet_initial_positions, np.ndarray): # Set initial positions if indicated #
			self.random_inititial_positions = False
			self.initial_positions = fleet_initial_positions
		elif fleet_initial_positions is None: # Random positions all visitable map #
			self.random_inititial_positions = True
			random_positions_indx = self.rng_initial_agents_positions.choice(np.arange(0, len(self.visitable_locations)), self.n_agents, replace=False) # a random index is selected as the maximum number of cells that can be visited
			self.initial_positions = self.visitable_locations[random_positions_indx] 
		elif fleet_initial_positions == 'fixed': # Random choose between 4 fixed deployment positions #
			self.random_inititial_positions = 'fixed'
			self.deployment_positions = np.zeros_like(self.scenario_map)
			if 'ypacarai' in self.scenario_map_name:
				self.deployment_positions[[46,46,49,49], [28,31,28,31]] = 1 # Ypacarai map
			elif 'acoruna_port' in self.scenario_map_name:
				self.deployment_positions[[32,30,28,26], [7,7,7,7]] = 1 # A Coruna port
			elif 'marinapalamos' in self.scenario_map_name:
				self.deployment_positions[[16,14,12,10], [9,9,9,9]] = 1 # marinapalamos
			elif 'challenging_map_big' in self.scenario_map_name:
				self.deployment_positions[[50,52,54,56,58,60], [2,2,2,2,2,2]] = 1 # Challenging map big
			elif 'challenging_map' in self.scenario_map_name:
				self.deployment_positions[[46,48,50,52], [3,3,3,3]] = 1 # Challenging map
				# self.deployment_positions[[44,46,48,50,52,54], [3,3,3,3,3,4]] = 1 # Challenging map
			elif 'comb_port' in self.scenario_map_name:
				self.deployment_positions[[16,14,12,10], [7,7,7,7]] = 1 # comb_port
			self.initial_positions = np.argwhere(self.deployment_positions == 1)[self.rng_initial_agents_positions.choice(len(np.argwhere(self.deployment_positions == 1)), self.n_agents, replace=False)]
		elif fleet_initial_positions == 'area': # Random deployment positions inside an area #
			self.random_inititial_positions = 'area'
			self.deployment_positions = np.zeros_like(self.scenario_map)
			self.deployment_positions[slice(45,50), slice(27,32)] = 1
			self.initial_positions = np.argwhere(self.deployment_positions == 1)[self.rng_initial_agents_positions.choice(len(np.argwhere(self.deployment_positions == 1)), self.n_agents, replace=False)]
		else:
			raise NotImplementedError("Check initial positions!")
		
		# Limits to be declared a death/done agent and initialize done dict #
		self.max_distance_travelled_by_team = max_distance_travelled_by_team
		self.max_distance_travelled_of_each_agent = np.repeat(max_distance_travelled_by_team, number_of_agents_by_team)
		self.max_steps_per_episode = max_steps_per_episode
		self.max_trashes_per_agent = max_trashes_per_agent
		self.steps = 0
		self.max_collisions = max_collisions
		self.done = {i:False for i in range(self.n_agents)} 
		self.dones_by_teams = {teams: False for teams in range(self.n_teams)}  
		self.active_agents = {key: not value for key, value in self.done.items()}
		self.n_active_agents = sum(self.active_agents.values())
		self.percentage_visited = 0.0
		# self.actions = None #(add7channel)
 
		# Load agents identification info #
		self.set_agents_id_info()
	
		# Create the fleets #
		self.fleet = DiscreteFleet(number_of_vehicles = self.n_agents,
							 team_id_of_each_agent = self.team_id_of_each_agent,
							 fleet_initial_positions = self.initial_positions,
							 n_actions_of_each_agent = self.n_actions_of_each_agent,
							 movement_length_of_each_agent = self.movement_length_of_each_agent,
							 vision_length_of_each_agent = self.vision_length_of_each_agent,
							 navigation_map = self.scenario_map,
							 check_collisions_within = self.flag_to_check_collisions_within)
		
		# Randomly generate obstacles if activated #
		self.generate_obstacles()
		self.visited_areas_map = self.scenario_map.copy()
		self.non_water_mask = self.scenario_map != 1 # - self.inside_obstacles_map # mask with True where no water

		# Init the redundancy mask #
		self.redundancy_mask = np.sum([agent.influence_mask for idx, agent in enumerate(self.fleet.vehicles) if self.active_agents[idx]], axis = 0)
		self.visited_areas_map[(self.redundancy_mask.astype(bool) * (1-self.non_water_mask)).astype(bool)] = 0.5 # 0 non visitable, 1 not visited yet, 0.5 visited

		# Idleness map (1-> max idleness, 0-> just visited) #
		self.idleness_map = (1 - self.redundancy_mask.astype(bool))*self.scenario_map

		# Create trash map #
		self.real_trash_map = self.init_real_trash_map()

		# Initialize model trash map #
		if self.number_of_agents_by_team[self.explorers_team_id] > 0:
			self.model_trash_map = np.zeros_like(self.scenario_map) 
			self.previous_model_trash_map = self.model_trash_map.copy()
			self.previousprevious_model_trash_map = self.previous_model_trash_map.copy()
		else: # oracle model
			self.model_trash_map = self.real_trash_map.copy()
			self.previous_model_trash_map = np.zeros_like(self.scenario_map)
			self.previousprevious_model_trash_map = np.zeros_like(self.scenario_map)

		# Info for training among others # 
		if self.n_agents == 1 and not self.dynamic:
			self.observation_space_shape = (3, *self.scenario_map.shape)
		elif self.n_agents > 1 and not self.dynamic:
			self.observation_space_shape = (4, *self.scenario_map.shape)
		elif self.n_agents == 1 and self.dynamic:
			self.observation_space_shape = (5, *self.scenario_map.shape)
		elif self.n_agents > 1 and self.dynamic:
			self.observation_space_shape = (6, *self.scenario_map.shape)
		# self.observation_space_shape = (4, *self.scenario_map.shape) # the channels with model history are removed to test is needed for cooperation (nohistory)
		# self.observation_space_shape = (5, *self.scenario_map.shape) # the channels with other agents positions is removed to test is needed for cooperation (no6channel) (no1channel)
		# self.observation_space_shape = (7, *self.scenario_map.shape) # extra channel to add the previous actions of the agents (add7channel)
		self.angle_set_of_each_agent = {idx: self.fleet.vehicles[idx].angle_set for idx in range(self.n_agents)}

	def set_agents_id_info(self):

		# Save the team id of each agent #
		self.explorers_team_id = 0
		self.cleaners_team_id = 1
		# for team_id, n_actions in enumerate(self.n_actions_by_team):
		# 	if n_actions == 9: 
		# 		self.explorers_team_id = team_id
		# 	elif n_actions == 10: # cleaners have one more action, the clean action
		# 		self.cleaners_team_id = team_id
			
		# Differentiate between agents by their teams #
		self.team_id_of_each_agent = np.repeat(np.arange(self.n_teams), self.number_of_agents_by_team)
		self.team_id_normalized_of_each_agent = (self.team_id_of_each_agent+1)/self.n_teams # decimal id between 0 and 1
		self.teams_ids = np.repeat(np.arange(self.n_teams), 1)
		self.masks_by_team = [self.team_id_of_each_agent == team for team in self.teams_ids]

		# Colors for visualization #
		if self.colored_agents:
			self.colors_agents = ['black', 'gainsboro']
			palettes_by_team = {0: ['green', 'mediumseagreen', 'seagreen', 'olive'], 1: ['darkred', 'indianred', 'tomato'], 2: ['sandybrown', 'peachpuff'], 3: ['darkmagenta']}
			for agent in range(self.n_agents):
				self.colors_agents.extend([palettes_by_team[self.team_id_of_each_agent[agent]].pop(0)])
			self.agents_colormap = matplotlib.colors.ListedColormap(self.colors_agents)
			self.n_colors_agents_render = len(self.colors_agents)

	def generate_obstacles(self):
		""" Generate randomly the obstacles map. """

		self.scenario_map = self.scenario_map_backup.copy()

		if self.obstacles:
			# Generate a random inside obstacles map #
			self.inside_obstacles_map = np.zeros_like(self.scenario_map)
			if 'big' in self.scenario_map_name:
				obstacles_pos_indx = self.rng_obstacle_positions.choice(np.arange(0, len(self.visitable_locations)), size=self.rng_obstacle_number.integers(30, 40), replace=False)
			else:
				obstacles_pos_indx = self.rng_obstacle_positions.choice(np.arange(0, len(self.visitable_locations)), size=self.rng_obstacle_number.integers(15, 20), replace=False)
			# Exclude the initial positions of the agents #
			initial_positions_indx = np.array([np.where(np.all(self.visitable_locations == pos, axis=1))[0][0] for pos in self.initial_positions])
			obstacles_pos_indx = np.delete(obstacles_pos_indx, np.where(np.isin(obstacles_pos_indx, initial_positions_indx))[0])
			self.inside_obstacles_map[self.visitable_locations[obstacles_pos_indx, 0], self.visitable_locations[obstacles_pos_indx, 1]] = 1.0
			
			# Update the scenario map and visitable locations #
			self.scenario_map = self.scenario_map - self.inside_obstacles_map
			self.visitable_locations = np.vstack(np.where(self.scenario_map != 0)).T # coords of visitable cells
			
			# Update the obstacle map for every agent #
			for i in range(self.n_agents):
				self.fleet.vehicles[i].navigation_map = self.scenario_map
		else:
			self.inside_obstacles_map = np.zeros_like(self.scenario_map)

	def reset_env(self):
		""" Reset the environment """

		# Reset the scenario map and visitable locations #
		self.scenario_map = self.scenario_map_backup.copy()
		self.visitable_locations = np.vstack(np.where(self.scenario_map != 0)).T # coords of visitable cells

		# Get the N random initial positions #
		if self.random_inititial_positions == 'area' or self.random_inititial_positions == 'fixed':
			self.initial_positions = np.argwhere(self.deployment_positions == 1)[self.rng_initial_agents_positions.choice(len(np.argwhere(self.deployment_positions == 1)), self.n_agents, replace=False)]
		elif self.random_inititial_positions is True:
			random_positions_indx = self.rng_initial_agents_positions.choice(np.arange(0, len(self.visitable_locations)), self.n_agents, replace=False)
			self.initial_positions = self.visitable_locations[random_positions_indx]

		# Randomly generate obstacles if activated #
		self.generate_obstacles()
		self.non_water_mask = self.scenario_map != 1 # - self.inside_obstacles_map # mask with True where no water
		self.visited_areas_map = self.scenario_map.copy()


		# Reset the information of the fleet #
		self.steps = 0
		self.fleet.reset_fleet(initial_positions=self.initial_positions)
		self.done = {agent_id: False for agent_id in range(self.n_agents)}
		self.dones_by_teams = {team: False for team in range(self.n_teams)}  
		self.active_agents = {agent_id: True for agent_id in range(self.n_agents)}
		self.n_active_agents = sum(self.active_agents.values())
		self.percentage_visited = 0.0
		# self.actions = None #(add7channel)

		# Compute the redundancy mask after reset #
		self.redundancy_mask = np.sum([agent.influence_mask for idx, agent in enumerate(self.fleet.vehicles) if self.active_agents[idx]], axis = 0)
		self.visited_areas_map[(self.redundancy_mask.astype(bool) * (1-self.non_water_mask)).astype(bool)] = 0.5 # 0 non visitable, 1 not visited yet, 0.5 visited

		# Idleness map (1-> max idleness, 0-> just visited) #
		self.idleness_map = (1 - self.redundancy_mask.astype(bool))*self.scenario_map
		
		# Reset the trash map #
		self.real_trash_map = self.init_real_trash_map()
		
		# Create an empty model after reset #
		if self.number_of_agents_by_team[self.explorers_team_id] > 0:
			self.model_trash_map = np.zeros_like(self.scenario_map) 
			self.previous_model_trash_map = self.model_trash_map.copy()
			self.previousprevious_model_trash_map = self.previous_model_trash_map.copy()
		else: # oracle model
			self.model_trash_map = self.real_trash_map.copy()
			self.previous_model_trash_map = np.zeros_like(self.scenario_map)
			self.previousprevious_model_trash_map = np.zeros_like(self.scenario_map)

		# Detect trash from cameras and update model #
		self.update_model_trash_map()

		# Update the states of the agents #
		self.capture_states()
		
		# Reset visualization #
		if self.render_fig is not None and self.activate_plot_graphics:
			plt.close(self.render_fig)
			self.render_fig = None
		
		return self.states
	
	def generate_trash_positions(self, max_number_of_trash_elements_per_spot = 70, max_number_of_pollution_spots = 4):
		""" Generate the positions of the trash elements with a MVN distribution. """

		# Random position of pollution spots inside of the navigable map #
		# pollution_spots_number = self.rng_pollution_spots_number.integers(1, max_number_of_pollution_spots+1)
		if 'big' in self.scenario_map_name:
			max_number_of_trash_elements_per_spot = 100
			min_number_of_trash_elements_per_spot = 60
			scale = 14
		else:
			max_number_of_trash_elements_per_spot = 75
			min_number_of_trash_elements_per_spot = 40
			scale = 10
		pollution_spots_number = 1
		pollution_spots_locations_indexes = self.rng_pollution_spots_locations_indexes.choice(np.arange(0, len(self.visitable_locations)), pollution_spots_number, replace=False)
		number_of_trash_elements_in_each_spot = self.rng_trash_elements_number.normal(loc=max_number_of_trash_elements_per_spot, scale=scale, size=pollution_spots_number).round().astype(int)
		number_of_trash_elements_in_each_spot[number_of_trash_elements_in_each_spot <= 0] = min_number_of_trash_elements_per_spot
		
		# Generate the trash positions #
		trash_positions_yx = np.array([])
		for j, index in enumerate(pollution_spots_locations_indexes):
			trash_positions_yx_to_add = self.rng_trash_positions_MVN.multivariate_normal(mean=self.visitable_locations[index], cov=[[20,0],[0,20]], size = number_of_trash_elements_in_each_spot[j])
			trash_positions_yx = np.vstack((trash_positions_yx, trash_positions_yx_to_add)) if trash_positions_yx.size else trash_positions_yx_to_add

		self.initial_number_of_trash_elements = len(trash_positions_yx)

		return trash_positions_yx

	def init_real_trash_map(self):
		""" Initialize the trash map with a number of trash elements. """

		# Reset the trash counter #
		self.history_trashes_removed_per_agent = {idx: 0 for idx in self.get_active_cleaners_positions().keys()}
		self.trashes_removed_per_agent = {}

		# Establish the wind direction during the episode #
		self.wind_direction = np.array([self.rng_wind_direction.uniform(-0.10, 0.10), self.rng_wind_direction.uniform(-0.10, 0.10)])
		
		# Generate the trash positions #
		self.trash_positions_yx = self.generate_trash_positions()

		# Save the vehicle that first sees each trash element with the step, and the one that collects it with the step #
		self.trash_tracking = True
		if self.trash_tracking:
			self.trash_remaining_info = np.full(self.trash_positions_yx.shape[0], -1, dtype=[('step_discover', 'i4'), ('vehicle_discover', 'i4')])
			self.trash_removed_info = np.zeros(0, dtype=[('step_discover', 'i4'), ('vehicle_discover', 'i4'),
																	   ('step_remove', 'i4'), ('vehicle_remove', 'i4')])

		# Saturate trash positions #
		self.saturate_trash_positions()

		# Discretize the trash map #
		real_trash_map = self.get_discretized_real_trash_map()

		return real_trash_map

	def update_real_trash_map(self, actions):
		"""" Update the position of each element of trash. Trash move with a component of wind and a random component. """

		# CLEAN PROCESS: Erase one trash if the cleaner agents go through a trash position, or remove a percentage of trash if the cleaner take the action to clean (action 9) #
		cleaners_actions = {idx: action for idx, action in actions.items() if self.team_id_of_each_agent[idx] == self.cleaners_team_id}
		cleaners_positions = self.get_active_cleaners_positions()
		rounded_trash_positions = self.trash_positions_yx.round()
		
		if self.n_actions_by_team[self.cleaners_team_id] == 10:
			# There is a cleaning action: #
			def get_indexes_to_clean_with_ramdomness(indexes, mean = 0.8, std = 0.1):
				return np.random.choice(indexes, round(len(indexes)*np.clip(0,1,np.random.normal(loc=mean,scale=std))), replace=False)
			self.trashes_removed_per_agent = {idx: np.array([np.random.choice(indexes)]) if cleaners_actions[idx] !=9 else get_indexes_to_clean_with_ramdomness(indexes) for idx, position in cleaners_positions.items() if len(indexes := np.where((rounded_trash_positions == position).all(1))[0]) > 0}
		else:
			# Trash is cleaned just with going through: #
			self.trashes_removed_per_agent = {idx: indexes for idx, position in cleaners_positions.items() if len(indexes := np.where((rounded_trash_positions == position).all(1))[0]) > 0}
			if self.max_trashes_per_agent is not None:
				# Limit the number of trashes removed per agent #
				self.trashes_removed_per_agent = {idx: indexes[:(self.max_trashes_per_agent-self.history_trashes_removed_per_agent[idx])] for idx, indexes in self.trashes_removed_per_agent.items() if len(indexes) > 0}
		
		if self.trashes_removed_per_agent:
			indexes_to_remove = np.concatenate([*self.trashes_removed_per_agent.values()])

			if self.trash_tracking:
				info_to_remove = np.zeros(len(indexes_to_remove), dtype=self.trash_removed_info.dtype)
				info_to_remove['step_discover'] = self.trash_remaining_info['step_discover'][indexes_to_remove]
				info_to_remove['vehicle_discover'] = self.trash_remaining_info['vehicle_discover'][indexes_to_remove]
				info_to_remove['step_remove'] = self.steps
				info_to_remove['vehicle_remove'] = np.concatenate([[idx]*len(self.trashes_removed_per_agent[idx]) for idx in self.trashes_removed_per_agent.keys()])
				self.trash_removed_info = np.append(self.trash_removed_info, info_to_remove)
				self.trash_remaining_info = np.delete(self.trash_remaining_info, indexes_to_remove)

			self.trash_positions_yx = np.delete(self.trash_positions_yx, indexes_to_remove, axis = 0)
		self.previous_trashes_removed_per_agent = self.trashes_removed_per_agent.copy()
		self.history_trashes_removed_per_agent = {idx: self.history_trashes_removed_per_agent[idx] + len(self.trashes_removed_per_agent[idx]) if idx in self.trashes_removed_per_agent else self.history_trashes_removed_per_agent[idx] for idx in self.history_trashes_removed_per_agent.keys()}
		
		# Movement of trash if dynamic #
		if self.dynamic:
			# Movement of trash: a random component and a wind component #
			random_component = np.random.uniform(-0.09, 0.09, self.trash_positions_yx.shape)
			self.trash_positions_yx += random_component + self.wind_direction

			# Saturate trash positions #
			self.saturate_trash_positions()

		# Discretize the updated real trash map after dynamism or cleaning #
		self.real_trash_map = self.get_discretized_real_trash_map()

	def saturate_trash_positions(self):
		""" Saturate trash positions outside of the navigable map to the closest navigable position. """

		# Saturate inside the scenario shape #
		self.trash_positions_yx = np.clip(self.trash_positions_yx, [0,0], np.array(self.scenario_map.shape)-1)

		# Saturate inside the navigable area #
		mask_of_which_need_to_saturate = self.scenario_map[np.round(self.trash_positions_yx)[:,0].astype(int), np.round(self.trash_positions_yx)[:,1].astype(int)] == 0
		poses_of_trashes_to_saturate = self.trash_positions_yx[mask_of_which_need_to_saturate]
		closest_visitable_locations = self.visitable_locations[np.argmin(np.linalg.norm(self.visitable_locations[:,np.newaxis] - poses_of_trashes_to_saturate, axis = 2), axis = 0)]
		vectors_from_pixel_centers_to_trashes = poses_of_trashes_to_saturate-closest_visitable_locations
		saturation_distances_from_pixel_centers = np.sign(np.min(vectors_from_pixel_centers_to_trashes, axis = 1))*0.49*np.min(np.abs(vectors_from_pixel_centers_to_trashes), axis = 1) / np.max(np.abs(vectors_from_pixel_centers_to_trashes), axis = 1) # solving similar triangles
		axes_saturated_to_0_49 = np.argmax(np.abs(vectors_from_pixel_centers_to_trashes), axis = 1)
		additions = np.zeros_like(poses_of_trashes_to_saturate)
		
		# Saturation of the component that cut the grid #
		additions[np.arange(len(poses_of_trashes_to_saturate)), axes_saturated_to_0_49] = [0.49 if distance > 0 else -0.49 for distance in vectors_from_pixel_centers_to_trashes[np.arange(len(poses_of_trashes_to_saturate)), axes_saturated_to_0_49]]

		# Saturation of the other component #
		additions[np.arange(len(poses_of_trashes_to_saturate)), 1-axes_saturated_to_0_49] = saturation_distances_from_pixel_centers

		# Apply the saturation #
		self.trash_positions_yx[mask_of_which_need_to_saturate] = closest_visitable_locations + additions

	def update_model_trash_map(self):
		""" The active agents capture new trash information from the environment with cameras. The length of vision depends on its area of influence. """
		
		self.previousprevious_model_trash_map = self.previous_model_trash_map.copy()
		self.previous_model_trash_map = self.model_trash_map.copy()
		if self.number_of_agents_by_team[self.explorers_team_id] > 0:
			self.model_trash_map[self.redundancy_mask.astype(bool)] = self.real_trash_map[self.redundancy_mask.astype(bool)]
		else: # oracle model
			self.model_trash_map = self.real_trash_map.copy()

		if self.trash_tracking:
			# Update the trash info with the new discovered trash #
			rounded_trash_positions = self.trash_positions_yx.round().astype(int)
			pair_newtrash_vehicle = {trash_id: agent_id for agent_id, agent in enumerate(self.fleet.vehicles) if self.active_agents[agent_id] for trash_id, trash_pos in enumerate(rounded_trash_positions) if agent.influence_mask[tuple(trash_pos)]}

			for trash_id, agent_id in pair_newtrash_vehicle.items():
				self.trash_remaining_info['step_discover'][trash_id] = self.steps
				self.trash_remaining_info['vehicle_discover'][trash_id] = agent_id

	def update_idleness_map(self, forget_factor = 0.9, minimum_idleness = 0):

		self.idleness_map = self.idleness_map + 1.0 / (forget_factor * self.max_steps_per_episode)
		self.idleness_map = self.idleness_map - self.redundancy_mask.astype(bool)
		self.idleness_map = np.clip(self.idleness_map * self.scenario_map, minimum_idleness, 1)

	def dijkstra(self, start):
		# Initialize distances and priority queue
		distances = {vertex: float('infinity') for vertex in self.graph}
		distances[start] = 0
		priority_queue = [(0, start)]
		predecessors = {vertex: None for vertex in self.graph}
		
		while priority_queue:
			current_distance, current_vertex = heapq.heappop(priority_queue)

			if current_distance > distances[current_vertex]:
				continue

			for neighbor, weight in self.graph[current_vertex].items():
				distance = current_distance + weight

				if distance < distances[neighbor]:
					distances[neighbor] = distance
					predecessors[neighbor] = current_vertex
					heapq.heappush(priority_queue, (distance, neighbor))
					
		return distances, predecessors
	
	def grid_to_graph(self, directions=None):
		rows = self.scenario_map.shape[0]
		cols = self.scenario_map.shape[1]
		graph = {}

		# Directions for 8 adjacent cells (including diagonals)
		if directions is None:
			directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

		for x in range(rows):
			for y in range(cols):
				if self.scenario_map[x,y] == 1:  # Assuming 1 represents a navigable cell
					graph[(x, y)] = {}
					for dx, dy in directions:
						nx, ny = x + dx, y + dy
						if 0 <= nx < rows and 0 <= ny < cols and self.scenario_map[nx,ny] == 1:
							if not is_reachable(self.scenario_map, (x, y), (nx, ny)):
								continue
							graph[(x, y)][(nx, ny)] = np.linalg.norm(np.array([x,y]) - np.array([nx,ny]))  # Assuming all edges have a weight of 1

		return graph

	def calculate_distance_and_predecessor_maps(self):
		""" Calculate the distance and predecessor maps for each pixel """
		distance_map = {}
		predecessor_map = {}
		
		for pos in list(self.graph.keys()):
			i,j = pos
			distances, predecessors = self.dijkstra((i, j))
			distance_map[(i, j)] = distances
			predecessor_map[(i, j)] = predecessors
				
		return distance_map, predecessor_map
	
	def step(self, actions: dict, dont_calculate_rewards = False):
		"""Execute all updates for each step"""

		# Update the steps #
		self.steps += 1
		# self.actions = actions # (add7channel)

		# Process movement actions. There are actions only for active agents #
		self.collisions_mask_dict = self.fleet.move_fleet(actions)

		# Update movement of trash map if dynamic and execute cleaning process #
		self.update_real_trash_map(actions)

		if self.fleet.fleet_collisions > 0 and any(self.collisions_mask_dict.values()) and self.steps == self.max_steps_per_episode:
			print("Nº collision:" + str(self.fleet.fleet_collisions))
		
		# Update the redundancy mask after movements #
		self.redundancy_mask = np.sum([agent.influence_mask for idx, agent in enumerate(self.fleet.vehicles) if self.active_agents[idx]], axis = 0)


		# Update visited map and new discovered areas divided by overlapping agents #
		self.new_discovered_area_per_agent = {idx: ((self.visited_areas_map[agent.influence_mask.astype(bool)] == 1).astype(int) / self.redundancy_mask[agent.influence_mask.astype(bool)] ).sum() 
										if self.active_agents[idx] and agent.team_id == self.explorers_team_id else 0 for idx, agent in enumerate(self.fleet.vehicles)}
		self.visited_areas_map[(self.redundancy_mask.astype(bool) * (1-self.non_water_mask)).astype(bool)] = 0.5 # 0 non visitable, 1 not visited yet, 0.5 visited
		self.percentage_visited = (self.visited_areas_map == 0.5).sum() / (self.visited_areas_map != 0).sum()
		
		# Update idleness map #
		self.idleness_discounted_per_agent = {idx: (self.idleness_map[agent.influence_mask.astype(bool)] / self.redundancy_mask[agent.influence_mask.astype(bool)] ).sum() 
										 if self.active_agents[idx] and agent.team_id == self.explorers_team_id else 0 for idx, agent in enumerate(self.fleet.vehicles)}
		self.update_idleness_map()
		
		# ponderation_r = np.array(
		# 		[np.sum(
		# 			ponderation_map[agent.influence_mask.astype(bool)] / self.redundancy_mask[agent.influence_mask.astype(bool)]
		# 			) if self.active_agents[idx] else 0 for idx, agent in enumerate(self.fleet.vehicles)
		# 			]
		# 		)

		# Detect trash from cameras and update model #
		self.update_model_trash_map()

		# Compute reward #
		rewards = self.get_reward(actions, dont_calculate_rewards)

		# Update the states of the agents #
		self.capture_states()

		# Plot graphics if activated #
		if self.activate_plot_graphics:
			self.render()

		# FINAL CONDITIONS #
		# By distance: 
		# self.done = {agent_id: (self.fleet.get_fleet_distances_traveled()[agent_id] > self.max_distance_travelled_of_each_agent[agent_id] or self.fleet.fleet_collisions > self.max_collisions) for agent_id in range(self.n_agents)}
		
		# By steps: 
		if self.steps >= self.max_steps_per_episode or len(self.trash_positions_yx) == 0:
			self.done = {agent_id: True for agent_id in range(self.n_agents)}
		
		self.dones_by_teams = {team: all([is_done for agent_id, is_done in self.done.items() if self.team_id_of_each_agent[agent_id] == team]) for team in self.teams_ids}  
		# If cleaners team is done, the episode is done #
		if self.dones_by_teams[self.cleaners_team_id] and self.number_of_agents_by_team[self.cleaners_team_id] > 0:
			self.done = {agent_id: True for agent_id in range(self.n_agents)}
			self.dones_by_teams = {team: True for team in self.teams_ids}
		self.active_agents = {key: not value for key, value in self.done.items()}
		self.n_active_agents = sum(self.active_agents.values())

		return self.states, rewards, self.done

	def capture_states(self):
		""" Update the states for every vehicle. Every channel will be an input of the Neural Network. """

		states = {}
		# Channel 0 -> Known boundaries/map
		if self.obstacles:
			self.obstacle_map = self.scenario_map - self.inside_obstacles_map
		else:
			self.obstacle_map = self.scenario_map

		# Create fleet position map #
		fleet_position_map_denoted_by_its_team_id = np.zeros_like(self.scenario_map)
		for agent_id, pose in self.get_active_agents_positions_dict().items():
			fleet_position_map_denoted_by_its_team_id[pose[0], pose[1]] = self.team_id_normalized_of_each_agent[agent_id] # set team id normalized in the position of each agent

		if self.colored_agents == True and self.activate_plot_graphics:
			fleet_position_map_colored = np.zeros_like(self.scenario_map)
			for agent_id, pose in self.get_active_agents_positions_dict().items():
					fleet_position_map_colored[pose[0], pose[1]] = (1/self.n_colors_agents_render)*(agent_id+2) + 0.01

		first_available_agent = np.argmax(list(self.active_agents.values())) # first True in active_agents
		for agent_id, active in self.active_agents.items():
			if active:
				observing_agent_position_with_trail = np.zeros_like(self.scenario_map)
				waypoints = self.fleet.vehicles[agent_id].waypoints
				trail = True # estela
				if trail: 
					if len(waypoints) > 10:
						trail_points = 10
					else:
						trail_points = len(waypoints)
				else:
					trail_points = 1
				trail_values = np.linspace(0, 1, trail_points+1)[1:]
				y_trail, x_trail =  zip(*waypoints[-trail_points:])
				observing_agent_position_with_trail[y_trail, x_trail] = trail_values

				agent_observation_of_fleet = fleet_position_map_denoted_by_its_team_id.copy()
				agents_to_remove_positions = np.array([pos for idx, pos in enumerate(self.fleet.fleet_positions) if (idx == agent_id) or (not self.active_agents[idx])])  # if its the observing agent or not active, save its position to remove
				agent_observation_of_fleet[agents_to_remove_positions[:,0], agents_to_remove_positions[:,1]] = 0.0
				# Map with the position of the agents and the selected action (add7channel)
				# agents_previous_actions_map = np.zeros_like(self.scenario_map)
				# if self.actions is not None:
				# 	self.previous_actions = np.array([self.actions[idx] if self.active_agents[idx] else -1 for idx in range(self.n_agents)])
				# 	agents_previous_actions_map[self.fleet.fleet_positions[:,0], self.fleet.fleet_positions[:,1]] = (self.previous_actions + 1)/self.n_actions_of_each_agent # +1 to avoid 0 that is empty area

				"""Each key from states dictionary is an agent, all states associated to that agent are concatenated in its value:"""
				if self.n_agents == 1 and not self.dynamic: # 3 channels
					states[agent_id] = np.concatenate(( 
						# self.visited_areas_map[np.newaxis], # Channel 0 -> Map with visited positions. 0 non visitable, 1 non visited, 0.5 visited.
						self.idleness_map[np.newaxis], # Channel 0 -> Map of idleness (1-> max idleness, 0-> just visited)
						(self.model_trash_map/(np.max(self.model_trash_map)+1E-5))[np.newaxis], # Channel 1 -> Trash model map (normalized)
						observing_agent_position_with_trail[np.newaxis], # Channel 3 -> Observing agent position map with a trail
					), dtype=np.float16)
				elif self.n_agents > 1 and not self.dynamic: # 4 channels
					states[agent_id] = np.concatenate(( 
						# self.visited_areas_map[np.newaxis], # Channel 0 -> Map with visited positions. 0 non visitable, 1 non visited, 0.5 visited.
						self.idleness_map[np.newaxis], # Channel 0 -> Map of idleness (1-> max idleness, 0-> just visited)
						(self.model_trash_map/(np.max(self.model_trash_map)+1E-5))[np.newaxis], # Channel 1 -> Trash model map (normalized)
						observing_agent_position_with_trail[np.newaxis], # Channel 2 -> Observing agent position map with a trail
						agent_observation_of_fleet[np.newaxis], # Channel 3 -> Others active agents position map
					), dtype=np.float16)
				elif self.n_agents == 1 and self.dynamic: # 5 channels
					states[agent_id] = np.concatenate(( 
						# self.visited_areas_map[np.newaxis], # Channel 0 -> Map with visited positions. 0 non visitable, 1 non visited, 0.5 visited.
						self.idleness_map[np.newaxis], # Channel 0 -> Map of idleness (1-> max idleness, 0-> just visited)
						(self.model_trash_map/(np.max(self.model_trash_map)+1E-5))[np.newaxis], # Channel 1 -> Trash model map (normalized)
						(self.previous_model_trash_map/np.max(self.previous_model_trash_map+1E-5))[np.newaxis], # Channel 2 -> Previous trash model map (normalized)
						(self.previousprevious_model_trash_map/np.max(self.previousprevious_model_trash_map+1E-5))[np.newaxis], # Channel 3 -> Previous previous trash model map (normalized)
						observing_agent_position_with_trail[np.newaxis], # Channel 4 -> Observing agent position map with a trail
					), dtype=np.float16)
				elif self.n_agents > 1 and self.dynamic: # 6 channels
					states[agent_id] = np.concatenate(( 
						# obstacle_map[np.newaxis], # Channel 0 -> Known boundaries/navigation map
						# self.visited_areas_map[np.newaxis], # Channel 0 -> Map with visited positions. 0 non visitable, 1 non visited, 0.5 visited. (no1channel)
						self.idleness_map[np.newaxis], # Channel 0 -> Map of idleness (1-> max idleness, 0-> just visited)
						(self.model_trash_map/(np.max(self.model_trash_map)+1E-5))[np.newaxis], # Channel 1 -> Trash model map (normalized)
						(self.previous_model_trash_map/np.max(self.previous_model_trash_map+1E-5))[np.newaxis], # Channel 2 -> Previous trash model map (normalized) (nohistory)
						(self.previousprevious_model_trash_map/np.max(self.previousprevious_model_trash_map+1E-5))[np.newaxis], # Channel 3 -> Previous previous trash model map (normalized) (nohistory)
						observing_agent_position_with_trail[np.newaxis], # Channel 4 -> Observing agent position map with a trail
						agent_observation_of_fleet[np.newaxis], # Channel 5 -> Others active agents position map (no6channel)
						# agents_previous_actions_map[np.newaxis], # Channel 6 -> Previous actions of the agents (add7channel)
					), dtype=np.float16)


				if agent_id == first_available_agent and self.activate_plot_graphics:
					gaussian_blurred_model_trash = gaussian_filter(self.model_trash_map, sigma=20, mode='constant', cval=0)
					gaussian_blurred_model_trash = (1-self.non_water_mask) * gaussian_blurred_model_trash/(np.max(gaussian_blurred_model_trash)+1E-5)
					if self.colored_agents == True:
						self.state_to_render_first_active_agent = np.concatenate((
							# self.visited_areas_map[np.newaxis], # AXIS 0
							self.idleness_map[np.newaxis], # AXIS 0
							self.real_trash_map[np.newaxis], # AXIS 1
							self.model_trash_map[np.newaxis], # AXIS 2
							fleet_position_map_colored[np.newaxis], # AXIS 3
							self.redundancy_mask[np.newaxis], # AXIS 4
							# gaussian_blurred_model_trash[np.newaxis], # AXIS 4
						))
					
					else:
						self.state_to_render_first_active_agent = np.concatenate(( 
							# self.visited_areas_map[np.newaxis], # AXIS 0
							self.idleness_map[np.newaxis], # AXIS 0
							self.real_trash_map[np.newaxis], # AXIS 1
							self.model_trash_map[np.newaxis], # AXIS 2
							observing_agent_position_with_trail[np.newaxis], # AXIS 3
							agent_observation_of_fleet[np.newaxis],	# AXIS 4
							self.redundancy_mask[np.newaxis], # AXIS 5
						))

		self.states = {agent_id: states[agent_id] for agent_id in range(self.n_agents) if self.active_agents[agent_id]}

	def render(self):
		""" Print visual representation of each state of the scenario. """

		if not any(self.active_agents.values()):
			return
		
		if self.render_fig is None: # create first frame of fig, if not already created

			if self.colored_agents == True:
				self.render_fig, self.axs = plt.subplots(1, 5, figsize=(17,5))
			else:
				self.render_fig, self.axs = plt.subplots(1, 6, figsize=(17,5))
			
			# AXIS 0: Plot the navigation map with trash elements #
			self.im0 = self.axs[0].imshow(self.state_to_render_first_active_agent[0], cmap = 'cet_linear_bgy_10_95_c74')
			self.im0_scatter = self.axs[0].scatter(self.trash_positions_yx[:,1], self.trash_positions_yx[:,0], c = 'red', s = 1)
			self.axs[0].set_title('Navigation map')

			# AXIS 1: Plot the discretized real trash map #
			self.state_to_render_first_active_agent[1][self.non_water_mask] = np.nan
			self.im1 = self.axs[1].imshow(self.state_to_render_first_active_agent[1], cmap ='cet_linear_bgyw_20_98_c66')
			self.axs[1].set_title("Real trash map")

			# AXIS 2: Plot the discretized model trash map #
			self.state_to_render_first_active_agent[2][self.non_water_mask] = np.nan
			self.im2 = self.axs[2].imshow(self.state_to_render_first_active_agent[2], cmap ='cet_linear_bgyw_20_98_c66')
			self.axs[2].set_title("Trash detected (model)")

			if self.colored_agents == True:
				# AXIS 3: Active colored agents positions #
				self.state_to_render_first_active_agent[3][self.non_water_mask] = 1/self.n_colors_agents_render + 0.01
				self.im3 = self.axs[3].imshow(self.state_to_render_first_active_agent[3], cmap = self.agents_colormap, vmin = 0.0, vmax = 1.0)
				self.im3_scatter = self.axs[3].scatter(self.trash_positions_yx[:,1], self.trash_positions_yx[:,0], c = 'white', s = 1)
				self.axs[3].set_title("Agents position")

				# AXIS 4: Redundancy mask #
				self.state_to_render_first_active_agent[4][self.non_water_mask] = np.nan
				# self.im4 = self.axs[4].imshow(self.state_to_render_first_active_agent[4], cmap = 'cet_linear_bgy_10_95_c74', vmin = 0.0, vmax = 4.0)
				self.im4 = self.axs[4].imshow(self.state_to_render_first_active_agent[4], cmap = 'cet_linear_bgy_10_95_c74', vmin = 0.0, vmax = 1.0)
				self.axs[4].set_title("Redundancy mask")
			else:
				# AXIS 3: Agent 0 position #
				self.state_to_render_first_active_agent[3][self.non_water_mask] = 0.75
				self.im3 = self.axs[3].imshow(self.state_to_render_first_active_agent[3], cmap = 'gray', vmin = 0.0, vmax = 1.0)
				self.axs[3].set_title("Observer agent position")

				# AXIS 4: Others-than-Agent 0 positions #
				self.state_to_render_first_active_agent[4][self.non_water_mask] = 0.75
				self.im4 = self.axs[4].imshow(self.state_to_render_first_active_agent[4], cmap = 'gray', vmin = 0.0, vmax = 1.0)
				self.axs[4].set_title("Others agents position")

				# AXIS 5: Redundancy mask #
				self.state_to_render_first_active_agent[5][self.non_water_mask] = np.nan
				self.im5 = self.axs[5].imshow(self.state_to_render_first_active_agent[5], cmap = 'cet_linear_bgy_10_95_c74', vmin = 0.0, vmax = 4.0)
				self.axs[5].set_title("Vision mask")

		else:
			# UPDATE FIG INFO/DATA IN EVERY RENDER CALL #
			# AXIS 0: Plot the navigation map with trash elements #
			self.im0.set_data(self.state_to_render_first_active_agent[0])
			self.im0_scatter.set_offsets(self.trash_positions_yx[:, [1, 0]]) # update scatter plot of trash
			# AXIS 1: Plot the trash map #
			self.im1.set_clim(0, np.max(self.state_to_render_first_active_agent[1]))
			self.state_to_render_first_active_agent[1][self.non_water_mask] = np.nan
			self.im1.set_data(self.state_to_render_first_active_agent[1])
			# AXIS 2: Plot trash detected #
			self.im2.set_clim(0, np.max(self.state_to_render_first_active_agent[2]))
			self.state_to_render_first_active_agent[2][self.non_water_mask] = np.nan
			self.im2.set_data(self.state_to_render_first_active_agent[2])
			if self.colored_agents == True:
				# AXIS 3: Active colored agents positions #
				self.state_to_render_first_active_agent[3][self.non_water_mask] = 1/self.n_colors_agents_render + 0.01
				self.im3.set_data(self.state_to_render_first_active_agent[3])
				self.im3_scatter.set_offsets(self.trash_positions_yx[:, [1, 0]]) # update scatter plot of trash

				# AXIS 4: Redundancy mask #
				self.state_to_render_first_active_agent[4][self.non_water_mask] = np.nan
				self.im4.set_data(self.state_to_render_first_active_agent[4])
			else:
				# AXIS 3: Agent 0 position #
				self.state_to_render_first_active_agent[3][self.non_water_mask] = 0.75
				self.im3.set_data(self.state_to_render_first_active_agent[3])
				# AXIS 4: Others-than-Agent 0 positions #
				self.state_to_render_first_active_agent[4][self.non_water_mask] = 0.75
				self.im4.set_data(self.state_to_render_first_active_agent[4])
				# AXIS 5: Redundancy mask #
				self.state_to_render_first_active_agent[5][self.non_water_mask] = np.nan
				self.im5.set_data(self.state_to_render_first_active_agent[5])

		plt.draw()	
		plt.pause(0.01)

	def get_discretized_real_trash_map(self):
		""" Returns the discretized trash map """
		
		real_trash_map = np.zeros_like(self.scenario_map)
		np.add.at(real_trash_map, (np.round(self.trash_positions_yx)[:,0].astype(int), np.round(self.trash_positions_yx)[:,1].astype(int)), 1)
		
		return real_trash_map

	def get_reward(self, actions, dont_calculate_rewards = False):
		""" Reward functions. Different reward functions depending on the team of the agent. """
		if dont_calculate_rewards:
			return {agent_id: 0 for agent_id in range(self.n_agents)}
		if 'negativedistance' in self.reward_function or 'negativedijkstra' in self.reward_function or 'negativelogdijkstra' in self.reward_function or 'negativeastar' in self.reward_function:
			
			# EXPLORERS TEAM #
			explorers_alive = [idx for idx, agent_team in enumerate(self.team_id_of_each_agent) if agent_team == self.explorers_team_id and self.active_agents[idx]]
			changes_in_whole_model = np.abs(self.model_trash_map - self.previous_model_trash_map)
			r_for_discover_trash = np.array(
				[np.sum(
					changes_in_whole_model[agent.influence_mask.astype(bool)] / self.redundancy_mask[agent.influence_mask.astype(bool)]
					) if idx in explorers_alive else 0 for idx, agent in enumerate(self.fleet.vehicles) # only explorers will get reward for finding trash
				])
			
			# r_for_discover_new_area = np.array([*self.new_discovered_area_per_agent.values()])
			r_for_discover_new_area = np.array([*self.idleness_discounted_per_agent.values()])
			
			# CLEANERS TEAM #
			cleaners_alive = [idx for idx, agent_team in enumerate(self.team_id_of_each_agent) if agent_team == self.cleaners_team_id and self.active_agents[idx]]
			r_for_cleaned_trash = np.array([len(self.trashes_removed_per_agent[idx]) if idx in cleaners_alive and idx in self.trashes_removed_per_agent else 0 for idx in range(self.n_agents)])

			if 'time' in self.reward_function:
				r_time_penalization = np.array([-1 if idx in cleaners_alive else 0 for idx in range(self.n_agents)])
			else:
				r_time_penalization = np.zeros(self.n_agents)
			
			# BOTH TEAMS #
			# If there is known trash, reward trough negative distance to closer trash #
			if np.any(self.model_trash_map):
				# Negative distance to closest trash in each step. Continuous penalization, lower when closer to trash #
				r_negative_distance_to_trash = np.array([-self.get_distance_to_closest_known_trash(agent.actual_agent_position) if self.active_agents[idx] else 0 for idx, agent in enumerate(self.fleet.vehicles)])
				# If the agent has removed trash, not penalize the distance with next closest trash #
				if np.any(self.trashes_removed_per_agent):
					r_negative_distance_to_trash = np.array([0 if idx in self.trashes_removed_per_agent else r_negative_distance_to_trash[idx] for idx, agent in enumerate(self.fleet.vehicles)])
			else:
				r_negative_distance_to_trash = np.zeros(self.n_agents)


			rewards = np.zeros(self.n_agents) \
					  + r_for_discover_trash * self.reward_weights[self.explorers_team_id] \
					  + r_for_cleaned_trash * self.reward_weights[self.cleaners_team_id] \
					  + r_for_discover_new_area * self.reward_weights[2] \
					  + r_negative_distance_to_trash * self.reward_weights[3] \
					  + r_time_penalization \

		elif self.reward_function == 'backtosimpledistanceppo':
			# ALL TEAMS #
			# Penalization for collision #
			penalization_for_collision = np.array([-50 if idx in self.collisions_mask_dict and self.collisions_mask_dict[idx] else 0 for idx in range(self.n_agents)])
			
			# CLEANERS TEAM #
			cleaners_alive = [idx for idx, agent_team in enumerate(self.team_id_of_each_agent) if agent_team == self.cleaners_team_id and self.active_agents[idx]]
			r_for_cleaned_trash = np.array([len(self.trashes_removed_per_agent[idx]) if idx in cleaners_alive and idx in self.trashes_removed_per_agent else 0 for idx in range(self.n_agents)])

			# If there is known trash, reward trough distance to closer trash #
			if np.any(self.model_trash_map):
				actual_distance_to_closest_trash = [self.get_distance_to_closest_known_trash(agent.actual_agent_position) if self.active_agents[idx] else 0 for idx, agent in enumerate(self.fleet.vehicles)]
				r_for_taking_action_that_approaches_to_trash = np.array([self.get_distance_to_closest_known_trash(agent.previous_agent_position, previous_model=True) - actual_distance_to_closest_trash[idx] if self.active_agents[idx] else 0 for idx, agent in enumerate(self.fleet.vehicles)])
				if np.any(self.previous_trashes_removed_per_agent):
					r_for_taking_action_that_approaches_to_trash = np.array([self.get_distance_to_closest_known_trash(agent.previous_agent_position, previous_model=False) - actual_distance_to_closest_trash[idx] if idx in self.previous_trashes_removed_per_agent else r_for_taking_action_that_approaches_to_trash[idx] for idx, agent in enumerate(self.fleet.vehicles)])
			else:
				r_for_taking_action_that_approaches_to_trash = np.zeros(self.n_agents)

			rewards = np.zeros(self.n_agents) \
					  + r_for_cleaned_trash * self.reward_weights[self.cleaners_team_id] \
					  + r_for_taking_action_that_approaches_to_trash \
					  + penalization_for_collision \
					  
		else:
			print(f"Reward function {self.reward_function} not implemented!!")
			exit()

		return {agent_id: rewards[agent_id] if self.active_agents[agent_id] else 0 for agent_id in range(self.n_agents)}
	
	def reset_seeds(self):
		""" Reset the seeds of the environment. """

		np.random.seed(self.seed)
		self.rng_initial_agents_positions = np.random.default_rng(seed=self.seed)
		self.rng_wind_direction = np.random.default_rng(seed=self.seed)
		self.rng_trash_elements_number = np.random.default_rng(seed=self.seed)
		self.rng_trash_positions_MVN = np.random.default_rng(seed=self.seed)
		self.rng_pollution_spots_number = np.random.default_rng(seed=self.seed)
		self.rng_pollution_spots_locations_indexes = np.random.default_rng(seed=self.seed)
		self.rng_obstacle_number = np.random.default_rng(seed=self.seed)
		self.rng_obstacle_positions = np.random.default_rng(seed=self.seed)
	
	def get_trashes_at_sight(self):
		""" Returns the number of trashes at sight of each agent. """

		trashes_at_sight = {}
		for idx, agent in enumerate(self.fleet.vehicles):
			if self.active_agents[idx]:
				trashes_at_sight[idx] = np.sum(self.model_trash_map[agent.influence_mask.astype(bool)])
			else:
				trashes_at_sight[idx] = 0
		
		return trashes_at_sight

	def get_traveled_distances(self):

		return {idx: veh.distance_traveled for idx, veh in enumerate(self.fleet.vehicles)}

	def get_percentage_cleaned_trash(self):
		""" Returns the percentage of cleaned trash. """

		return 1 - len(self.trash_positions_yx) / self.initial_number_of_trash_elements
	
	def get_percentage_discovered_trash(self):
		""" Returns the percentage of discovered trash. """
		
		trash_remaining_discovered = np.sum(self.trash_remaining_info['step_discover']!=-1)
		trash_removed = len(self.trash_removed_info)
		total_discovered_trash = trash_remaining_discovered + trash_removed

		return total_discovered_trash / self.initial_number_of_trash_elements

	def get_closest_known_trash_to_position(self, position):
		""" Returns the position of the closer known trash to the given position. """

		known_trash_positions = np.argwhere(self.model_trash_map > 0)
		return known_trash_positions[np.argmin(np.linalg.norm(known_trash_positions - position, axis = 1))]
	
	def get_distance_to_closest_known_trash(self, position, previous_model=False):
		""" Returns the distance from the closer known trash to the given position. """

		if previous_model:
			trash_positions = np.argwhere(self.previous_model_trash_map > 0)
		else:
			trash_positions = np.argwhere(self.model_trash_map > 0)

		if 'logdijkstra' in self.reward_function:
			distances_to_trash = [np.log(d**3 + 1) if (d:= self.dijkstra_distance_map[tuple(position)][tuple(trash_pos)]) > 0 else 0 for trash_pos in trash_positions] # Avoid log(0)
			# distances_to_trash = [np.log(d + 1) if d > 0 else 0 for d in distances_to_trash] # Avoid log(0)
		elif 'dijkstra' in self.reward_function:
			distances_to_trash = [self.dijkstra_distance_map[tuple(position)][tuple(trash_pos)] for trash_pos in trash_positions]
		elif 'astar' in self.reward_function:
			# Calculate the distance to the closest trash with A* algorithm
			distances_to_trash = [a_star_find_path(self.scenario_map, tuple(position), tuple(trash_pos), return_distance=True) for trash_pos in trash_positions]
		else:
			distances_to_trash = np.linalg.norm(trash_positions - position, axis = 1)

		return np.min(distances_to_trash)
	
	def check_if_there_was_reachable_trash(self, previous_position):
		""" Return if there was a reachable trash in the previous step, and it is still there. """
		
		previous_trash_positions = np.argwhere(self.previous_model_trash_map > 0)
		distances_to_previous_trashes = np.linalg.norm(previous_trash_positions - previous_position, axis = 1)
		previous_reachable_trash_positions = previous_trash_positions[distances_to_previous_trashes < 1.5]

		actual_trash_positions_viewed_from_past = np.argwhere(self.model_trash_map > 0)
		distances_to_actual_trashes_viewed_from_past = np.linalg.norm(actual_trash_positions_viewed_from_past - previous_position, axis = 1)
		actual_reachable_trash_positions_viewed_from_past = actual_trash_positions_viewed_from_past[distances_to_actual_trashes_viewed_from_past < 1.5] # 1,5 ~ np.sqrt(2*self.movement_length_by_team[self.cleaners_team_id])

		if len(previous_reachable_trash_positions) > 0:
			all_present = np.all([np.any(np.all(position == actual_reachable_trash_positions_viewed_from_past, axis=1))
						for position in previous_reachable_trash_positions])
		else:
			all_present = False
		
		return all_present

	def get_active_cleaners_positions(self):
		
		return {idx: veh.actual_agent_position for idx, veh in enumerate(self.fleet.vehicles) if self.active_agents[idx] and veh.team_id == self.cleaners_team_id}

	def get_active_agents_positions_dict(self):

		return {idx: veh.actual_agent_position for idx, veh in enumerate(self.fleet.vehicles) if self.active_agents[idx]}

	def get_model_mse(self):
		""" Returns the trash MSE. The model and the real trash map are compared as density of trash, filtered with a gaussian filter. """

		sigma = 1 # standard deviation for gaussian kernel
		real_trash_density = gaussian_filter(self.real_trash_map, sigma=sigma)
		model_trash_density = gaussian_filter(self.model_trash_map, sigma=sigma)

		return mean_squared_error(real_trash_density, model_trash_density)

	def get_model_rmse(self):
		""" Returns the trash RMSE. The model and the real trash map are compared as density of trash, filtered with a gaussian filter. """

		sigma = 1 # standard deviation for gaussian kernel
		real_trash_density = gaussian_filter(self.real_trash_map, sigma=sigma)
		model_trash_density = gaussian_filter(self.model_trash_map, sigma=sigma)

		return root_mean_squared_error(real_trash_density, model_trash_density)
	
	def get_changes_in_model(self):
		""" Returns the changes in the model """

		return np.sum(np.abs(self.model_trash_map - self.previous_model_trash_map))

	def get_redundancy_max(self):
		""" Returns the max number of agents that are in overlapping areas. """
		
		return np.max(self.redundancy_mask)
	
	def save_environment_configuration(self, path):
		""" Save the environment configuration in the current directory as a json file"""

		environment_configuration = {

			'scenario_map_name': self.scenario_map_name,
			'number_of_agents_by_team': self.number_of_agents_by_team,
			'n_actions': self.n_actions_by_team,
			'max_distance_travelled_by_team': self.max_distance_travelled_by_team,
			'max_steps_per_episode': self.max_steps_per_episode,
			'fleet_initial_positions': self.backup_fleet_initial_positions_entry if isinstance(self.backup_fleet_initial_positions_entry, str) else self.backup_fleet_initial_positions_entry.tolist(),
			'seed': self.seed,
			'movement_length_by_team': self.movement_length_by_team,
			'vision_length_by_team': self.vision_length_by_team,
			'flag_to_check_collisions_within': self.flag_to_check_collisions_within,
			'max_collisions': self.max_collisions,
			'reward_function': self.reward_function,
			'reward_weights': self.reward_weights,
			'dynamic': self.dynamic,
			'obstacles': self.obstacles,
		}

		with open(path + '/environment_config.json', 'w') as f:
			json.dump(environment_configuration, f, indent=4)
	
@staticmethod
def is_reachable(navigation_map, current_position, next_position):
	""" Check if the next position is reachable or navigable """
	if navigation_map[int(next_position[0]), int(next_position[1])] == 0:
		return False 
	x, y = next_position
	dx = x - current_position[0]
	dy = y - current_position[1]
	steps = max(abs(dx), abs(dy))
	dx = dx / steps if steps != 0 else 0
	dy = dy / steps if steps != 0 else 0
	reachable = True
	for step in range(1, steps + 1):
		px = round(current_position[0] + dx * step)
		py = round(current_position[1] + dy * step)
		if navigation_map[px, py] == 0:
			reachable = False
			break

	return reachable

if __name__ == '__main__':

	from Algorithms.DRL.ActionMasking.ActionMaskingUtils import ConsensusSafeActionMasking
	
	seed = 24
	np.random.seed(seed)
	scenario_map_name = 'acoruna_port' # ypacarai_map_low_res, ypacarai_lake_58x41, acoruna_port, marinapalamos, comb_port, challenging_map, challenging_map_big

	# Agents info #
	n_actions_explorers = 8
	n_actions_cleaners = 8
	n_explorers = 2
	n_cleaners = 2
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
		initial_positions = np.array([[32, 7], [30, 7], [28, 7], [26, 7]])[:n_agents, :] # a coruña port
		# initial_positions = None

	# Create environment # 
	env = MultiAgentCleanupEnvironment(scenario_map_name = scenario_map_name,
							   number_of_agents_by_team=(n_explorers,n_cleaners),
							   n_actions_by_team=(n_actions_explorers, n_actions_cleaners),
							   max_distance_travelled_by_team = (max_distance_travelled_explorers, max_distance_travelled_cleaners),
							   max_steps_per_episode = max_steps_per_episode,
							   max_trashes_per_agent = None,
							   fleet_initial_positions = initial_positions, # None, 'area', 'fixed' or positions array
							   seed = seed,
							   movement_length_by_team =  (movement_length_explorers, movement_length_cleaners),
							   vision_length_by_team = (vision_length_explorers, vision_length_cleaners),
							   flag_to_check_collisions_within = True,
							   max_collisions = 1000,
							   reward_function = 'negativelogdijkstra',
							   reward_weights = (1, 20, 2, 10),
							   dynamic = True,
							   obstacles = True,
							   show_plot_graphics = True,
							 )
	
	action_masking_module = ConsensusSafeActionMasking(navigation_map = env.scenario_map, 
													angle_set_of_each_agent = env.angle_set_of_each_agent,  
													movement_length_of_each_agent = env.movement_length_of_each_agent)
 
	env.reset_env()
	env.render()

	R = [] # reward
	MSE = [] 
	
	actions = {i: np.random.randint(env.n_actions_of_each_agent[i]) for i in range(n_agents)} 
	done = {i:False for i in range(n_agents)} 

	while any([not value for value in done.values()]): # while at least 1 active
	
		q = {idx: np.random.rand(env.n_actions_of_each_agent[idx]) for idx in range(n_agents) if env.active_agents[idx]} # only generate q values for active agents

		for agent_id, action in actions.items(): 
			if env.active_agents[agent_id]:
				if action == 8: # if action is stay, overwrite q of actual action to a very low value, so it will not be selected again
					q[agent_id][action] = -1000
				else:
					q[agent_id][action] = 1000 # overwrite q of actual action to a very high value, so it will be selected until collision
	
		actions = action_masking_module.query_actions(q, env.get_active_agents_positions_dict()) # only generate actions for active agents
		
		s, r, done = env.step(actions)

		R.append(list(r.values()))
		MSE.append(env.get_model_mse())

		print("Actions: " + str(dict(sorted(actions.items()))))
		print("Rewards: " + str(r))

	env.render()
	plt.show()

	# Reward and Error final graphs #
	final_fig, final_axes = plt.subplots(1, 2, figsize=(15,5))

	final_axes[0].plot(np.cumsum(np.asarray(R),axis=0), '-o')
	final_axes[0].set(title = 'Reward', xlabel = 'Step', ylabel = 'Individual Reward')
	final_axes[0].legend([f'Agent {i}' for i in range(n_agents)])
	final_axes[0].grid()

	final_axes[1].plot(MSE, '-o')
	final_axes[1].set(title = 'Error', xlabel = 'Step', ylabel = 'Mean Squared Error')
	final_axes[1].grid()

	plt.show()

	print("Finish")