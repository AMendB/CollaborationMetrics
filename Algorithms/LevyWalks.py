import numpy as np
from scipy.stats import levy

class LevyWalksFleet:
    """ Class that implements the Levy Walks strategy for explorer agents and Dijkstra pathfinding for cleaner agents. """

    def __init__(self, env, seed):
        
        # Get environment info #
        self.env = env
        self.n_agents = self.env.n_agents
        self.explorers_team_id = self.env.explorers_team_id
        self.cleaners_team_id = self.env.cleaners_team_id
        self.team_id_of_each_agent = self.env.team_id_of_each_agent
        self.movement_length_of_each_agent = self.env.movement_length_of_each_agent
        self.angle_set_of_each_agent = self.env.angle_set_of_each_agent
        self.vision_length_of_each_agent = self.env.vision_length_of_each_agent
        self.n_actions_of_each_agent = self.env.n_actions_of_each_agent
        self.fleet = self.env.fleet

        self.current_action = {agent_id: None for agent_id in range(self.n_agents)}
        self.rng = np.random.default_rng(seed=seed)

        self.levy_step_length = {agent_id: 1 for agent_id in range(self.n_agents)}   
    
    def action_to_vector(self, action, agent_id):
        """ Transform an action to a vector """

        vectors = np.array([[np.cos(2*np.pi*i/self.n_actions_of_each_agent[agent_id]), np.sin(2*np.pi*i/self.n_actions_of_each_agent[agent_id])] for i in range(self.n_actions_of_each_agent[agent_id])])

        return np.round(vectors[action]).astype(int)
    
    def is_reachable(self, action, current_position, agent_id):
        """ Check if the there is a path between the current position and the next position. """
        next_position = current_position + self.action_to_vector(action, agent_id) * self.movement_length_of_each_agent[agent_id]
        next_position = np.round(next_position).astype(int)

        if self.navigable_map[int(next_position[0]), int(next_position[1])] == 0:
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
            if self.navigable_map[px, py] == 0:
                reachable = False
                break

        return reachable
    
    def get_explorer_action(self, current_position, agent_id):
        """ Get the action for an explorer agent using Levy Walks. """

        if self.current_action[agent_id] is None:
            self.current_action[agent_id] = self.select_action_without_collision(current_position, agent_id)
        
        # Compute if there is an obstacle or reached the border #
        OBS = not self.is_reachable(self.current_action[agent_id], current_position, agent_id)

        if OBS or self.levy_step_length[agent_id] <= 0:
            self.current_action[agent_id] = self.select_action_without_collision(current_position, agent_id)

        self.levy_step_length[agent_id] -= 1
        if self.levy_step_length[agent_id] <= 0:
            # Sample a new step length from a Levy distribution #
            self.levy_step_length[agent_id] = int(levy.rvs(scale=1, size=1, random_state=self.rng)[0])
            if self.levy_step_length[agent_id] < 1:
                self.levy_step_length[agent_id] = 1
            if self.levy_step_length[agent_id] > 0.8*max(self.navigable_map.shape):
                self.levy_step_length[agent_id] = int(0.8*max(self.navigable_map.shape))

        return self.current_action[agent_id]
    
    def get_dijkstra_action(self, current_position, agent_id):
        """ Get the action for a cleaner agent to get closer to the trash using Dijkstra distance. """

        agent = self.fleet.vehicles[agent_id]

        next_movements = np.array([(0,0) if angle < 0 else np.round([np.cos(angle), np.sin(angle)]) * agent.movement_length for angle in agent.angle_set]).astype(int)
        next_positions = current_position + next_movements
        next_positions = np.clip(next_positions, (0,0), np.array(self.navigable_map.shape)-1) # saturate movement if out of indexes values (map edges)
        next_allowed_actionpose_dict = {action: next_position for action, next_position in enumerate(next_positions) if self.is_reachable(action, current_position, agent_id)} # remove next positions that leads to a collision
        
        selected_action = None
        min_distance_to_trash = np.inf
        for action, next_position in next_allowed_actionpose_dict.items():
            distance_to_trash = self.env.get_distance_to_closest_known_trash(next_position)
            if distance_to_trash < min_distance_to_trash:
                min_distance_to_trash = distance_to_trash
                selected_action = action
        
        return selected_action

    def opposite_action(self, action, agent_id):
        """ Compute the opposite action """
        return (action + self.n_actions_of_each_agent[agent_id]//2) % self.n_actions_of_each_agent[agent_id]

    def select_action_without_collision(self, current_position, agent_id):
        """ Select an action without collision """
        actions_caused_collision = [not self.is_reachable(action, current_position, agent_id) for action in range(self.n_actions_of_each_agent[agent_id])]

        # Select a random action without collision and that is not the oppositve previous action #
        if self.current_action[agent_id] is not None:
            oppos_action = self.opposite_action(self.current_action[agent_id], agent_id)
            actions_caused_collision[oppos_action] = True
        try:
            action = self.rng.choice(np.where(np.logical_not(actions_caused_collision))[0])
        except:
            # action = np.random.randint(self.number_of_actions)
            action = oppos_action

        return action

    def get_agent_action(self, current_position, agent_id):
        """ Get the action for a single agent given the conditions of the environment. """

        if self.team_id_of_each_agent[agent_id] == self.explorers_team_id:
            action = self.get_explorer_action(current_position, agent_id)
        elif self.team_id_of_each_agent[agent_id] == self.cleaners_team_id and not np.any(self.model_trash_map):
            action = self.get_explorer_action(current_position, agent_id)
        else:
            action = self.get_dijkstra_action(current_position, agent_id)
        
        next_position = current_position + self.action_to_vector(action, agent_id) * self.movement_length_of_each_agent[agent_id]

        return action, next_position

    def get_agents_actions(self):
        """ Get the actions for each agent given the conditions of the environment. """
        
        self.navigable_map = self.env.scenario_map.copy() # 1 where navigable, 0 where not navigable
        self.model_trash_map = self.env.model_trash_map
        active_agents_positions = self.env.get_active_agents_positions_dict()

        actions = {}

        for agent_id, agent_position in sorted(active_agents_positions.items(), reverse=True): # First decide the cleaners team
            actions[agent_id], next_position = self.get_agent_action(agent_position, agent_id)
            self.navigable_map[next_position[0], next_position[1]] = 0 # to avoid collisions between agents
        
        return actions