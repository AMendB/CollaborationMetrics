import numpy as np

class OneStepGreedyFleet:
    """ Class to implement a 1-step greedy agent for the CleanupEnvironment. """

    def __init__(self, env) -> None:

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
        self.reward_weights = self.env.reward_weights

    def is_reachable(self, navigation_map, current_position, next_position):
        """ Check if the there is a path between the current position and the next position. """
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

    def compute_influence_mask(self, agent_position, vision_length): 
        """ Compute influence area around actual position. It is what the agent can see. """

        influence_mask = np.zeros_like(self.navigable_map) 

        pose_x, pose_y = agent_position

        # State - coverage area #
        range_x_axis = np.arange(0, self.navigable_map.shape[0]) # posible positions in x-axis
        range_y_axis = np.arange(0, self.navigable_map.shape[1]) # posible positions in y-axis

        # Compute the circular mask (area) #
        mask = (range_x_axis[np.newaxis, :] - pose_x) ** 2 + (range_y_axis[:, np.newaxis] - pose_y) ** 2 <= vision_length ** 2 

        influence_mask[mask.T] = 1.0 # converts True values to 1 and False values to 0

        return influence_mask
    
    def get_agent_action_with_best_future_reward(self, agent_position, agent_id):
        """ Get the most rewarding action for the agent given the conditions of the environment. """

        agent = self.fleet.vehicles[agent_id]

        next_movements = np.array([(0,0) if angle < 0 else np.round([np.cos(angle), np.sin(angle)]) * agent.movement_length for angle in agent.angle_set]).astype(int)
        next_positions = agent_position + next_movements
        next_positions = np.clip(next_positions, (0,0), np.array(self.navigable_map.shape)-1) # saturate movement if out of indexes values (map edges)
        next_allowed_actionpose_dict = {action: next_position for action, next_position in enumerate(next_positions) if self.is_reachable(self.navigable_map, agent_position, next_position)} # remove next positions that leads to a collision
        # next_allowed_actionpose_dict = {action: next_position for action, next_position in enumerate(next_positions) if self.navigable_map[next_position[0], next_position[1]] == 1} # remove next positions that leads to a collision
        
        best_action = None
        best_reward = -np.inf
        dict_actions_rewards = {}

        for action, next_position in next_allowed_actionpose_dict.items():
            future_influence_mask = self.compute_influence_mask(next_position, agent.vision_length)
            
            # ALL TEAMS #
            if np.any(self.model_trash_map):
                r_for_taking_action_that_approaches_to_trash = -self.env.get_distance_to_closest_known_trash(next_position)
                if self.model_trash_map[next_position[0], next_position[1]] > 0:
                    r_for_taking_action_that_approaches_to_trash = 0
            else:
                r_for_taking_action_that_approaches_to_trash = 0


            # EXPLORERS TEAM #
            if agent.team_id == self.explorers_team_id:
                r_for_discover_new_area = (self.visited_areas_map[future_influence_mask.astype(bool)] == 1).sum()
            else:  
                r_for_discover_new_area = 0
            
            # CLEANERS TEAM #
            if agent.team_id == self.cleaners_team_id:
                r_for_cleaned_trash = self.model_trash_map[next_position[0], next_position[1]]
            else:
                r_for_cleaned_trash = 0

            # Exchange ponderation between exploration/exploitation when the 80% of the map is visited #
            # if self.percentage_visited > 0.8:
            #     ponderation_for_discover_new_area = self.reward_weights[self.explorers_team_id]
            # else:
            #     ponderation_for_discover_new_area = self.reward_weights[2]
            ponderation_for_discover_new_area = self.reward_weights[2]

            reward = r_for_taking_action_that_approaches_to_trash \
                        + r_for_discover_new_area * ponderation_for_discover_new_area \
                        + r_for_cleaned_trash * self.reward_weights[self.cleaners_team_id] \
            
            dict_actions_rewards[action] = reward

            if reward > best_reward:
                best_reward = reward
                best_action = action

        # Check if the best reward is the same for multiple actions. If so, choose randomly between them.
        if list(dict_actions_rewards.values()).count(best_reward) > 1:
            best_action = np.random.choice([action for action, reward in dict_actions_rewards.items() if reward == best_reward])

        # In case no action is selected, choose randomly any action
        if best_action is None:
            best_action = np.random.choice(self.n_actions_of_each_agent[agent_id])
            next_position = agent_position # stays in the same position
        else:
            next_position = next_allowed_actionpose_dict[best_action]

        return best_action, next_position

    def get_agents_actions(self):
        """ Get the actions for each agent given the conditions of the environment. """
        
        self.navigable_map = self.env.scenario_map.copy() # 1 where navigable, 0 where not navigable
        self.visited_areas_map = self.env.visited_areas_map.copy()
        self.model_trash_map = self.env.model_trash_map
        self.percentage_visited = self.env.percentage_visited
        active_agents_positions = self.env.get_active_agents_positions_dict()

        actions = {}

        for agent_id, agent_position in sorted(active_agents_positions.items(), reverse=True): # First decide the cleaners team
            actions[agent_id], next_position = self.get_agent_action_with_best_future_reward(agent_position, agent_id)
            self.navigable_map[next_position[0], next_position[1]] = 0 # to avoid collisions between agents
        
        return actions
    
    def get_agent_potential_rewards(self, agent_position, agent_id):
        
        agent = self.fleet.vehicles[agent_id]

        next_movements = np.array([(0,0) if angle < 0 else np.round([np.cos(angle), np.sin(angle)]) * agent.movement_length for angle in agent.angle_set]).astype(int)
        next_positions = agent_position + next_movements
        next_positions = np.clip(next_positions, (0,0), np.array(self.navigable_map.shape)-1) # saturate movement if out of indexes values (map edges)
        next_allowed_actionpose_dict = {action: next_position for action, next_position in enumerate(next_positions) if self.is_reachable(self.navigable_map, agent_position, next_position)} # remove next positions that leads to a collision
        # next_allowed_actionpose_dict = {action: next_position for action, next_position in enumerate(next_positions) if self.navigable_map[next_position[0], next_position[1]] == 1} # remove next positions that leads to a non-navigable area 
        
        best_reward = -np.inf
        rewards = {}
        for action, next_position in enumerate(next_positions):
            if action in next_allowed_actionpose_dict:
                future_influence_mask = self.compute_influence_mask(next_position, agent.vision_length)
                
                # ALL TEAMS #
                if np.any(self.model_trash_map):
                    r_for_taking_action_that_approaches_to_trash = -self.env.get_distance_to_closest_known_trash(next_position)
                    if self.model_trash_map[next_position[0], next_position[1]] >0:
                        r_for_taking_action_that_approaches_to_trash = 0
                else:
                    r_for_taking_action_that_approaches_to_trash = 0

                # EXPLORERS TEAM #
                if agent.team_id == self.explorers_team_id:
                    r_for_discover_new_area = (self.visited_areas_map[future_influence_mask.astype(bool)] == 1).sum()
                else:  
                    r_for_discover_new_area = 0
                
                # CLEANERS TEAM #
                if agent.team_id == self.cleaners_team_id:
                    r_for_cleaned_trash = self.model_trash_map[next_position[0], next_position[1]]                
                else:
                    r_for_cleaned_trash = 0

                # Exchange ponderation between exploration/exploitation when the 80% of the map is visited #
                # if self.percentage_visited > 0.8:
                #     ponderation_for_discover_new_area = self.reward_weights[self.explorers_team_id]
                # else:
                #     ponderation_for_discover_new_area = self.reward_weights[2]
                ponderation_for_discover_new_area = self.reward_weights[2]

                reward = r_for_taking_action_that_approaches_to_trash \
                            + r_for_discover_new_area * ponderation_for_discover_new_area \
                            + r_for_cleaned_trash * self.reward_weights[self.cleaners_team_id] \
                            
                rewards[action] = reward
                
                if reward > best_reward:
                    best_reward = reward
            else:
                rewards[action] = -np.inf    

        # Check if the best reward is the same for multiple actions. If so, choose randomly between them and add an extra.
        if list(rewards.values()).count(best_reward) > 1:
            best_action = np.random.choice([action for action, reward in rewards.items() if reward == best_reward])     
            rewards[best_action] += 1   

        return np.array([*rewards.values()], dtype = np.float32)

    def get_agents_q_values(self):
        """ Get the q_values for each agent given the conditions of the environment. """
        
        self.navigable_map = self.env.scenario_map.copy() # 1 where navigable, 0 where not navigable
        self.visited_areas_map = self.env.visited_areas_map.copy()
        self.model_trash_map = self.env.model_trash_map 
        self.percentage_visited = self.env.percentage_visited
        active_agents_positions = self.env.get_active_agents_positions_dict()

        q_values = {}

        for agent_id, agent_position in active_agents_positions.items():
            q_values[agent_id] = self.get_agent_potential_rewards(agent_position, agent_id)
        
        return q_values
