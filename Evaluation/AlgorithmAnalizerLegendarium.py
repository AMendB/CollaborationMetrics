import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import colorcet
import numpy as np
sys.path.append('.')

from sylegendarium import Legendarium
from sylegendarium import load_experiment_pd
from sylegendarium import load_experiments

import pandas as pd
import mysql.connector
from datetime import datetime
import utm

def calculate_average_metrics(df_processed):

    df = df_processed.copy()

    max_steps_per_episode = df['max_steps_per_episode'][0]
    
    # # Plot the average metrics: percentage_of_trash_collected, traveled_distances, model_rmse...
    # plt.figure(figsize=(10, 6))
    # # Plot mean and 95% confidence interval
    # sns.lineplot(data=df, x='step', y='percentage_of_trash_collected', hue='algorithm', palette='tab10')
    # sns.lineplot(data=df, x='step', y='percentage_of_trash_discovered', hue='algorithm', palette='tab10', linestyle='--')
    # plt.title(f'Average Percentage of Trash Collected (line) and Discovered (dashed) - {map_name}')
    # plt.xlabel('Step')
    # plt.ylabel('Percentage')
    # plt.legend()
    # # plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # # Plot the average metrics: model_rmse...
    # plt.figure(figsize=(10, 6))
    # sns.lineplot(data=df, x='step', y='model_rmse', hue='algorithm', palette='tab10')
    # plt.title(f'Average Model RMSE - {map_name}')
    # plt.xlabel('Step')
    # plt.ylabel('Model RMSE')
    # plt.legend()
    # # plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    #  Print average and confidence interval of metrics: percentage of trash collected (PTC) and RMSE at 33%, 66% and 100% of the steps
    for agent_id in range(df['n_agents'][0]):
        df[f'acc_rewards_{agent_id}'] = df.groupby(['run', 'map_name', 'algorithm', 'objective', 'reward_function', 'reward_weights'])[f'rewards_{agent_id}'].cumsum()
    columns_names_metrics = ['avg_rmse', 'ci_rmse', 'avg_ptc', 'ci_ptc']
    columns_names_acc_rewards = [f'avg_acc_rewards_{i}' for i in range(df['n_agents'][0])]
    df_table = pd.DataFrame(columns=['algorithm'] 
                            + [name + f'_{percentage}%' for name in columns_names_metrics for percentage in [33, 66, 100]]
                            + [name + f'_{percentage}%' for name in columns_names_acc_rewards for percentage in [33, 66, 100]])
    
    for algorithm in df['algorithm'].unique():
        df_alg = df[df['algorithm'] == algorithm]
        row = {name: np.nan for name in list(df_table.columns)}
        row['algorithm'] = algorithm
        for percentage in [33, 66, 100]:
            step = round(max_steps_per_episode*percentage/100)
            row[f'avg_rmse_{percentage}%'] = df_alg[df_alg['step'] == step]['model_rmse'].mean()
            row[f'ci_rmse_{percentage}%'] = df_alg[df_alg['step'] == step]['model_rmse'].sem(ddof=0) * 1.96
            row[f'avg_ptc_{percentage}%'] = df_alg[df_alg['step'] == step]['percentage_of_trash_collected'].mean() * 100
            row[f'ci_ptc_{percentage}%'] = df_alg[df_alg['step'] == step]['percentage_of_trash_collected'].sem(ddof=0) * 1.96 * 100
            avg_acc_rewards = [df_alg[df_alg['step'] == step][f'acc_rewards_{i}'].mean() for i in range(df_alg['n_agents'].iloc[0])]
            ci_acc_rewards = [df_alg[df_alg['step'] == step][f'acc_rewards_{i}'].sem(ddof=0) * 1.96 for i in range(df_alg['n_agents'].iloc[0])]
            
            for i, (avg, ci) in enumerate(zip(avg_acc_rewards, ci_acc_rewards)):
                row[f'avg_acc_rewards_{i}_{percentage}%'] = avg
                row[f'ci_acc_rewards_{i}_{percentage}%'] = ci
        df_table = pd.concat([df_table, pd.DataFrame([row])], ignore_index=True)
    
    # Print information
    # avg_ptc_rmse_dict = {}
    print(f"Average PTC and RMSE with CI at 100% of the steps for each algorithm:")
    for algorithm in df_table['algorithm'].unique():
        avg_ptc = df_table[df_table['algorithm'] == algorithm]['avg_ptc_100%'].values[0]
        ci_ptc = df_table[df_table['algorithm'] == algorithm]['ci_ptc_100%'].values[0]
        avg_rmse = df_table[df_table['algorithm'] == algorithm]['avg_rmse_100%'].values[0]
        ci_rmse = df_table[df_table['algorithm'] == algorithm]['ci_rmse_100%'].values[0]
        # avg_ptc_rmse_dict[algorithm] = {'avg_ptc': avg_ptc, 'ci_ptc': ci_ptc, 'avg_rmse': avg_rmse, 'ci_rmse': ci_rmse}
        print(f"{algorithm} -- PTC: {avg_ptc:.2f} ± {ci_ptc}, RMSE: {avg_rmse} ± {ci_rmse}")


def epsilon_study(epsilon_study_path, save_data = False):

    experiment_name = 'evaluation'
    df_experiment = pd.DataFrame()

    # Load every experiment in subfolders
    for folder in os.listdir(epsilon_study_path):
        if os.path.isdir(os.path.join(epsilon_study_path, folder)):
            df_new = load_experiment_pd(experiment_name, os.path.join(epsilon_study_path, folder))
            df_experiment = pd.concat([df_experiment, df_new], ignore_index=True)

    map_name = df_experiment['map_name'].unique()[0]
    team_id_of_each_agent = df_experiment['team_id_of_each_agent'][0]
    teams_name = df_experiment['teams_name'][0]
    df_experiment = df_experiment.drop(columns=['team_id_of_each_agent', 'teams_name'])
    team_name_of_each_agent = {i: teams_name[team_id_of_each_agent[i]] for i in range(len(team_id_of_each_agent))}
    df_experiment['initial_trash'] = df_experiment.groupby('run')['trash_remaining'].transform('max')

    df = df_experiment.copy()
    df = df[(df['algorithm'] != 'LawnMower') & (df['algorithm'] != 'WanderingAgent')].reset_index(drop=True)

    # Add epsilon and epsilon_team to the algorithm name
    df['algorithm'] = df.apply(lambda row: f"Eps{row['epsilon']}_EpsTm{row['epsilon_team']}_{row['algorithm']}" if not pd.isna(row['epsilon']) and not pd.isna(row['epsilon_team']) else row['algorithm'], axis=1)

    # Delete unnecessary columns for analysis
    columns_to_drop = ['date', 'agents_positions', 'actions', 'dones', 'ground_truth', 'model']
    df.drop(columns=columns_to_drop, inplace=True)

    # Padding the DataFrame with the last value to ensure all episodes have the same number of steps
    max_steps_per_episode = df['max_steps_per_episode'][0]
    df = df.groupby(['run', 'map_name', 'algorithm', 'objective', 'reward_function', 'reward_weights'])[df.columns].apply(lambda group: group.set_index('step').reindex(range(max_steps_per_episode+1), method='ffill').reset_index(), include_groups=False).reset_index(drop=True)
    # Remove all episodes that last the maximum number of steps 
    # df = df.groupby(['run', 'map_name', 'algorithm', 'objective', 'reward_function', 'reward_weights']).filter(lambda group: (group['step'] < max_steps_per_episode).all()).reset_index(drop=True)
    # Remove all episodes that ended early
    # df = df.groupby(['run', 'map_name', 'algorithm', 'objective', 'reward_function', 'reward_weights']).filter(lambda group: (group['step'] == max_steps_per_episode).any()).reset_index(drop=True)

    # Convert columns with dictionaries to separate columns
    columns_with_dicts = ['rewards', 'reward_components', 'traveled_distances', 'cleaned_trashes', 'history_cleaned_trashes', 'trashes_at_sight']
    for col in columns_with_dicts:
        if col in df.columns:
            # Expand the dictionary column into separate columns
            dict_df = pd.json_normalize(df[col]).fillna(0)  # Fill NaN values with 0
            if col == 'cleaned_trashes':
                # Avoid empty dicts in 'cleaned_trashes' column
                dict_df = dict_df.map(lambda x: len(x) if isinstance(x, np.ndarray) else 0)
            # Rename the columns to include the original column name
            dict_df.columns = [f"{col}_{sub_col}" for sub_col in dict_df.columns]
            # Concatenate the new columns to the original DataFrame
            df = pd.concat([df, dict_df], axis=1)
            # Drop the original dictionary column
            df.drop(columns=[col], inplace=True) 


    # Print average and confidence interval of metrics: percentage of trash collected (PTC) and RMSE at 33%, 66% and 100% of the steps
    for agent_id in range(df['n_agents'][0]):
        df[f'acc_rewards_{agent_id}'] = df.groupby(['run', 'map_name', 'algorithm', 'objective', 'reward_function', 'reward_weights'])[f'rewards_{agent_id}'].cumsum()
    columns_names_metrics = ['avg_rmse', 'ci_rmse', 'avg_ptc', 'ci_ptc']
    columns_names_acc_rewards = [f'avg_acc_rewards_{i}' for i in range(df['n_agents'][0])]
    df_table = pd.DataFrame(columns=['algorithm'] 
                            + [name + f'_{percentage}%' for name in columns_names_metrics for percentage in [33, 66, 100]]
                            + [name + f'_{percentage}%' for name in columns_names_acc_rewards for percentage in [33, 66, 100]])
    
    for algorithm in df['algorithm'].unique():
        df_alg = df[df['algorithm'] == algorithm]
        row = {name: np.nan for name in list(df_table.columns)}
        row['algorithm'] = algorithm
        row['epsilon'] = df_alg['epsilon'].iloc[0]
        row['epsilon_team'] = df_alg['epsilon_team'].iloc[0]
        for percentage in [33, 66, 100]:
            step = round(max_steps_per_episode*percentage/100)
            row[f'avg_rmse_{percentage}%'] = df_alg[df_alg['step'] == step]['model_rmse'].mean()
            row[f'ci_rmse_{percentage}%'] = df_alg[df_alg['step'] == step]['model_rmse'].sem(ddof=0) * 1.96
            row[f'avg_ptc_{percentage}%'] = df_alg[df_alg['step'] == step]['percentage_of_trash_collected'].mean() * 100
            row[f'ci_ptc_{percentage}%'] = df_alg[df_alg['step'] == step]['percentage_of_trash_collected'].sem(ddof=0) * 1.96 * 100
            avg_acc_rewards = [df_alg[df_alg['step'] == step][f'acc_rewards_{i}'].mean() for i in range(df_alg['n_agents'].iloc[0])]
            ci_acc_rewards = [df_alg[df_alg['step'] == step][f'acc_rewards_{i}'].sem(ddof=0) * 1.96 for i in range(df_alg['n_agents'].iloc[0])]
            
            for i, (avg, ci) in enumerate(zip(avg_acc_rewards, ci_acc_rewards)):
                row[f'avg_acc_rewards_{i}_{percentage}%'] = avg
                row[f'ci_acc_rewards_{i}_{percentage}%'] = ci
        df_table = pd.concat([df_table, pd.DataFrame([row])], ignore_index=True)
    
    # Print information grouped by epsilon_team
    if save_data:
        file_path = os.path.join(epsilon_study_path, f'epsilon_study_results_{map_name}.txt')
        file = open(file_path, 'w')
    print(f"Average PTC and RMSE with CI at 100% of the steps for each algorithm:")
    for epsilon_team in df['epsilon_team'].unique():
        text_line = f"\nEpsilon Team: {teams_name[epsilon_team]}"
        print(f"{text_line}")
        if save_data:
            file.write(text_line + '\n')
        df_team = df_table[df_table['epsilon_team'] == epsilon_team]
        for algorithm in df_team['algorithm'].unique():
            avg_ptc = df_team[df_team['algorithm'] == algorithm]['avg_ptc_100%'].values[0]
            ci_ptc = df_team[df_team['algorithm'] == algorithm]['ci_ptc_100%'].values[0]
            avg_rmse = df_team[df_team['algorithm'] == algorithm]['avg_rmse_100%'].values[0]
            ci_rmse = df_team[df_team['algorithm'] == algorithm]['ci_rmse_100%'].values[0]
            text_line = f"{algorithm} -- PTC: {avg_ptc:.2f} ± {ci_ptc}, RMSE: {avg_rmse} ± {ci_rmse}"
            print(f"{text_line}")
            if save_data:
                file.write(text_line + '\n')
    if save_data:
        file.close()


    def get_slope(epsilon_team=None, metric=None, x=None, y=None):
        # If input x and y are not given, calculate them from df_table
        if x is None or y is None:
            df_team = df_table[df_table['epsilon_team'] == epsilon_team]
            x = df_team['epsilon']
            y = df_team[f'avg_{metric}_100%']
        slope, intercept, r_value, p_value, std_err = linregress(x=x, y=y)
        return slope, intercept, r_value

    def plot_epsilon_linear_regression(epsilon_team, metric):
        df_team = df_table[df_table['epsilon_team'] == epsilon_team]
        x = df_team['epsilon']
        y = df_team[f'avg_{metric}_100%']
        slope, intercept, r_value = get_slope(x=x, y=y)
        print(f"Linear regression results for {metric.upper()} vs epsilon - {map_name}:")
        print(f"Slope: {slope}, Intercept: {intercept}, R-squared: {r_value**2}")

        if metric == 'rmse':
            label = 'Root Mean Squared Error'
        elif metric == 'ptc':
            label = 'Percentage of Trash Collected'

        plt.figure(figsize=(10, 6))
        plt.errorbar(
            x=df_team['epsilon'],
            y=df_team[f'avg_{metric}_100%'],
            yerr=df_team[f'ci_{metric}_100%'],
            fmt='o', capsize=5, label=label, color='black'
        )
        plt.plot(df_team['epsilon'], slope * x + intercept, color='red', label='Linear Regression')
        plt.title(f'Linear Regression for {metric.upper()} vs Epsilon {teams_name[epsilon_team]} - {map_name}')
        plt.xlabel('Epsilon')
        plt.ylabel(f'{metric.upper()}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        path_fig = os.path.join(epsilon_study_path, f'epsilon_study_{metric}_EpsTm{epsilon_team}_{map_name}.svg')
        plt.savefig(path_fig)
        plt.show()
        # save figure
        plt.close()

    # Calculate linear regression for PTC vs epsilon, and plot it
    from scipy.stats import linregress
    for epsilon_team in df['epsilon_team'].unique():
        print(f"\nEpsilon Team: {teams_name[epsilon_team]}")
        plot_epsilon_linear_regression(epsilon_team, 'ptc')
        plot_epsilon_linear_regression(epsilon_team, 'rmse')

    # Sentivity matrix:
    # slope of RMSE over epsilon for each epsilon_team
    # slope of PTC over epsilon for each epsilon_team
    sensitivity_matrix = pd.DataFrame(columns=['epsilon_team', 'slope_rmse', 'slope_ptc'])
    for epsilon_team in df['epsilon_team'].unique():
        slope_rmse, _, _ = get_slope(epsilon_team=epsilon_team, metric='rmse')
        slope_ptc, _, _ = get_slope(epsilon_team=epsilon_team, metric='ptc')
        # Add row to the sensitivity_matrix dataframe
        sensitivity_matrix = pd.concat([sensitivity_matrix if not sensitivity_matrix.empty else None, pd.DataFrame({'epsilon_team': ['Policy of ' + teams_name[epsilon_team]], 'slope_rmse': [slope_rmse], 'slope_ptc': [slope_ptc]})], ignore_index=True)
    print(f"\nSensitivity Matrix:\n {sensitivity_matrix}")

    if save_data:
        with open(file_path, 'a') as file:
            file.write(f"\nSensitivity Matrix:\n {sensitivity_matrix.to_string(index=False)}")

def process_dataframe(df_experiment):
    
    df = df_experiment.copy()

    # Filter out LawnMower and WanderingAgent algorithms
    df = df[(df['algorithm'] != 'LawnMower') & (df['algorithm'] != 'WanderingAgent')].reset_index(drop=True)
    df['algorithm'] = df['algorithm'].replace('DRLIndependent_Networks_Per_Team', 'DRL')

    # Delete unnecessary columns for analysis
    columns_to_drop = ['date', 'agents_positions', 'actions', 'dones', 'ground_truth', 'model']
    df.drop(columns=columns_to_drop, inplace=True)

    # Padding the DataFrame with the last value to ensure all episodes have the same number of steps
    max_steps_per_episode = df['max_steps_per_episode'][0]
    df = df.groupby(['run', 'map_name', 'algorithm', 'objective', 'reward_function', 'reward_weights'])[df.columns].apply(lambda group: group.set_index('step').reindex(range(max_steps_per_episode+1), method='ffill').reset_index(), include_groups=False).reset_index(drop=True)
    # Remove all episodes that last the maximum number of steps 
    # df = df.groupby(['run', 'map_name', 'algorithm', 'objective', 'reward_function', 'reward_weights']).filter(lambda group: (group['step'] < max_steps_per_episode).all()).reset_index(drop=True)
    # Remove all episodes that ended early
    # df = df.groupby(['run', 'map_name', 'algorithm', 'objective', 'reward_function', 'reward_weights']).filter(lambda group: (group['step'] == max_steps_per_episode).any()).reset_index(drop=True)


    # Convert columns with dictionaries to separate columns
    columns_with_dicts = ['rewards', 'reward_components', 'traveled_distances', 'cleaned_trashes', 'history_cleaned_trashes', 'trashes_at_sight', 'trash_removed_info', 'trash_remaining_info']
    for col in columns_with_dicts:
        if col in df.columns:
            # Expand the dictionary column into separate columns
            dict_df = pd.json_normalize(df[col]).fillna(0)  # Fill NaN values with 0
            if col == 'cleaned_trashes':
                # Convert the 'cleaned_trashes' column to a list of integers to avoid empty dicts when there are no cleaned trashes
                dict_df = dict_df.map(lambda x: len(x) if isinstance(x, np.ndarray) else 0)
            # Rename the columns to include the original column name
            dict_df.columns = [f"{col}_{sub_col}" for sub_col in dict_df.columns]
            # Concatenate the new columns to the original DataFrame
            df = pd.concat([df, dict_df], axis=1)
            # Drop the original dictionary column
            df.drop(columns=[col], inplace=True)
        
    return df
    

def calculate_secondary_metrics(df_experiment):
    ''' This part is to analyze the results in depth, including: 
    - From the total discovered by explorers, how many are cleaned?
    - From the total of cleaned, how many have been discovered by explorers?
    - From the total of trash discovered, how many have been discovered by explorers?
    - From the total of trash discovered, how many have been discovered by cleaners?
    - Mean trashes discovered at each step and mean trashes cleaned at each step by each vehicle
    - Time since trash is discovered until it is removed. Boxplot at the end of the experiment to see all trash removed
    - Time since trash is discovered by explorers until it is removed
    - Time since trash is discovered by cleaners until it is removed
    - Mean trashes at sight by each vehicle
    - Time distance from the mean peak of scouts reward to the mean peak of cleaners reward
    - Boxplot between the time step difference between the mean peak of scouts reward and the mean peak of cleaners reward
    Among others...
    # TODO: Change boxplot to violinplot
    '''

    df = df_experiment.copy()

    # Filter out LawnMower and WanderingAgent algorithms
    df = df[(df['algorithm'] != 'LawnMower') & (df['algorithm'] != 'WanderingAgent')].reset_index(drop=True)
    df['algorithm'] = df['algorithm'].replace('DRLIndependent_Networks_Per_Team', 'DRL')

    # Delete unnecessary columns for analysis
    columns_to_drop = ['date', 'agents_positions', 'actions', 'dones', 'ground_truth', 'model']
    df.drop(columns=columns_to_drop, inplace=True)

    # Padding the DataFrame with the last value to ensure all episodes have the same number of steps
    max_steps_per_episode = df['max_steps_per_episode'][0]
    df = df.groupby(['run', 'map_name', 'algorithm', 'objective', 'reward_function', 'reward_weights'])[df.columns].apply(lambda group: group.set_index('step').reindex(range(max_steps_per_episode+1), method='ffill').reset_index(), include_groups=False).reset_index(drop=True)

    # Convert columns with dictionaries to separate columns
    columns_with_dicts = ['rewards', 'reward_components', 'traveled_distances', 'cleaned_trashes', 'history_cleaned_trashes', 'trashes_at_sight']
    for col in columns_with_dicts:
        if col in df.columns:
            # Expand the dictionary column into separate columns
            dict_df = pd.json_normalize(df[col]).fillna(0)  # Fill NaN values with 0
            if col == 'cleaned_trashes':
                # Convert the 'cleaned_trashes' column to a list of integers to avoid empty dicts when there are no cleaned trashes
                dict_df = dict_df.map(lambda x: len(x) if isinstance(x, np.ndarray) else 0)
            # Rename the columns to include the original column name
            dict_df.columns = [f"{col}_{sub_col}" for sub_col in dict_df.columns]
            # Concatenate the new columns to the original DataFrame
            df = pd.concat([df, dict_df], axis=1)
            # Drop the original dictionary column
            df.drop(columns=[col], inplace=True) 

    # Extract information from np.full arrays
    explorers_ids = [i for i, team in team_name_of_each_agent.items() if team == 'Explorers']
    cleaners_ids = [i for i, team in team_name_of_each_agent.items() if team == 'Cleaners']

    remaining_discovered_by_explorers = df['trash_remaining_info'].apply(lambda x: len([veh for veh in x['vehicle_discover'] if veh in explorers_ids]))
    removed_discovered_by_explorers = df['trash_removed_info'].apply(lambda x: len([veh for veh in x['vehicle_discover'] if veh in explorers_ids]))
    df['total_discovered_by_explorers'] = remaining_discovered_by_explorers + removed_discovered_by_explorers

    remaining_discovered_by_cleaners = df['trash_remaining_info'].apply(lambda x: len([veh for veh in x['vehicle_discover'] if veh in cleaners_ids]))
    removed_discovered_by_cleaners = df['trash_removed_info'].apply(lambda x: len([veh for veh in x['vehicle_discover'] if veh in cleaners_ids]))
    df['total_discovered_by_cleaners'] = remaining_discovered_by_cleaners + removed_discovered_by_cleaners

    total_removed = df['trash_removed_info'].apply(lambda x: len(x))
    remaining_discovered = df['trash_remaining_info'].apply(lambda x: np.sum(x['vehicle_discover'] != -1))
    total_discovered = remaining_discovered + total_removed

    # From the total discovered by explorers, how many are cleaned?
    df['percentage_cleaned_of_discovered_by_explorers'] = (removed_discovered_by_explorers / df['total_discovered_by_explorers'] * 100).fillna(0)

    # From the total of cleaned, how many have been discovered by explorers?
    df['percentage_cleaned_discovered_by_explorers'] = (removed_discovered_by_explorers / total_removed * 100).fillna(0)

    # From the total of trash discovered, how many have been discovered by explorers?
    df['percentage_discovered_by_explorers_of_total_discovered'] = (df['total_discovered_by_explorers'] / total_discovered * 100).fillna(0)

    # From the total of trash discovered, how many have been discovered by cleaners?
    df['percentage_discovered_by_cleaners_of_total_discovered'] = (df['total_discovered_by_cleaners'] / total_discovered * 100).fillna(0)

    # Plot the results
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='step', y='percentage_cleaned_of_discovered_by_explorers', hue='algorithm', palette='tab10')
    plt.title(f'From the total discovered by explorers, how many are cleaned? - {map_name}')
    plt.xlabel('Step')
    plt.ylabel('Percentage (%)')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='step', y='percentage_cleaned_discovered_by_explorers', hue='algorithm', palette='tab10')
    plt.title(f'From the total of cleaned, how many have been discovered by explorers? - {map_name}')
    plt.xlabel('Step')
    plt.ylabel('Percentage (%)')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='step', y='percentage_discovered_by_explorers_of_total_discovered', hue='algorithm', palette='tab10')
    plt.title(f'From the total of trash discovered, how many have been discovered by explorers? - {map_name}')
    plt.xlabel('Step')
    plt.ylabel('Percentage (%)')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='step', y='percentage_discovered_by_cleaners_of_total_discovered', hue='algorithm', palette='tab10')
    plt.title(f'From the total of trash discovered, how many have been discovered by cleaners? - {map_name}')
    plt.xlabel('Step')
    plt.ylabel('Percentage (%)')
    plt.grid(True)
    plt.show()

    # Mean trashes discovered at each step and mean trashes cleaned at each step by each vehicle
    for idx in explorers_ids:
        # Se ha descubierto si el id del vehículo es igual al id del explorador y el paso de descubrimiento es igual al paso actual
        df[f'discovered_trashes_{idx}'] = df.apply(
            lambda row: sum(
                veh == idx and step == row['step']
                for veh, step in zip(row['trash_remaining_info']['vehicle_discover'], row['trash_remaining_info']['step_discover'])
            ),
            axis=1
        )
    
    plt.figure(figsize=(12, 6))
    for idx in explorers_ids:
        sns.lineplot(data=df[df['algorithm']=='DRL'], x='step', y=f'discovered_trashes_{idx}', hue='algorithm', palette='tab10')
    plt.title(f'Mean trashes discovered by explorers - {map_name}')
    plt.xlabel('Step')
    plt.ylabel('Number of trashes')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    for idx in cleaners_ids:
        sns.lineplot(data=df[df['algorithm']=='DRL'], x='step', y=f'cleaned_trashes_{idx}', hue='algorithm', palette='tab10')
    plt.title(f'Mean trashes cleaned by cleaners - {map_name}')
    plt.xlabel('Step')
    plt.ylabel('Number of trashes')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    for idx in explorers_ids:
        sns.lineplot(data=df[df['algorithm']=='DRL'], x='step', y=f'trashes_at_sight_{idx}', hue='algorithm', palette='tab10')
    plt.title(f'Mean trashes at sight by explorers - {map_name}')
    plt.xlabel('Step')
    plt.ylabel('Number of trashes')
    plt.grid(True)
    plt.show()

    # Time since trash is discovered until it is removed. Boxplot at the end of the experiment to see all trash removed
    df['time_from_discovery_to_removal'] = df['trash_removed_info'].apply(lambda x: x['step_remove'] - x['step_discover'])
    df_exploded = df[df['step'] == 150][['algorithm', 'time_from_discovery_to_removal']].explode('time_from_discovery_to_removal').reset_index(drop=True)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_exploded, x='algorithm', y='time_from_discovery_to_removal', hue='algorithm', palette='tab10')
    plt.title(f'Time from discovery to removal of trash - {map_name}')
    plt.xlabel('Algorithm')
    plt.ylabel('Time (steps)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    print("Average time steps from discovery to removal of trash:")
    print(df_exploded.groupby('algorithm')['time_from_discovery_to_removal'].mean())

    # Time since trash is discovered by explorers until it is removed
    df['time_from_explorers_discovery_to_removal'] = df['trash_removed_info'].apply(lambda x: [x['step_remove'][index] - x['step_discover'][index] for index in range(len(x)) if x['vehicle_discover'][index] in explorers_ids])
    df_exploded_explorers = df[df['step'] == 150][['algorithm', 'time_from_explorers_discovery_to_removal']].explode('time_from_explorers_discovery_to_removal').reset_index(drop=True)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_exploded_explorers, x='algorithm', y='time_from_explorers_discovery_to_removal', hue='algorithm', palette='tab10')
    plt.title(f'Time from SCOUTS discovery to removal of trash - {map_name}')
    plt.xlabel('Algorithm')
    plt.ylabel('Time (steps)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    # Time distance from the mean peak of scouts reward to the mean peak of cleaners reward
    df['reward_scouts'] = 0
    df['reward_cleaners'] = 0
    for idx in explorers_ids:
        df['reward_scouts'] += df[f'rewards_{idx}']
    for idx in cleaners_ids:
        df['reward_cleaners'] += df[f'rewards_{idx}']

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='step', y='reward_scouts', hue='algorithm', palette='tab10')
    plt.title(f'Reward of SCOUTS - {map_name}')
    plt.xlabel('Algorithm')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='step', y='reward_cleaners', hue='algorithm', palette='tab10')
    plt.title(f'Reward of CLEANERS - {map_name}')
    plt.xlabel('Algorithm')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.show()

    # Boxplot between the time step difference between the mean peak of scouts reward and the mean peak of cleaners reward
    step_peak_scouts = df.groupby(['algorithm', 'run'])['reward_scouts'].idxmax()
    step_peak_cleaners = df.groupby(['algorithm', 'run'])['reward_cleaners'].idxmax()
    peaks_diff = step_peak_cleaners - step_peak_scouts
    df_peaks_diff = peaks_diff.reset_index(name='value')

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_peaks_diff, x='algorithm', y='value', hue='algorithm', palette='tab10')
    plt.title(f'Time step difference between the mean peak of SCOUTS reward and the mean peak of CLEANERS reward - {map_name}')
    plt.xlabel('Algorithm')
    plt.ylabel('Time step difference')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


def derived_accumulated_rewards(df_processed):
    
    df = df_processed.copy()

    # Derived of the accumulated rewards for each team
    for agent_id in range(df['n_agents'][0]):
        df[f'acc_rewards_{agent_id}'] = df.groupby(['run', 'map_name', 'algorithm', 'objective', 'reward_function', 'reward_weights'])[f'rewards_{agent_id}'].cumsum()

    team_names = set(team_name_of_each_agent.values())
    for team in team_names:
        agent_ids_in_the_team = [agent_id for agent_id, team_name in team_name_of_each_agent.items() if team_name == team]
        df[f'acc_rewards_team_{team}'] = df[[f'acc_rewards_{agent_id}' for agent_id in agent_ids_in_the_team]].sum(axis=1)

    # Plot the accumulated rewards for each team
    for team in team_names:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='step', y=f'acc_rewards_team_{team}', hue='algorithm', palette='tab10')
        plt.title(f'Accumulated Rewards for Team {team} - {map_name}')
        plt.xlabel('Step')
        plt.ylabel('Accumulated Rewards')
        plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    # Average accumulated rewards for each team at each step
    avg_acc_rewards = pd.DataFrame()
    for team in team_names:
        avg_acc_rewards[team] = df.groupby(['step', 'algorithm'])[f'acc_rewards_team_{team}'].mean().reset_index()[f'acc_rewards_team_{team}']

    # Derived of average accumulated rewards for each team
    derived_accumulated_rewards_explorers = avg_acc_rewards['Explorers'].diff().fillna(0)
    derived_accumulated_rewards_cleaners = avg_acc_rewards['Cleaners'].diff().fillna(0)

    # Plot the derived of average accumulated rewards for each team
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=derived_accumulated_rewards_explorers, label=f'Derived of Average Accumulated Rewards - Team Explorers', color='blue')
    sns.lineplot(data=derived_accumulated_rewards_cleaners, label=f'Derived of Average Accumulated Rewards - Team Cleaners', color='orange')
    plt.title(f'Derived of Average Accumulated Rewards - {map_name}')
    plt.xlabel('Step')
    plt.ylabel('Derived of Average Accumulated Rewards')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def gini_coefficient(x):
    x = np.array(x)
    return np.sum(np.abs(np.subtract.outer(x, x)))/(2*len(x)**2*x.mean())

def gini_study(df_processed):
    ''' Study the evolution of Gini coefficient over time (steps) for several metrics:
    - Trash collected by cleaners
    - Accumulated rewards of explorers -> Gini assumes not negative values, so this metric is not valid if rewards can be negative!
    -----------
    The Gini coefficient is a measure of statistical dispersion that represents the inequality among values of a distribution.'''
    
    df = df_processed.copy()
    
    ## Evolution of Gini coefficient for trash collected by cleaners over time ##
    # Identify cleaner agents
    cleaners_ids = [i for i, team in team_name_of_each_agent.items() if team == 'Cleaners']
   
    # Calculate Gini coefficient for each step
    gini_per_step = []
    for step in range(1, df['max_steps_per_episode'].iloc[0] + 1):
        df_step = df[df['step'] == step].copy()
        for run in df_step['run'].unique():
            for alg in df_step['algorithm'].unique():
                run_data = df_step[(df_step['run'] == run) & (df_step['algorithm'] == alg)]
                if len(run_data) > 0:
                    # Get trash collected by each cleaner at this step
                    cleaner_collections = [run_data[f'history_cleaned_trashes_{c_id}'].values[0] for c_id in cleaners_ids]
                    # Only calculate Gini if at least one cleaner has collected trash
                    if sum(cleaner_collections) > 0:
                        gini = gini_coefficient(cleaner_collections)
                        gini_per_step.append({
                            'step': step,
                            'run': run,
                            'algorithm': alg,
                            'gini_coefficient': gini
                        })
    
    # Create DataFrame with Gini coefficients over time
    df_gini_evolution = pd.DataFrame(gini_per_step)
    
    # Visualize evolution of Gini coefficient over time
    if not df_gini_evolution.empty:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_gini_evolution, x='step', y='gini_coefficient', hue='algorithm', palette='tab10')
        plt.title(f'Evolution of Gini Coefficient for Trash Collection Distribution - {map_name}')
        plt.xlabel('Step')
        plt.ylabel('Gini Coefficient (0=equal, 1=unequal)')
        plt.grid(True)
        plt.show()


    # ## Evolution of Gini coefficient for accumulated rewards of explorers over time ##
    # # Identify explorer agents
    # explorers_ids = [i for i, team in team_name_of_each_agent.items() if team == 'Explorers']

    # # Calculate accumulated rewards for each explorer
    # for agent_id in explorers_ids:
    #     df[f'acc_rewards_{agent_id}'] = df.groupby(['run', 'map_name', 'algorithm', 'objective', 'reward_function', 'reward_weights'])[f'rewards_{agent_id}'].cumsum()
    
    # # Calculate Gini coefficient for each step
    # gini_per_step = []
    # for step in range(1, df['max_steps_per_episode'].iloc[0] + 1):
    #     df_step = df[df['step'] == step].copy()
    #     for run in df_step['run'].unique():
    #         for alg in df_step['algorithm'].unique():
    #             run_data = df_step[(df_step['run'] == run) & (df_step['algorithm'] == alg)]
    #             if len(run_data) > 0:
    #                 # Get accumulated rewards by each explorer at this step
    #                 explorer_rewards = [run_data[f'acc_rewards_{e_id}'].values[0] for e_id in explorers_ids]
    #                 # Only calculate Gini if at least one explorer has a reward
    #                 if sum(explorer_rewards) > 0:
    #                     gini = gini_coefficient(explorer_rewards)
    #                     gini_per_step.append({
    #                         'step': step,
    #                         'run': run,
    #                         'algorithm': alg,
    #                         'gini_coefficient': gini
    #                     })

    # # Create DataFrame with Gini coefficients over time
    # df_gini_evolution = pd.DataFrame(gini_per_step)
    
    # # Visualize evolution of Gini coefficient over time
    # if not df_gini_evolution.empty:
    #     plt.figure(figsize=(12, 6))
    #     sns.lineplot(data=df_gini_evolution, x='step', y='gini_coefficient', hue='algorithm', palette='tab10')
    #     plt.title(f'Evolution of Gini Coefficient for Explorers Accumulated Rewards Distribution - {map_name}')
    #     plt.xlabel('Step')
    #     plt.ylabel('Gini Coefficient (0=equal, 1=unequal)')
    #     plt.grid(True)
    #     plt.show()

if __name__ == '__main__':
    
    experiment_name = 'evaluation'
    # experiment_name = 'merged_experiment'

    # Load the experiment data´
    df_experiment = load_experiment_pd(experiment_name, 'Evaluation/Results/EPSILON_TEST_comb_port.4.timenegativelogdijkstra_2.30086970665265_2.1921767314965317_1.8604886309628847_4.098596840990528/Eps_0.3_EpsTm_0_comb_port.4.timenegativelogdijkstra_2.30086970665265_2.1921767314965317_1.8604886309628847_4.098596840990528')
    # df_experiment = load_experiment_pd(experiment_name, 'Evaluation/Results/eps0.3_comb_port.4.negativelogdijkstra_2.30086970665265_2.1921767314965317_1.8604886309628847_4.098596840990528')
    # df_experiment = load_experiment_pd(experiment_name, 'Evaluation/Results/eps0.3_comb_port.4.negativelogdijkstra_2.30086970665265_2.1921767314965317_1.8604886309628847_4.098596840990528')
    map_name = df_experiment['map_name'].unique()[0]
    team_id_of_each_agent = df_experiment['team_id_of_each_agent'][0]
    teams_name = df_experiment['teams_name'][0]
    df_experiment = df_experiment.drop(columns=['team_id_of_each_agent', 'teams_name'])
    team_name_of_each_agent = {i: teams_name[team_id_of_each_agent[i]] for i in range(len(team_id_of_each_agent))}
    df_experiment['initial_trash'] = df_experiment.groupby('run')['trash_remaining'].transform('max')
    # print(df_experiment.head())
    # print(df_experiment.columns)

    df_processed = process_dataframe(df_experiment)

    # calculate_average_metrics(df_processed)
    # derived_accumulated_rewards(df_processed)

    # epsilon_study(epsilon_study_path='Evaluation/Results/EPSILON_TEST_greedydijkstra_recompensamal',
    epsilon_study(epsilon_study_path='Evaluation/Results/EPSILON_TEST_comb_port.4.timenegativelogdijkstra_2.30086970665265_2.1921767314965317_1.8604886309628847_4.098596840990528',
                  save_data=True)

    # calculate_secondary_metrics(df_processed)

    # gini_study(df_processed)


