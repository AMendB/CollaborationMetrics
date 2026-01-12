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
from scipy.stats import linregress


def calculate_global_metrics(df_processed, show_plots=True, experiment_path=None, save_data = False):

    df = df_processed.copy()
    max_steps_per_episode = df['max_steps_per_episode'][0]
    df['normalized_step'] = df['step'] / max_steps_per_episode
    
    # A.a. Global Metrics: Percentage of Target Achieved (PTA) for both teams
    plt.figure(figsize=(4, 2.7))  # Adjust for single column (~8.5 cm)
    sns.lineplot(data=df, x='normalized_step', y='percentage_of_trash_collected',
                hue='algorithm', linewidth=1.6)
    sns.lineplot(data=df, x='normalized_step', y='percentage_of_trash_discovered',
                hue='algorithm', linestyle='--', linewidth=1.6)
    handles, labels = plt.gca().get_legend_handles_labels()
    algs = df['algorithm'].unique()
    new_labels = [f"{lbl} (Collected)" for lbl in algs] + [f"{lbl} (Discovered)" for lbl in algs]
    plt.legend(handles, new_labels, fontsize=7)
    plt.title(f'Average Percentage of Target Achieved', fontsize=9)
    plt.xlabel('Normalized Time', fontsize=8)
    plt.ylabel('Percentage', fontsize=8)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    if save_data:
        plt.savefig(os.path.join(experiment_path, f'A.a_average_pta_both_teams_{map_name}.pdf'), bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()
    plt.close()
    
    # A.b. Global Metrics: Root Mean Squared Error (RMSE)
    plt.figure(figsize=(4, 2))  # Adjust for single column (~8.5 cm)
    sns.lineplot(data=df, x='normalized_step', y='model_rmse',
                hue='algorithm', linewidth=1.6)
    plt.legend(fontsize=7)
    plt.title(f'Average Root Mean Squared Error', fontsize=9)
    plt.xlabel('Normalized Time', fontsize=8)
    plt.ylabel('RMSE', fontsize=8)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    if save_data:
        plt.savefig(os.path.join(experiment_path, f'A.b_average_rmse_{map_name}.pdf'), bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()
    plt.close()

    # A.c. Global Metrics: Normalized time to x% of trash collected
    for percentage in [50, 90]:
        _ = return_normalized_time_to_percentage(df, percentage=percentage)

    # A.d. Global Metrics: Troughput
    _ = return_troughput(df)

    # A.e. Global Metrics: Idleness Reduction Rate
    plot_idleness_reduction_rate(df, show_plots, experiment_path, save_data)

    # A.f. Global Metrics: Expected Service Cost Reduction ponderated_distances_from_cleaners_to_known_trash
    df['cost_service'] = df['ponderated_distances_from_cleaners_to_known_trash'].apply(
        lambda dict_of_arrays: sum(np.sum(arr) for arr in dict_of_arrays.values())
    )
    def conditional_diff(series):
        """ Calculates the difference (current_step - previous_step) only if the previous_step is NOT zero. Otherwise, it returns NaN."""
        result = []
        # The first element is set to zero
        result.append(0) 
        
        # Iterate from the second element (index 1)
        for i in range(1, len(series)):
            current_value = series.iloc[i]
            previous_value = series.iloc[i-1]
            
            if previous_value != 0:
                # If the previous value is NOT zero, calculate the difference
                diff = previous_value - current_value
                result.append(diff)
            else:
                # If the previous value IS zero, the result is zero
                result.append(0)
                
        # Return the result as a Pandas Series, preserving the original index
        return result

    # Apply the custom function to the 'sum_of_all_distances' column after grouping
    df['cost_service_reduction'] = (
        df
        .groupby(["algorithm", "run"])['cost_service']
        .transform(conditional_diff)
    )

    plt.figure(figsize=(4, 2.7))  # Adjust for single column (~8.5 cm)
    sns.lineplot(data=df, x='normalized_step', y='cost_service_reduction',
                hue='algorithm', linewidth=1.6)
    plt.legend(fontsize=7)
    plt.title(f'Average Cost Service Reduction', fontsize=9)
    plt.xlabel('Normalized Time', fontsize=8)
    plt.ylabel('Cost Service Reduction', fontsize=8)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    if save_data:
        plt.savefig(os.path.join(experiment_path, f'A.f_average_cost_service_reduction_{map_name}.pdf'), bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()
    plt.close()

def calculate_inter_team_metrics(df_processed, show_plots=True, experiment_path=None, save_data = False):

    df = df_processed.copy()
    max_steps_per_episode = df['max_steps_per_episode'][0]
    df['normalized_step'] = df['step'] / max_steps_per_episode

    explorers_ids = [i for i, team in team_name_of_each_agent.items() if team == 'Explorers' or team == 'Scouts']
    cleaners_ids = [i for i, team in team_name_of_each_agent.items() if team == 'Cleaners' or team == 'Foragers']

    # B.a. Inter-team Metrics: Discovery-to-Service Latency
    df['time_from_explorers_discovery_to_removal'] = df['trash_removed_info'].apply(lambda x: [(x['step_remove'][index] - x['step_discover'][index])/max_steps_per_episode for index in range(len(x)) if x['vehicle_discover'][index] in explorers_ids])
    df_exploded_explorers = df[df['step'] == 150][['algorithm', 'time_from_explorers_discovery_to_removal']].explode('time_from_explorers_discovery_to_removal').reset_index(drop=True)
    plt.figure(figsize=(4, 3))
    sns.violinplot(data=df_exploded_explorers, x='time_from_explorers_discovery_to_removal', y='algorithm', hue='algorithm', palette='tab10')
    plt.title(f'Discovery-to-Service Latency', fontsize=9)
    plt.xlabel('Normalized Time', fontsize=8)
    # plt.ylabel('Algorithm', fontsize=8)
    plt.gca().set_ylabel('')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7, rotation=90, va='center')
    plt.grid(True, linewidth=0.3, which='both')
    plt.tight_layout()
    if save_data:
        plt.savefig(os.path.join(experiment_path, f'B.a_discovery_to_service_latency.pdf'), bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()
    plt.close()

    # B.b. Inter-team Metrics: Inter Team Temporal Lag
    df["max_variation_pta_scouts"] = df.groupby(["algorithm", "run"])["percentage_of_trash_discovered"].diff().abs()
    df["max_variation_pta_cleaners"] = df.groupby(["algorithm", "run"])["percentage_of_trash_collected"].diff().abs()
    step_peak_scouts = df.groupby(['algorithm', 'run'])['max_variation_pta_scouts'].idxmax()
    step_peak_cleaners = df.groupby(['algorithm', 'run'])['max_variation_pta_cleaners'].idxmax()
    number_steps_episode = df.groupby(['algorithm', 'run'])['step'].max()
    peaks_diff = np.abs(step_peak_cleaners - step_peak_scouts)/ number_steps_episode
    df_peaks_diff = peaks_diff.reset_index(name='value')
    plt.figure(figsize=(4, 3))
    sns.violinplot(data=df_peaks_diff, x='value', y='algorithm', hue='algorithm', palette='tab10')#, cut=0)
    plt.title(f'Inter Team Temporal Lag', fontsize=9)
    plt.xlabel('Normalized Time', fontsize=8)
    # plt.ylabel('Algorithm', fontsize=8)
    plt.gca().set_ylabel('')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7, rotation=90, va='center')
    plt.grid(True, linewidth=0.3, which='both')
    plt.tight_layout()
    if save_data:
        plt.savefig(os.path.join(experiment_path, f'B.b_inter_team_temporal_lag.pdf'), bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()
    plt.close()

    # B.c. Inter-team Metrics: Cooperative Success Ratio = From the total of cleaned, how many have been discovered by explorers?
    removed_discovered_by_explorers = df['trash_removed_info'].apply(lambda x: len([veh for veh in x['vehicle_discover'] if veh in explorers_ids]))
    acc_removed = df['trash_removed_info'].apply(lambda x: len(x))
    df['ratio_cleaned_discovered_by_explorers'] = (removed_discovered_by_explorers / acc_removed).fillna(0)
    plt.figure(figsize=(4, 2))
    sns.lineplot(data=df, x='normalized_step', y='ratio_cleaned_discovered_by_explorers', hue='algorithm', palette='tab10')
    plt.legend(fontsize=7)
    plt.title(f'Cooperative Success Ratio', fontsize=9)
    plt.xlabel('Normalized Time', fontsize=8)
    plt.ylabel('CSR', fontsize=8)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.grid(True, linewidth=0.3, which='both')
    plt.tight_layout()
    if save_data:
        plt.savefig(os.path.join(experiment_path, f'B.c_cooperative_success_ratio.pdf'), bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()
    plt.close()
    print("\nFinal Cooperative Success Ratio values:")
    print(df[df['normalized_step'] == 1].groupby('algorithm')['ratio_cleaned_discovered_by_explorers'].mean())

    # B.d. Inter-team Metrics: Robustness of Cooperation (RoC)
    # TODO

    # B.e. Inter-team Metrics: Cooperation Sensitivity under Stochastic Corruption (CSSC)
    epsilon_study_paths = [
        'Evaluation/Results/FINAL_ALGORITHMS_EPSILON_TEST/EPSILON_TEST_comb_port.4.negativedijkstra_2.12..._SEED=3',
        'Evaluation/Results/FINAL_ALGORITHMS_EPSILON_TEST/EPSILON_TEST_greedydijkstra_2.12..._SEED=3',
        'Evaluation/Results/FINAL_ALGORITHMS_EPSILON_TEST/EPSILON_TEST_levywalksdijkstra_2.12..._SEED=3'
    ]
    plot_coop_sens_stochastic_corruption(epsilon_study_paths=epsilon_study_paths, show_plots=show_plots, save_data=save_data)

def calculate_intra_team_metrics(df_processed, show_plots=True, experiment_path=None, save_data = False):

    df = df_processed.copy()
    max_steps_per_episode = df['max_steps_per_episode'][0]

    explorers_ids = [i for i, team in team_name_of_each_agent.items() if team == 'Explorers' or team == 'Scouts']
    cleaners_ids = [i for i, team in team_name_of_each_agent.items() if team == 'Cleaners' or team == 'Foragers']

    # C.a. Intra-team Metrics: Gini Coefficient of Items Collected among Foragers
    gini_study(df, show_plots=show_plots, experiment_path=experiment_path, save_data=save_data)

    # C.b. Intra-team Metrics: Coverage Overlap (CO) (among Scouts)
    plot_coverage_overlap(df, show_plots=show_plots, experiment_path=experiment_path, save_data=save_data)

    # C.c. Intra-team Metrics: Marginal Contribution (MC)
    # Manually calculate with average_metrics results


def calculate_average_metrics(df_processed, show_plots=True, experiment_path=None, save_data = False):

    df = df_processed.copy()

    max_steps_per_episode = df['max_steps_per_episode'][0]
    
    # # Plot the average metrics: percentage_of_trash_collected, traveled_distances, model_rmse...
    plt.figure(figsize=(10, 6))
    # Plot mean and 95% confidence interval
    sns.lineplot(data=df, x='step', y='percentage_of_trash_collected', hue='algorithm', palette='tab10')
    sns.lineplot(data=df, x='step', y='percentage_of_trash_discovered', hue='algorithm', palette='tab10', linestyle='--')
    plt.title(f'Average Percentage of Trash Collected (line) and Discovered (dashed) - {map_name}')
    plt.xlabel('Step')
    plt.ylabel('Percentage')
    # plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_data:
        plt.savefig(os.path.join(experiment_path, f'average_ptc_and_discovered_{map_name}.svg'))
    if show_plots:
        plt.show()
    plt.close()


    # # Plot the average metrics: model_rmse...
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='step', y='model_rmse', hue='algorithm', palette='tab10')
    plt.title(f'Average Model RMSE - {map_name}')
    plt.xlabel('Step')
    plt.ylabel('Model RMSE')
    # plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_data:
        plt.savefig(os.path.join(experiment_path, f'average_model_rmse_{map_name}.svg'))
    if show_plots:
        plt.show()
    plt.close()


    #  Calculate average and confidence interval of metrics: percentage of trash collected (PTC) and RMSE at 33%, 66% and 100% of the steps
    for agent_id in range(df['n_agents'][0]):
        df[f'acc_rewards_{agent_id}'] = df.groupby(['run', 'map_name', 'algorithm', 'objective', 'reward_function', 'reward_weights'])[f'rewards_{agent_id}'].cumsum()
    columns_names_metrics = ['avg_rmse', 'ci_rmse', 'avg_ptc', 'ci_ptc', 'avg_ptd', 'ci_ptd']
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
            row[f'avg_ptd_{percentage}%'] = df_alg[df_alg['step'] == step]['percentage_of_trash_discovered'].mean() * 100
            row[f'ci_ptd_{percentage}%'] = df_alg[df_alg['step'] == step]['percentage_of_trash_discovered'].sem(ddof=0) * 1.96 * 100
            avg_acc_rewards = [df_alg[df_alg['step'] == step][f'acc_rewards_{i}'].mean() for i in range(df_alg['n_agents'].iloc[0])]
            ci_acc_rewards = [df_alg[df_alg['step'] == step][f'acc_rewards_{i}'].sem(ddof=0) * 1.96 for i in range(df_alg['n_agents'].iloc[0])]
            
            for i, (avg, ci) in enumerate(zip(avg_acc_rewards, ci_acc_rewards)):
                row[f'avg_acc_rewards_{i}_{percentage}%'] = avg
                row[f'ci_acc_rewards_{i}_{percentage}%'] = ci
        df_table.loc[len(df_table)] = row
    
    # Print information
    # avg_ptc_rmse_dict = {}
    print(f"\nAverage PTC and RMSE with CI at 100% of the steps for each algorithm:")
    if save_data:
        file_path = os.path.join(experiment_path, f'average_metrics_{df["map_name"].unique()[0]}.txt')
        file = open(file_path, 'w')
    for algorithm in df_table['algorithm'].unique():
        avg_ptc = df_table[df_table['algorithm'] == algorithm]['avg_ptc_100%'].values[0]
        ci_ptc = df_table[df_table['algorithm'] == algorithm]['ci_ptc_100%'].values[0]
        avg_rmse = df_table[df_table['algorithm'] == algorithm]['avg_rmse_100%'].values[0]
        ci_rmse = df_table[df_table['algorithm'] == algorithm]['ci_rmse_100%'].values[0]
        avg_ptd = df_table[df_table['algorithm'] == algorithm]['avg_ptd_100%'].values[0]
        ci_ptd = df_table[df_table['algorithm'] == algorithm]['ci_ptd_100%'].values[0]
        # avg_ptc_rmse_dict[algorithm] = {'avg_ptc': avg_ptc, 'ci_ptc': ci_ptc, 'avg_rmse': avg_rmse, 'ci_rmse': ci_rmse}
        text_line = f"{algorithm} -- PTC: {avg_ptc:.2f}, ci: {ci_ptc:.2f} || RMSE: {avg_rmse:.4f}, ci: {ci_rmse:.4f} || PTD: {avg_ptd:.2f}, ci: {ci_ptd:.2f}"
        if save_data:
            file.write(text_line + '\n')
        print(text_line)

    print("\nFinished calculating average metrics.\n")

def get_slope(df_table=None, epsilon_team=None, metric=None, x=None, y=None):
    # If input x and y are not given, calculate them from df_table
    if x is None or y is None:
        df_team = df_table[df_table['epsilon_team'] == epsilon_team]
        x = df_team['epsilon']
        y = df_team[f'avg_{metric}_100%']
    slope, intercept, r_value, p_value, std_err = linregress(x=x, y=y)
    return slope, intercept, r_value

def epsilon_study(epsilon_study_path, show_plots = False, save_data = False):

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
    # If 'Explorers' and 'Cleaners' are in teams_name, rename them to 'Scouts' and 'Foragers'
    for team_id, team_name in teams_name.items():
        if team_name == 'Explorers':
            teams_name[team_id] = 'Scouts'
        elif team_name == 'Cleaners':
            teams_name[team_id] = 'Foragers'
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
    columns_with_dicts = ['rewards', 'reward_components', 'traveled_distances', 'cleaned_trashes', 'history_cleaned_trashes', 'trashes_at_sight', 'coverage_overlap_ratio']
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
    columns_names_metrics = ['avg_rmse', 'ci_rmse', 'avg_ptc', 'ci_ptc', 'avg_ptd', 'ci_ptd']
    columns_names_acc_rewards = [f'avg_acc_rewards_{i}' for i in range(df['n_agents'][0])]
    df_table = pd.DataFrame(columns=['algorithm'] 
                            + [name + f'_{percentage}%' for name in columns_names_metrics for percentage in [33, 66, 100]]
                            + [name + f'_{percentage}%' for name in columns_names_acc_rewards for percentage in [33, 66, 100]]
                            + ['epsilon', 'epsilon_team', 'rmse_all_values', 'ptc_all_values', 'ptd_all_values'])
    
    for algorithm in df['algorithm'].unique():
        df_alg = df[df['algorithm'] == algorithm].reset_index(drop=True)
        row = {name: np.nan for name in list(df_table.columns)}
        row['algorithm'] = algorithm
        row['epsilon'] = df_alg['epsilon'].iloc[0]
        row['epsilon_team'] = df_alg['epsilon_team'].iloc[0]
        row[f'rmse_all_values'] = df_alg[df_alg['step'] == max_steps_per_episode]['model_rmse'].values
        row[f'ptc_all_values'] = df_alg[df_alg['step'] == max_steps_per_episode]['percentage_of_trash_collected'].values
        row[f'ptd_all_values'] = df_alg[df_alg['step'] == max_steps_per_episode]['percentage_of_trash_discovered'].values
        for percentage in [33, 66, 100]:
            step = round(max_steps_per_episode*percentage/100)
            row[f'avg_rmse_{percentage}%'] = df_alg[df_alg['step'] == step]['model_rmse'].mean()
            row[f'ci_rmse_{percentage}%'] = df_alg[df_alg['step'] == step]['model_rmse'].sem(ddof=0) * 1.96
            row[f'avg_ptc_{percentage}%'] = df_alg[df_alg['step'] == step]['percentage_of_trash_collected'].mean() * 100
            row[f'ci_ptc_{percentage}%'] = df_alg[df_alg['step'] == step]['percentage_of_trash_collected'].sem(ddof=0) * 1.96 * 100
            row[f'avg_ptd_{percentage}%'] = df_alg[df_alg['step'] == step]['percentage_of_trash_discovered'].mean() * 100
            row[f'ci_ptd_{percentage}%'] = df_alg[df_alg['step'] == step]['percentage_of_trash_discovered'].sem(ddof=0) * 1.96 * 100
            avg_acc_rewards = [df_alg[df_alg['step'] == step][f'acc_rewards_{i}'].mean() for i in range(df_alg['n_agents'].iloc[0])]
            ci_acc_rewards = [df_alg[df_alg['step'] == step][f'acc_rewards_{i}'].sem(ddof=0) * 1.96 for i in range(df_alg['n_agents'].iloc[0])]
            
            for i, (avg, ci) in enumerate(zip(avg_acc_rewards, ci_acc_rewards)):
                row[f'avg_acc_rewards_{i}_{percentage}%'] = avg
                row[f'ci_acc_rewards_{i}_{percentage}%'] = ci
        df_table.loc[len(df_table)] = row
    
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
            avg_ptd = df_team[df_team['algorithm'] == algorithm]['avg_ptd_100%'].values[0]
            ci_ptd = df_team[df_team['algorithm'] == algorithm]['ci_ptd_100%'].values[0]
            text_line = f"{algorithm} -- PTC: {avg_ptc:.2f}, ci: {ci_ptc:.2f} || RMSE: {avg_rmse:.4f}, ci: {ci_rmse:.4f} || PTD: {avg_ptd:.2f}, ci: {ci_ptd:.2f}"
            print(f"{text_line}")
            if save_data:
                file.write(text_line + '\n')
    if save_data:
        file.close()

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
        if save_data:
            path_fig = os.path.join(epsilon_study_path, f'epsilon_study_{metric}_EpsTm{epsilon_team}_{map_name}.svg')
            plt.savefig(path_fig)
        if show_plots:
            plt.show()
        plt.close()

    def plot_epsilon_linear_regression_all_points(epsilon_team, metric):
        df_team = df_table[df_table['epsilon_team'] == epsilon_team]
        x = df_team['epsilon']
        y = df_team[f'avg_{metric}_100%']
        slope, intercept, r_value = get_slope(x=x, y=y)
        print(f"Linear regression results for {metric.upper()} vs epsilon - {map_name}:")
        print(f"Slope: {slope}, Intercept: {intercept}, R-squared: {r_value**2}")

        if metric == 'rmse':
            label = 'Root Mean Squared Error'
            w = 1
        elif metric == 'ptc':
            label = 'Percentage of Trash Collected'
            w = 100

        plt.figure(figsize=(10, 6))
        for epsilon, values in zip(df_team['epsilon'], df_team[f'{metric}_all_values']):
            plt.scatter([epsilon]*len(values), values*w, color='gray', alpha=0.5)
        plt.plot(df_team['epsilon'], slope * x + intercept, color='red', label='Linear Regression')
        plt.title(f'Linear Regression for {metric.upper()} vs Epsilon {teams_name[epsilon_team]} - {map_name}')
        plt.xlabel('Epsilon')
        plt.ylabel(f'{label}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_data:
            path_fig = os.path.join(epsilon_study_path, f'epsilon_study_{metric}_EpsTm{epsilon_team}_{map_name}_all_points.svg')
            plt.savefig(path_fig)
        if show_plots:
            plt.show()
        plt.close()

    def plot_epsilon_linear_regression_boxplot(epsilon_team, metric):
        df_team = df_table[df_table['epsilon_team'] == epsilon_team]
        x = df_team['epsilon']
        y = df_team[f'avg_{metric}_100%']
        slope, intercept, r_value = get_slope(x=x, y=y)
        print(f"Linear regression results for {metric.upper()} vs epsilon - {map_name}:")
        print(f"Slope: {slope}, Intercept: {intercept}, R-squared: {r_value**2}")

        if metric == 'rmse':
            label = 'Root Mean Squared Error'
            w = 1
        elif metric == 'ptc':
            label = 'Percentage of Trash Collected'
            w = 100

        plt.figure(figsize=(10, 6))
        for epsilon, values in zip(df_team['epsilon'], df_team[f'{metric}_all_values']):
            plt.boxplot(values*w, positions=[epsilon], widths=0.03)
        plt.plot(df_team['epsilon'], slope * x + intercept, color='red', label='Linear Regression')
        plt.title(f'Linear Regression for {metric.upper()} vs Epsilon {teams_name[epsilon_team]} - {map_name}')
        plt.xlabel('Epsilon')
        plt.ylabel(f'{label}')
        plt.xlim(-0.1,1.1)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_data:
            path_fig = os.path.join(epsilon_study_path, f'epsilon_study_{metric}_EpsTm{epsilon_team}_{map_name}_boxplot.svg')
            plt.savefig(path_fig)
        if show_plots:
            plt.show()
        plt.close()

    def plot_epsilon_linear_regression_pointplot(epsilon_team, metric):
        if metric == 'rmse':
            label = 'Root Mean Squared Error'
            name = 'RMSE'
            w = 1
        elif metric == 'ptc':
            label = 'Percentage of Target Achieved (Foragers)'
            name = 'ptaForagers'
            w = 0.01
        elif metric == 'ptd':
            label = 'Percentage of Target Achieved (Scouts)'
            name = 'ptaScouts'
            w = 0.01

        df_team = df_table[df_table['epsilon_team'] == epsilon_team]
        x = df_team['epsilon']
        y = df_team[f'avg_{metric}_100%'] * w
        slope, intercept, r_value = get_slope(x=x, y=y)
        print(f"Linear regression results for {metric.upper()} vs epsilon - {map_name}:")
        print(f"Slope: {slope}, Intercept: {intercept}, R-squared: {r_value**2}")

        # Explode for violin
        df_exploded = df_team.explode(f'{metric}_all_values').reset_index(drop=True)
        df_exploded.rename(columns={f'{metric}_all_values': f'{metric}_value'}, inplace=True)
        df_exploded[f'{metric}_value'] = df_exploded[f'{metric}_value']

        # Map categorical epsilon values to numerical positions
        eps_values = sorted(x.unique())
        positions = range(len(eps_values))
        pos_map = {e: i for i, e in enumerate(eps_values)}
        df_exploded["eps_pos"] = df_exploded["epsilon"].map(pos_map)
        if save_data:
            path_csv = os.path.join(epsilon_study_path, f'epsilon_study_{name}_EpsTm{teams_name[epsilon_team]}_{map_name}_exploded.csv')
            df_exploded.to_csv(path_csv, index=False)

        tab10_colors = sns.color_palette("tab10")

        plt.figure(figsize=(5, 2.5))
        # sns.violinplot(data=df_exploded, x='eps_pos', y=f'{metric}_value', inner='quartile', hue='epsilon', cut=0)
        # sns.barplot(data=df_exploded, x='eps_pos', y=f'{metric}_value', errorbar='ci', capsize=0.1, ci=95, palette='gray')
        sns.pointplot(data=df_exploded, x='eps_pos', y=f'{metric}_value', errorbar=('ci', 95),  
                      markers='o', markersize=2, capsize=0.3, color='black', linestyles="", linewidth=1)
        x_line = np.array(list(positions))
        y_line = slope * np.array(eps_values) + intercept
        plt.plot(x_line, y_line, linestyle='--', linewidth=1, color=tab10_colors[2], label='Linear fit')
        plt.title(f'CSSC in {teams_name[epsilon_team].lower()} team')
        plt.xlabel('Epsilon')
        plt.ylabel(f'{label}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.xticks(positions[::2], eps_values[::2])
        if save_data:
            path_fig = os.path.join(epsilon_study_path, f'epsilon_study_{name}_EpsTm{teams_name[epsilon_team]}_{map_name}_pointplot.svg')
            plt.savefig(path_fig)
            path_fig = os.path.join(epsilon_study_path, f'epsilon_study_{name}_EpsTm{teams_name[epsilon_team]}_{map_name}_pointplot.pdf')
            plt.savefig(path_fig)
        if show_plots:
            plt.show()
        plt.close()

    # Calculate linear regression for PTC vs epsilon, and plot it
    for epsilon_team in df['epsilon_team'].unique():
        print(f"\nEpsilon Team: {teams_name[epsilon_team]}")
        # plot_epsilon_linear_regression(epsilon_team, 'ptc')
        # plot_epsilon_linear_regression(epsilon_team, 'rmse')
        # plot_epsilon_linear_regression_all_points(epsilon_team, 'ptc')
        # plot_epsilon_linear_regression_all_points(epsilon_team, 'rmse')
        # plot_epsilon_linear_regression_boxplot(epsilon_team, 'ptc')
        # plot_epsilon_linear_regression_boxplot(epsilon_team, 'rmse')
        plot_epsilon_linear_regression_pointplot(epsilon_team, 'ptc')
        plot_epsilon_linear_regression_pointplot(epsilon_team, 'rmse')
        plot_epsilon_linear_regression_pointplot(epsilon_team, 'ptd')

    # Sentivity matrix:
    # slope of RMSE over epsilon for each epsilon_team
    # slope of PTC over epsilon for each epsilon_team
    sensitivity_matrix = pd.DataFrame(columns=['epsilon_team', 'slope_rmse', 'slope_ptc', 'slope_ptd'])
    for epsilon_team in df['epsilon_team'].unique():
        slope_rmse, _, _ = get_slope(df_table=df_table, epsilon_team=epsilon_team, metric='rmse')
        slope_ptc, _, _ = get_slope(df_table=df_table, epsilon_team=epsilon_team, metric='ptc')
        slope_ptd, _, _ = get_slope(df_table=df_table, epsilon_team=epsilon_team, metric='ptd')
        # Add row to the sensitivity_matrix dataframe
        sensitivity_matrix = pd.concat([sensitivity_matrix if not sensitivity_matrix.empty else None, pd.DataFrame({'epsilon_team': ['Policy of ' + teams_name[epsilon_team]], 'slope_rmse': [slope_rmse], 'slope_ptc': [slope_ptc], 'slope_ptd': [slope_ptd]})], ignore_index=True)
    print(f"\nSensitivity Matrix:\n {sensitivity_matrix}")

    if save_data:
        with open(file_path, 'a') as file:
            file.write(f"\nSensitivity Matrix:\n {sensitivity_matrix.to_string(index=False)}")
        
    print("Epsilon study analysis completed.")

def process_dataframe(df_experiment):
    
    df = df_experiment.copy()

    # df = df[(df['algorithm'] != 'LawnMower') & (df['algorithm'] != 'WanderingAgent')].reset_index(drop=True)
    df['algorithm'] = df['algorithm'].replace('DRLIndNets', 'DRL')
    df['algorithm'] = df['algorithm'].replace('LevyWalks', 'Lévy Walks')
    df['algorithm'] = df['algorithm'].replace('LevyWalksdijkstra', 'Lévy Walks')
    df['algorithm'] = df['algorithm'].replace('Greedydijkstra', 'Greedy')

    # Delete unnecessary columns for analysis
    columns_to_drop = ['date', 'agents_positions', 'actions', 'dones', 'ground_truth', 'model']
    df.drop(columns=columns_to_drop, inplace=True)

    # Add epsilon and epsilon_team to the algorithm name
    df['algorithm'] = df.apply(lambda row: f"Eps{row['epsilon']}_EpsTm{row['epsilon_team']}_{row['algorithm']}" if row['epsilon']!=-1 and row['epsilon_team']!=-1 else row['algorithm'], axis=1)

    # Padding the DataFrame with the last value to ensure all episodes have the same number of steps
    max_steps_per_episode = df['max_steps_per_episode'][0]
    df = df.groupby(['run', 'map_name', 'algorithm', 'objective', 'reward_function', 'reward_weights'])[df.columns].apply(lambda group: group.set_index('step').reindex(range(max_steps_per_episode+1), method='ffill').reset_index(), include_groups=False).reset_index(drop=True)
    # Remove all episodes that last the maximum number of steps 
    # df = df.groupby(['run', 'map_name', 'algorithm', 'objective', 'reward_function', 'reward_weights']).filter(lambda group: (group['step'] < max_steps_per_episode).all()).reset_index(drop=True)
    # Remove all episodes that ended early
    # df = df.groupby(['run', 'map_name', 'algorithm', 'objective', 'reward_function', 'reward_weights']).filter(lambda group: (group['step'] == max_steps_per_episode).any()).reset_index(drop=True)


    # Convert columns with dictionaries to separate columns
    columns_with_dicts = ['rewards', 'reward_components', 'traveled_distances', 'cleaned_trashes', 'history_cleaned_trashes', 'trashes_at_sight', 'coverage_overlap_ratio']
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
    

def calculate_secondary_metrics(df_experiment, show_plots=True, experiment_path=None, save_data = False):
    ''' This part is to analyze the results in depth, including: 
    - From the total discovered by explorers, how many are cleaned?
    - From the total of cleaned, how many have been discovered by explorers?
    - From the total of trash discovered, how many have been discovered by explorers?
    - From the total of trash discovered, how many have been discovered by cleaners?
    - Mean trashes discovered at each step and mean trashes cleaned at each step by each vehicle
    - Time since trash is discovered until it is removed. Violin plot at the end of the experiment to see all trash removed
    - Time since trash is discovered by explorers until it is removed
    - Time since trash is discovered by cleaners until it is removed
    - Mean trashes at sight by each vehicle
    - Time distance from the mean peak of scouts reward to the mean peak of cleaners reward
    - Violin plot between the time step difference between the mean peak of scouts reward and the mean peak of cleaners reward
    Among others...
    '''

    df = df_experiment.copy()
    plot_colors = sns.color_palette('tab10')

    # Extract information from np.full arrays
    explorers_ids = [i for i, team in team_name_of_each_agent.items() if team == 'Explorers' or team == 'Scouts']
    cleaners_ids = [i for i, team in team_name_of_each_agent.items() if team == 'Cleaners' or team == 'Foragers']

    remaining_discovered_by_explorers = df['trash_remaining_info'].apply(lambda x: len([veh for veh in x['vehicle_discover'] if veh in explorers_ids]))
    removed_discovered_by_explorers = df['trash_removed_info'].apply(lambda x: len([veh for veh in x['vehicle_discover'] if veh in explorers_ids]))
    df['acc_discovered_by_explorers'] = remaining_discovered_by_explorers + removed_discovered_by_explorers

    remaining_discovered_by_cleaners = df['trash_remaining_info'].apply(lambda x: len([veh for veh in x['vehicle_discover'] if veh in cleaners_ids]))
    removed_discovered_by_cleaners = df['trash_removed_info'].apply(lambda x: len([veh for veh in x['vehicle_discover'] if veh in cleaners_ids]))
    df['acc_discovered_by_cleaners'] = remaining_discovered_by_cleaners + removed_discovered_by_cleaners

    acc_removed = df['trash_removed_info'].apply(lambda x: len(x))
    remaining_discovered = df['trash_remaining_info'].apply(lambda x: np.sum(x['vehicle_discover'] != -1))
    acc_discovered = remaining_discovered + acc_removed

    df['instant_discovered'] = acc_discovered.diff().fillna(0).apply(lambda x: x if x >= 0 else 0)
    df['instant_removed'] = acc_removed.diff().fillna(0).apply(lambda x: x if x >= 0 else 0)

    # Subplot: Up: instant discovered. Below: instant removed.
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    df[f'instant_discovered_smoothed'] = df.groupby(['algorithm'])[f'instant_discovered'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    sns.lineplot(data=df, x='step', y='instant_discovered_smoothed', hue='algorithm', palette='tab10')
    plt.title(f'Instant Discovered and Removed Trash (smoothed) - {map_name}')
    plt.ylabel('Number of Discovered')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    df[f'instant_removed_smoothed'] = df.groupby(['algorithm'])[f'instant_removed'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    sns.lineplot(data=df, x='step', y='instant_removed_smoothed', hue='algorithm', palette='tab10')
    plt.xlabel('Step')
    plt.ylabel('Number of Removed')
    plt.grid(True)
    plt.tight_layout()
    if save_data:
        plt.savefig(os.path.join(experiment_path, f'm2_instant_discovered_and_removed_{map_name}_smoothed.svg'))
    if show_plots:
        plt.show()
    plt.close()

    # From the total discovered by explorers, how many are cleaned?
    df['percentage_cleaned_of_discovered_by_explorers'] = (removed_discovered_by_explorers / df['acc_discovered_by_explorers'] * 100).fillna(0)

    # From the total of cleaned, how many have been discovered by explorers?
    df['percentage_cleaned_discovered_by_explorers'] = (removed_discovered_by_explorers / acc_removed * 100).fillna(0)

    # From the total of trash discovered, how many have been discovered by explorers?
    df['percentage_discovered_by_explorers_of_total_discovered'] = (df['acc_discovered_by_explorers'] / acc_discovered * 100).fillna(0)

    # From the total of trash discovered, how many have been discovered by cleaners?
    df['percentage_discovered_by_cleaners_of_total_discovered'] = (df['acc_discovered_by_cleaners'] / acc_discovered * 100).fillna(0)

    # Plot the results
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='step', y='percentage_cleaned_of_discovered_by_explorers', hue='algorithm', palette='tab10')
    plt.title(f'From the total discovered by explorers, how many are cleaned? - {map_name}')
    plt.xlabel('Step')
    plt.ylabel('Percentage (%)')
    plt.grid(True)
    if save_data:
        plt.savefig(os.path.join(experiment_path, f'm2_percentage_cleaned_of_discovered_by_explorers_{map_name}.svg'))
    if show_plots:
        plt.show()
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='step', y='percentage_cleaned_discovered_by_explorers', hue='algorithm', palette='tab10')
    plt.title(f'From the total of cleaned, how many have been discovered by explorers? - {map_name}')
    plt.xlabel('Step')
    plt.ylabel('Percentage (%)')
    plt.grid(True)
    if save_data:
        plt.savefig(os.path.join(experiment_path, f'm2_percentage_cleaned_discovered_by_explorers_{map_name}.svg'))
    if show_plots:
        plt.show()
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='step', y='percentage_discovered_by_explorers_of_total_discovered', hue='algorithm', palette='tab10')
    plt.title(f'From the total of trash discovered, how many have been discovered by explorers? - {map_name}')
    plt.xlabel('Step')
    plt.ylabel('Percentage (%)')
    plt.grid(True)
    if save_data:
        plt.savefig(os.path.join(experiment_path, f'm2_percentage_discovered_by_explorers_of_total_discovered_{map_name}.svg'))
    if show_plots:
        plt.show()
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='step', y='percentage_discovered_by_cleaners_of_total_discovered', hue='algorithm', palette='tab10')
    plt.title(f'From the total of trash discovered, how many have been discovered by cleaners? - {map_name}')
    plt.xlabel('Step')
    plt.ylabel('Percentage (%)')
    plt.grid(True)
    if save_data:
        plt.savefig(os.path.join(experiment_path, f'm2_percentage_discovered_by_cleaners_of_total_discovered_{map_name}.svg'))
    if show_plots:
        plt.show()
    plt.close()

    # Mean trashes discovered at each step and mean trashes cleaned at each step by each vehicle
    for idx in explorers_ids:
        # Trash is discovered if the vehicle id is equal to the agent id and the step of discovery is equal to the current step
        df[f'discovered_trashes_{idx}'] = df.apply(
            lambda row: sum(
                veh == idx and step == row['step']
                for veh, step in zip(row['trash_remaining_info']['vehicle_discover'], row['trash_remaining_info']['step_discover'])
            ),
            axis=1
        )
    
    plt.figure(figsize=(12, 6))
    for alg_idx, algorithm in enumerate(df['algorithm'].unique()):
        for idx in explorers_ids:
            if df['algorithm'].unique().size > 1:
                sns.lineplot(data=df[df['algorithm']==algorithm], x='step', y=f'discovered_trashes_{idx}', color=plot_colors[alg_idx], label=f'{algorithm} - Explorer {idx}')
            else:
                sns.lineplot(data=df[df['algorithm']==algorithm], x='step', y=f'discovered_trashes_{idx}', color=plot_colors[idx], label=f'Explorer {idx}')
    plt.title(f'Mean trashes discovered by each explorer - {map_name}')
    plt.xlabel('Step')
    plt.ylabel('Number of trashes')
    plt.grid(True)
    if save_data:
        plt.savefig(os.path.join(experiment_path, f'm2_mean_trashes_discovered_by_each_explorer_{map_name}.svg'))
    if show_plots:
        plt.show()
    plt.close()

    plt.figure(figsize=(12, 6))
    for alg_idx, algorithm in enumerate(df['algorithm'].unique()):
        for idx in cleaners_ids:
            if df['algorithm'].unique().size > 1:
                sns.lineplot(data=df[df['algorithm']==algorithm], x='step', y=f'cleaned_trashes_{idx}', color=plot_colors[alg_idx], label=f'{algorithm} - Cleaner {idx}')
            else:
                sns.lineplot(data=df[df['algorithm']==algorithm], x='step', y=f'cleaned_trashes_{idx}', color=plot_colors[idx], label=f'Cleaner {idx}')
    plt.title(f'Mean trashes cleaned by each cleaner - {map_name}')
    plt.xlabel('Step')
    plt.ylabel('Number of trashes')
    plt.grid(True)
    if save_data:
        plt.savefig(os.path.join(experiment_path, f'm2_mean_trashes_cleaned_by_each_cleaner_{map_name}.svg'))
    if show_plots:
        plt.show()
    plt.close()

    plt.figure(figsize=(12, 6))
    for alg_idx, algorithm in enumerate(df['algorithm'].unique()):
        for idx in explorers_ids:
            if df['algorithm'].unique().size > 1:
                sns.lineplot(data=df[df['algorithm']==algorithm], x='step', y=f'trashes_at_sight_{idx}', color=plot_colors[alg_idx], label=f'{algorithm} - Explorer {idx}')
            else:
                sns.lineplot(data=df[df['algorithm']==algorithm], x='step', y=f'trashes_at_sight_{idx}', color=plot_colors[idx], label=f'Explorer {idx}')
    plt.title(f'Mean trashes at sight by each explorer - {map_name}')
    plt.xlabel('Step')
    plt.ylabel('Number of trashes')
    plt.grid(True)
    if save_data:
        plt.savefig(os.path.join(experiment_path, f'm2_mean_trashes_at_sight_by_each_explorer_{map_name}.svg'))
    if show_plots:
        plt.show()
    plt.close()

    # Time since trash is discovered until it is removed. Violin plot at the end of the experiment to see all trash removed
    df['time_from_discovery_to_removal'] = df['trash_removed_info'].apply(lambda x: x['step_remove'] - x['step_discover'])
    df_exploded = df[df['step'] == 150][['algorithm', 'time_from_discovery_to_removal']].explode('time_from_discovery_to_removal').reset_index(drop=True)

    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df_exploded, x='algorithm', y='time_from_discovery_to_removal', hue='algorithm', palette='tab10')
    plt.title(f'Time from discovery to removal of trash - {map_name}')
    plt.xlabel('Algorithm')
    plt.ylabel('Time (steps)')
    plt.xticks(rotation=45)
    plt.grid(True)
    if save_data:
        plt.savefig(os.path.join(experiment_path, f'm2_time_from_discovery_to_removal_{map_name}.svg'))
    if show_plots:
        plt.show()
    plt.close()

    print("Average time steps from discovery to removal of trash:")
    print(df_exploded.groupby('algorithm')['time_from_discovery_to_removal'].mean())

    # Time since trash is discovered by explorers until it is removed
    df['time_from_explorers_discovery_to_removal'] = df['trash_removed_info'].apply(lambda x: [x['step_remove'][index] - x['step_discover'][index] for index in range(len(x)) if x['vehicle_discover'][index] in explorers_ids])
    df_exploded_explorers = df[df['step'] == 150][['algorithm', 'time_from_explorers_discovery_to_removal']].explode('time_from_explorers_discovery_to_removal').reset_index(drop=True)
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df_exploded_explorers, x='algorithm', y='time_from_explorers_discovery_to_removal', hue='algorithm', palette='tab10')
    plt.title(f'Time from SCOUTS discovery to removal of trash - {map_name}')
    plt.xlabel('Algorithm')
    plt.ylabel('Time (steps)')
    plt.xticks(rotation=45)
    plt.grid(True)
    if save_data:
        plt.savefig(os.path.join(experiment_path, f'm2_time_from_explorers_discovery_to_removal_{map_name}.svg'))
    if show_plots:
        plt.show()
    plt.close()


    # Reward evolution of scouts and cleaners
    df['reward_scouts'] = 0
    df['reward_cleaners'] = 0
    for idx in explorers_ids:
        df['reward_scouts'] += df[f'rewards_{idx}']
    for idx in cleaners_ids:
        df['reward_cleaners'] += df[f'rewards_{idx}']
    plt.figure(figsize=(12, 6))
    df[f'reward_scouts_smoothed'] = df.groupby(['algorithm'])[f'reward_scouts'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    sns.lineplot(data=df, x='step', y='reward_scouts_smoothed', hue='algorithm', palette='tab10')
    plt.title(f'Reward of SCOUTS - {map_name}')
    plt.xlabel('Algorithm')
    plt.ylabel('Reward')
    plt.grid(True)
    if save_data:
        plt.savefig(os.path.join(experiment_path, f'm2_reward_scouts_smoothed_{map_name}.svg'))
    if show_plots:
        plt.show()
    plt.close()
    plt.figure(figsize=(12, 6))
    df[f'reward_cleaners_smoothed'] = df.groupby(['algorithm'])[f'reward_cleaners'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    sns.lineplot(data=df, x='step', y='reward_cleaners_smoothed', hue='algorithm', palette='tab10')
    plt.title(f'Reward of CLEANERS - {map_name}')
    plt.xlabel('Algorithm')
    plt.ylabel('Reward')
    plt.grid(True)
    if save_data:
        plt.savefig(os.path.join(experiment_path, f'm2_reward_cleaners_smoothed_{map_name}.svg'))
    if show_plots:
        plt.show()
    plt.close()


    # Time step difference from the peaks of scouts rewards to the peaks of cleaners rewards, presented as a violin plot
    step_peak_scouts = df.groupby(['algorithm', 'run'])['reward_scouts'].idxmax()
    step_peak_cleaners = df.groupby(['algorithm', 'run'])['reward_cleaners'].idxmax()
    peaks_diff = step_peak_cleaners - step_peak_scouts
    df_peaks_diff = peaks_diff.reset_index(name='value')

    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df_peaks_diff, x='algorithm', y='value', hue='algorithm', palette='tab10')
    plt.title(f'Time step difference between the mean peak of SCOUTS reward and the mean peak of CLEANERS reward - {map_name}')
    plt.xlabel('Algorithm')
    plt.ylabel('Time step difference')
    plt.xticks(rotation=45)
    plt.grid(True)
    if save_data:
        plt.savefig(os.path.join(experiment_path, f'm2_time_step_difference_peak_scouts_cleaners_{map_name}.svg'))
    if show_plots:
        plt.show()
    plt.close()

    print("Finished calculating secondary metrics.")

def derived_accumulated_rewards(df_processed, show_plots=False, experiment_path=None, save_data = False):

    plot_colors = sns.color_palette('tab10')
    df = df_processed.copy()

    ## Calculate the accumulated rewards for each team ##
    for agent_id in range(df['n_agents'][0]):
        df[f'acc_rewards_{agent_id}'] = df.groupby(['run', 'map_name', 'algorithm', 'objective', 'reward_function', 'reward_weights'])[f'rewards_{agent_id}'].cumsum()
    team_names = set(team_name_of_each_agent.values())
    for team in team_names:
        agent_ids_in_the_team = [agent_id for agent_id, team_name in team_name_of_each_agent.items() if team_name == team]
        df[f'acc_rewards_team_{team}'] = df[[f'acc_rewards_{agent_id}' for agent_id in agent_ids_in_the_team]].sum(axis=1)

    ## Calculate the accumulated individual components of the rewards for each team ##
    reward_components = set([col.split('.')[-1] for col in df.columns if col.startswith('reward_components_')])
    # Get accumulated reward components for each agent
    for component in reward_components:
        for agent_id in range(df['n_agents'][0]):
            # Check if the component exists for the agent
            if f'reward_components_{agent_id}.{component}' in df.columns:
                # Check if there is more than two unique values (to avoid constant columns)
                if df[f'reward_components_{agent_id}.{component}'].nunique() > 1:
                    df[f'acc_reward_components_{agent_id}.{component}'] = df.groupby(['run', 'map_name', 'algorithm', 'objective', 'reward_function', 'reward_weights'])[f'reward_components_{agent_id}.{component}'].cumsum()
                else:
                    # If the component is constant, delete the column
                    df.drop(columns=[f'reward_components_{agent_id}.{component}'], inplace=True)
    # Get accumulated reward components for each team
    for team in team_names:
        agent_ids_in_the_team = [agent_id for agent_id, team_name in team_name_of_each_agent.items() if team_name == team]
        for component in reward_components:
            # Check if at least one agent in the team has this component
            if any(f'acc_reward_components_{agent_id}.{component}' in df.columns for agent_id in agent_ids_in_the_team):
                df[f'acc_reward_components_team_{team}.{component}'] = df[[f'acc_reward_components_{agent_id}.{component}' for agent_id in agent_ids_in_the_team if f'acc_reward_components_{agent_id}.{component}' in df.columns]].sum(axis=1)

    # Plot the accumulated rewards for each team
    # for team in team_names:
    #     plt.figure(figsize=(10, 6))
    #     sns.lineplot(data=df, x='step', y=f'acc_rewards_team_{team}', hue='algorithm', palette='tab10')
    #     plt.title(f'Accumulated Rewards for Team {team} - {map_name}')
    #     plt.xlabel('Step')
    #     plt.ylabel('Accumulated Rewards')
    #     plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    #     plt.grid(True)
    #     plt.tight_layout()
    #     if save_data:
    #         path_fig = os.path.join(experiment_path, f'accumulated_rewards_team_{team}_{map_name}.svg')
    #         plt.savefig(path_fig)
    #     if show_plots:
    #         plt.show()
    #     plt.close()

    # Plot the accumulated reward for each team on top and subplots of each component below
    team_reward_components_names = {team_name: [col.split('.')[-1] for col in df.columns if col.startswith(f'acc_reward_components_team_{team_name}.')] for team_name in team_names}
    for team in team_names:
        components_names = team_reward_components_names[team]
        n_components = len(components_names)
        # fig, axs = plt.subplots(n_components + 1, 1, figsize=(10, 2 * (n_components + 1)), sharex=True)
        fig, axs = plt.subplots(n_components + 1, 1, figsize=(10, 8), sharex=True)
        # Plot accumulated rewards for the team
        palette_algorithms=[tuple(np.clip(np.array(plot_colors[0]) + 0.15*i, 0, 1)) for i in range(df['algorithm'].nunique())]
        sns.lineplot(data=df, x='step', y=f'acc_rewards_team_{team}', hue='algorithm', palette=palette_algorithms, ax=axs[0])
        axs[0].set_title(f'Accumulated Rewards for Team {team} - {map_name}')
        axs[0].set_ylabel('Accumulated Rewards')
        # axs[0].legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        axs[0].legend()
        axs[0].grid(True)
        # Plot each reward component
        for i, component in enumerate(components_names):
            palette_algorithms=[tuple(np.clip(np.array(plot_colors[i + 1]) + 0.15*j, 0, 1)) for j in range(df['algorithm'].nunique())]
            sns.lineplot(data=df, x='step', y=f'acc_reward_components_team_{team}.{component}', hue='algorithm', palette=palette_algorithms, ax=axs[i + 1])
            axs[i + 1].set_title(f'Accumulated Reward Component "{component}" for Team {team} - {map_name}')
            # axs[i + 1].set_ylabel(f'Accumulated {component}')
            axs[i + 1].set_ylabel(f'Reward units')
            # axs[i + 1].legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
            axs[i + 1].legend()
            axs[i + 1].grid(True)
        plt.xlabel('Step')
        plt.tight_layout()
        if save_data:
            path_fig = os.path.join(experiment_path, f'accumulated_rewards_and_components_{team}_{map_name}.svg')
            plt.savefig(path_fig)
        if show_plots:
            plt.show()
        plt.close()
    
    # Average accumulated rewards for each team at each step, and for each component
    for algorithm in df['algorithm'].unique():
        df_algo = df[df['algorithm'] == algorithm]
        avg_acc_rewards = pd.DataFrame()
        for team in team_names:
            avg_acc_rewards[team] = df_algo.groupby(['step'])[f'acc_rewards_team_{team}'].mean().reset_index()[f'acc_rewards_team_{team}']
            for component in team_reward_components_names[team]:
                avg_acc_rewards[f'{team}.{component}'] = df_algo.groupby(['step', 'algorithm'])[f'acc_reward_components_team_{team}.{component}'].mean().reset_index()[f'acc_reward_components_team_{team}.{component}']

        ## Plot the derived of average accumulated rewards for each team and each component ##
        for team in team_names:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=avg_acc_rewards[team].diff().fillna(0), label=f'Total reward')
            for component in team_reward_components_names[team]:
                sns.lineplot(data=avg_acc_rewards[f'{team}.{component}'].diff().fillna(0), label=f'Reward Component "{component}"')
            plt.title(f'Derived of Average Accumulated Rewards and Components - Alg. {algorithm} - Team {team} - {map_name}')
            plt.xlabel('Step')
            plt.ylabel('Derived of Average Accumulated Rewards')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            if save_data:
                path_fig = os.path.join(experiment_path, f'derived_of_accumulated_rewards_and_components_{algorithm}_{team}_{map_name}.svg')
                plt.savefig(path_fig)
            if show_plots:
                plt.show()
            plt.close()

    # Plot the same but in subplots
    for team in team_names:
        components_names = team_reward_components_names[team]
        n_components = len(components_names)
        fig, axs = plt.subplots(n_components + 1, 1, figsize=(10, 8), sharex=True)
        # Plot derived of accumulated rewards for the team
        sns.lineplot(data=avg_acc_rewards[team].diff().fillna(0), label=f'Total reward', color=plot_colors[0], ax=axs[0])
        axs[0].set_title(f'Derived of Average Accumulated Rewards - Team {team} - {map_name}')
        axs[0].set_ylabel('Rw Units')
        axs[0].legend()
        axs[0].grid(True)
        # Plot each reward component
        for i, component in enumerate(components_names):
            sns.lineplot(data=avg_acc_rewards[f'{team}.{component}'].diff().fillna(0), label=f'Reward Component "{component}"', color=plot_colors[i + 1], ax=axs[i + 1])
            # axs[i + 1].set_title(f'Derived of Average Accumulated Reward Component "{component}" - Team {team} - {map_name}')
            axs[i + 1].set_ylabel('Rw Units')
            axs[i + 1].legend()
            axs[i + 1].grid(True)
        plt.xlabel('Step')
        plt.tight_layout()
        if save_data:
            path_fig = os.path.join(experiment_path, f'derived_of_accumulated_rewards_and_components_subplots_{team}_{map_name}.svg')
            plt.savefig(path_fig)
        if show_plots:
            plt.show()
        plt.close()

    # # Derived of average accumulated rewards for each team
    # derived_accumulated_rewards_explorers = avg_acc_rewards['Explorers'].diff().fillna(0)
    # derived_accumulated_rewards_cleaners = avg_acc_rewards['Cleaners'].diff().fillna(0)

    # # Plot the derived of average accumulated rewards for each team
    # plt.figure(figsize=(10, 6))
    # sns.lineplot(data=derived_accumulated_rewards_explorers, label=f'Derived of Average Accumulated Rewards - Team Explorers', color='blue')
    # sns.lineplot(data=derived_accumulated_rewards_cleaners, label=f'Derived of Average Accumulated Rewards - Team Cleaners', color='orange')
    # plt.title(f'Derived of Average Accumulated Rewards - {map_name}')
    # plt.xlabel('Step')
    # plt.ylabel('Derived of Average Accumulated Rewards')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # if show_plots:
    #     plt.show()
    # plt.close()

    print("Derived accumulated rewards analysis completed.")

def mean_instant_rewards(df_processed, show_plots=False, experiment_path=None, save_data = False):

    plot_colors = sns.color_palette('tab10')
    df = df_processed.copy()

    ## Calculate the mean instant rewards for each team ##
    team_names = set(team_name_of_each_agent.values())
    for team in team_names:
        agent_ids_in_the_team = [agent_id for agent_id, team_name in team_name_of_each_agent.items() if team_name == team]
        df[f'rewards_team_{team}'] = df[[f'rewards_{agent_id}' for agent_id in agent_ids_in_the_team]].sum(axis=1)

    # Plot the mean instant rewards for each team
    # for team in team_names:
    #     plt.figure(figsize=(10, 6))
    #     sns.lineplot(data=df, x='step', y=f'rewards_team_{team}', hue='algorithm', palette='tab10')
    #     plt.title(f'Instant Rewards for Team {team} - {map_name}')
    #     plt.xlabel('Step')
    #     plt.ylabel('Rw Units')
    #     plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    #     plt.grid(True)
    #     plt.tight_layout()
    #     if save_data:
    #         path_fig = os.path.join(experiment_path, f'mean_instant_r_{team}_{map_name}.svg')
    #         plt.savefig(path_fig)
    #     if show_plots:
    #         plt.show()
    #     plt.close()

    # # Calculate the mean instant components of the rewards for each team ##
    reward_components = set([col.split('.')[-1] for col in df.columns if col.startswith('reward_components_')])
    # Get instant reward components for each team
    for team in team_names:
        agent_ids_in_the_team = [agent_id for agent_id, team_name in team_name_of_each_agent.items() if team_name == team]
        for component in reward_components:
            df[f'reward_components_team_{team}.{component}'] = df[[f'reward_components_{agent_id}.{component}' for agent_id in agent_ids_in_the_team]].sum(axis=1)
            #Check if there is more than one unique values (to avoid constant columns)
            if df[f'reward_components_team_{team}.{component}'].nunique() <= 1:
                df.drop(columns=[f'reward_components_team_{team}.{component}'], inplace=True)  

    # Plot the instant reward for each team on top and subplots of each component below
    team_reward_components_names = {team_name: [col.split('.')[-1] for col in df.columns if col.startswith(f'reward_components_team_{team_name}.')] for team_name in team_names}
    for team in team_names:
        components_names = team_reward_components_names[team]
        n_components = len(components_names)
        fig, axs = plt.subplots(n_components + 1, 1, figsize=(10, 8), sharex=True)
        # Smooth the lines using rolling mean
        df[f'rewards_team_{team}_smoothed'] = df.groupby(['algorithm'])[f'rewards_team_{team}'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
        # Plot instant rewards for the team
        palette_algorithms=[tuple(np.clip(np.array(plot_colors[0]) + 0.15*i, 0, 1)) for i in range(df['algorithm'].nunique())]
        sns.lineplot(data=df, x='step', y=f'rewards_team_{team}_smoothed', hue='algorithm', palette=palette_algorithms, ax=axs[0])
        axs[0].set_title(f'Instant Rewards for Team {team} (Avg. {df["run"].nunique()} episodes) - {map_name}')
        axs[0].set_ylabel('Instant Rewards')
        # axs[0].legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        axs[0].legend()
        axs[0].grid(True)
        # Plot each reward component
        for i, component in enumerate(components_names):
            # Smooth the lines using rolling mean
            df[f'reward_components_team_{team}.{component}_smoothed'] = df.groupby(['algorithm'])[f'reward_components_team_{team}.{component}'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
            palette_algorithms = [tuple(np.clip(np.array(plot_colors[i + 1]) + 0.15*j, 0, 1)) for j in range(df['algorithm'].nunique())]
            sns.lineplot(data=df, x='step', y=f'reward_components_team_{team}.{component}_smoothed', hue='algorithm', palette=palette_algorithms, ax=axs[i + 1])
            axs[i + 1].set_title(f'Instant Reward Component "{component}" for Team {team} - {map_name}')
            axs[i + 1].set_ylabel(f'Reward units')
            # axs[i + 1].legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
            axs[i + 1].legend()
            axs[i + 1].grid(True)
        plt.xlabel('Step')
        plt.tight_layout()
        if save_data:
            path_fig = os.path.join(experiment_path, f'mean_instant_r_components_{team}_smoothed_{map_name}.svg')
            plt.savefig(path_fig)
        if show_plots:
            plt.show()
        plt.close()

    print("Mean instant rewards analysis completed.")

def gini_coefficient(x):
    x = np.array(x)
    return np.sum(np.abs(np.subtract.outer(x, x)))/(2*len(x)**2*x.mean())

def gini_study(df_processed, show_plots=False, experiment_path=None, save_data = False):
    ''' Study the evolution of Gini coefficient over time (steps) for several metrics:
    - Trash collected by cleaners
    - Accumulated rewards of explorers -> Gini assumes not negative values, so this metric is not valid if rewards can be negative!
    -----------
    The Gini coefficient is a measure of statistical dispersion that represents the inequality among values of a distribution.'''
    
    df = df_processed.copy()
    max_steps_per_episode = df['max_steps_per_episode'].iloc[0]
    
    ## Evolution of Gini coefficient for trash collected by cleaners over time ##
    # Identify cleaner agents
    cleaners_ids = [i for i, team in team_name_of_each_agent.items() if team == 'Cleaners' or team == 'Foragers']
   
    # Calculate Gini coefficient for each step
    gini_per_step = []
    for step in range(1, max_steps_per_episode + 1):
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
                            'step': step / max_steps_per_episode,
                            'run': run,
                            'algorithm': alg,
                            'gini_coefficient': gini
                        })
    
    # Create DataFrame with Gini coefficients over time
    df_gini_evolution = pd.DataFrame(gini_per_step)

    # Visualize evolution of Gini coefficient over time
    if not df_gini_evolution.empty:
        plt.figure(figsize=(4, 2))
        sns.lineplot(data=df_gini_evolution, x='step', y='gini_coefficient', hue='algorithm', palette='tab10')
        plt.title(f'Evolution of Gini Coefficient for Items Collected', fontsize=9)
        plt.xlabel('Normalized Time', fontsize=8)
        plt.ylabel('Gini Coefficient', fontsize=8)
        plt.legend(fontsize=7)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.grid(True, linewidth=0.3)
        plt.tight_layout()
        if save_data:
            plt.savefig(os.path.join(experiment_path, f'C.a_gini_coefficient_items_collection_{map_name}.pdf'))
        if show_plots:
            plt.show()
        plt.close()

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
    #     plt.close()

def return_normalized_time_to_percentage(df_processed, percentage):
    ''' Calculate the time step at which a certain percentage of the initial trash is collected '''
    df = df_processed.copy()
    results = []
    for alg in df['algorithm'].unique():
        df_alg = df[df['algorithm'] == alg]
        for run in df_alg['run'].unique():
            df_run = df_alg[df_alg['run'] == run]
            initial_trash = df_run['initial_trash'].iloc[0]
            target_trash = initial_trash * (1 - percentage / 100)
            time_step = df_run[df_run['trash_remaining'] <= target_trash]['step']
            if not time_step.empty:
                time_to_percentage = time_step.iloc[0]
            else:
                time_to_percentage = df_run['step'].max()
            results.append({
                'algorithm': alg,
                'run': run,
                f'normalized_time_to_{percentage}': time_to_percentage/df_run['max_steps_per_episode'].iloc[0]
            })
    df_results = pd.DataFrame(results)
    print(f"\nTime steps to reach {percentage}% of trash collected:")
    print(df_results.groupby('algorithm')[f'normalized_time_to_{percentage}'].mean())
    return df_results

def return_troughput(df_processed):
    ''' Calculate the throughput: average trash collected per time step '''
    df = df_processed.copy()
    results = []
    for alg in df['algorithm'].unique():
        df_alg = df[df['algorithm'] == alg]
        n_cleaners = sum(1 for team in team_name_of_each_agent.values() if team == 'Cleaners' or team == 'Foragers')
        for run in df_alg['run'].unique():
            df_run = df_alg[df_alg['run'] == run]
            initial_trash = df_run['initial_trash'].iloc[0]
            number_steps_episode = df_run['step'].max()
            trash_collected = initial_trash - df_run['trash_remaining'].iloc[-1]
            throughput = trash_collected / (n_cleaners * number_steps_episode)
            results.append({
                'algorithm': alg,
                'run': run,
                'throughput': throughput
            })
    df_results = pd.DataFrame(results)
    print(f"\nThroughput (average trash collected per time step and cleaner agent):")
    print(df_results.groupby('algorithm')['throughput'].mean())
    return df_results

def plot_idleness_reduction_rate(df_processed, show_plots=False, experiment_path=None, save_data=False):
    ''' Calculate the idleness reduction rate over time '''
    df = df_processed.copy()
    df['normalized_step'] = df['step'] / df['max_steps_per_episode']
    
    # Calculate idleness map mean at each step
    n_visitable_nodes = np.sum(scenario_map > 0)
    df["idleness_mean"] = df["idleness_map"].apply(lambda map: map.sum() / n_visitable_nodes)


    # Calculate idleness reduction rate
    df["idleness_reduction_rate"] = df.groupby(["algorithm", "run"])["idleness_mean"].diff() * -1

    # Dual plot, idleness mean and idleness reduction rate
    fig, axes = plt.subplots(2, 1, figsize=(4, 3), sharex=True)  # altura mayor para dos subplots

    # Subplot 1: Idleness Mean
    sns.lineplot(
        data=df, x="normalized_step", y="idleness_mean",
        hue="algorithm", estimator="mean", ax=axes[0], linewidth=1.6
    )
    axes[0].set_title(f"Idleness Metrics", fontsize=9)
    # axes[0].set_ylabel("Idleness \nMean", fontsize=8)
    axes[0].set_ylabel("MI", fontsize=8)
    # axes[0].set_ylabel("I(t)", fontsize=8)
    axes[0].grid(True, linewidth=0.3)
    axes[0].tick_params(axis='both', labelsize=7)
    axes[0].legend(fontsize=7, title=None)

    # Subplot 2: Idleness Reduction Rate
    sns.lineplot(
        data=df, x="normalized_step", y="idleness_reduction_rate",
        hue="algorithm", estimator="mean", ax=axes[1], linewidth=1.6,
    )
    # axes[1].set_ylabel("Idleness \nReduction Rate", fontsize=8)
    axes[1].set_ylabel("IRR", fontsize=8)
    # axes[1].set_ylabel("IR Rate (t)", fontsize=8)
    axes[1].set_xlabel("Normalized Time", fontsize=8)
    axes[1].grid(True, linewidth=0.3)
    axes[1].tick_params(axis='both', labelsize=7)
    axes[1].legend(fontsize=7, title=None)

    plt.tight_layout()
    if save_data:
        plt.savefig(os.path.join(experiment_path, f'A.e_idleness_reduction_rate_{map_name}.pdf'), bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()
    plt.close()

def return_expected_service_cost_reduction(df_processed):
    ''' Evolution of expected service cost reduction over time '''
    df = df_processed.copy()
    for alg in df['algorithm'].unique():
        df_alg = df[df['algorithm'] == alg]
        pass

def plot_coop_sens_stochastic_corruption(epsilon_study_paths, show_plots=False, save_data=False):
    ''' Plot comparison of different of the performance of different algorithms with varying epsilon values '''
    metrics = ['rmse', 'ptc', 'ptd']
    teams_name = {0: 'Scout', 1: 'Forager'}
    main_folder = os.path.dirname(epsilon_study_paths[0])

    df_rmse = pd.DataFrame()
    df_pta_foragers = pd.DataFrame()
    df_pta_scouts = pd.DataFrame()
    for path in epsilon_study_paths:
        df_rmse = pd.concat([df_rmse, pd.read_csv(os.path.join(path, 'epsilon_study_RMSE_EpsTmForagers_comb_port_exploded.csv'))], ignore_index=True)
        df_pta_foragers = pd.concat([df_pta_foragers, pd.read_csv(os.path.join(path, 'epsilon_study_ptaForagers_EpsTmScouts_comb_port_exploded.csv'))], ignore_index=True)
        df_pta_scouts = pd.concat([df_pta_scouts, pd.read_csv(os.path.join(path, 'epsilon_study_ptaScouts_EpsTmForagers_comb_port_exploded.csv'))], ignore_index=True)


    for i, metric in enumerate(metrics):
        if metric == 'rmse':
            label = 'Root Mean Squared Error'
            epsilon_team = 1
            metric_name = 'RMSE'
            w = 1
            df_exploded = df_rmse.copy()
        elif metric == 'ptc':
            label = 'Percentage of Target Achieved (Collection)'
            epsilon_team = 0
            metric_name = 'ptaForagers'
            w = 0.01
            df_exploded = df_pta_foragers.copy()
        elif metric == 'ptd':
            label = 'Percentage of Target Achieved (Discovery)'
            epsilon_team = 1
            metric_name = 'ptaScouts'
            w = 0.01
            df_exploded = df_pta_scouts.copy()

        df_exploded['algorithm_name'] = df_exploded['algorithm'].apply(lambda x: x.split('_')[-1])
        df_exploded['algorithm_name'] = df_exploded['algorithm_name'].replace('DRLIndNets', 'DRL')
        df_exploded['algorithm_name'] = df_exploded['algorithm_name'].replace('LevyWalks', 'Lévy Walks')
        df_exploded['algorithm_name'] = df_exploded['algorithm_name'].replace('LevyWalksdijkstra', 'Lévy Walks')
        df_exploded['algorithm_name'] = df_exploded['algorithm_name'].replace('Greedydijkstra', 'Greedy')

        # Map categorical epsilon values to numerical positions
        eps_values = sorted(df_exploded["epsilon"].unique())
        positions = range(len(eps_values))
        pos_map = {e: i for i, e in enumerate(eps_values)}
        df_exploded["eps_pos"] = df_exploded["epsilon"].map(pos_map)

        tab10_colors = sns.color_palette("tab10")
        line_styles = ['dashed', 'dotted', 'dashdot']
        markers = ['o', 's', 'D']

        plt.figure(figsize=(5, 3))
        sns.pointplot(data=df_exploded, x='eps_pos', y=f'{metric}_value', hue='algorithm_name', errorbar=('ci', 95),  
                    markers=markers, markersize=0, capsize=0.1, palette='tab10', linestyles="", linewidth=1, alpha=0.3, legend=False)
        sns.pointplot(data=df_exploded, x='eps_pos', y=f'{metric}_value', hue='algorithm_name', errorbar=None,  
                    markers=markers, markersize=3, capsize=0.3, palette='tab10', linestyles="", linewidth=0.8)
        
        for i, algorithm_name in enumerate(df_exploded['algorithm_name'].unique()):
            df_algo = df_exploded[df_exploded['algorithm_name'] == algorithm_name]
            x = df_algo['epsilon']
            y = df_algo[f'avg_{metric}_100%'] * w
            slope, intercept, r_value = get_slope(x=x, y=y)
            x_line = np.array(list(positions))
            y_line = slope * np.array(eps_values) + intercept
            plt.plot(x_line, y_line, linestyle=line_styles[i], linewidth=1.6, color=tab10_colors[i], alpha=0.9)#, label='Linear fit')

        plt.title(f'CSSC in {teams_name[epsilon_team]} Team', fontsize=9)
        plt.xlabel('Epsilon', fontsize=8)
        plt.ylabel(f'{label}', fontsize=8)
        plt.legend(fontsize=7)
        plt.grid(True, linewidth=0.3)
        plt.tight_layout()
        plt.xticks(positions[::2], eps_values[::2], fontsize=7)
        plt.yticks(fontsize=7)
        if save_data:
            path_fig = os.path.join(main_folder, f'B.e_epsilon_study_{metric_name}_EpsTm{teams_name[epsilon_team]}_{map_name}_pointplot.svg')
            plt.savefig(path_fig)
            path_fig = os.path.join(main_folder, f'B.e_epsilon_study_{metric_name}_EpsTm{teams_name[epsilon_team]}_{map_name}_pointplot.pdf')
            plt.savefig(path_fig)
        if show_plots:
            plt.show()
        plt.close()

def plot_coverage_overlap(df_processed, show_plots=False, experiment_path=None, save_data=False):
    ''' Plot coverage and overlap metrics over time '''
    df = df_processed.copy()

    # Smooth the lines using rolling mean
    team_id_scouts = 0
    # df['overlap_smoothed'] = df.groupby(['algorithm'])[f'coverage_overlap_ratio_{team_id_scouts}'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    df['normalized_step'] = df['step'] / df['max_steps_per_episode']

    plt.figure(figsize=(4, 2))
    # sns.lineplot(data=df, x="normalized_step", y="overlap_smoothed", hue="algorithm", linewidth=1.6, palette='tab10')
    sns.lineplot(data=df, x="normalized_step", y=f"coverage_overlap_ratio_{team_id_scouts}", hue="algorithm", linewidth=1.6, palette='tab10')
    plt.title(f'Coverage Overlap Scouts Team', fontsize=9)
    plt.xlabel('Normalized Time', fontsize=8)
    plt.ylabel("Coverage Overlap", fontsize=8)
    plt.legend(fontsize=7, title=None)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    if save_data:
        plt.savefig(os.path.join(experiment_path, f'C.b_coverage_overlap_{map_name}.pdf'), bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()
    plt.close()

if __name__ == '__main__':
    
    experiment_name = 'evaluation'
    # experiment_name = 'merged_experiment'

    # experiment_path = 'Evaluation/Results/EPSILON_TEST_comb_port.4.timenegativelogdijkstra_2.30..._SEED=3/Eps_0.0_EpsTm_0_comb_port.4.timenegativelogdijkstra_2.30_2.19_1.86_4.10'
    # experiment_path = 'Evaluation/Results/EPSILON_TEST_comb_port.4.timenegativelogdijkstra_2.30..._SEED=3/Epsmerged_0_0.5_1_Tm0'
    # experiment_path = 'Evaluation/Results/EPSILON_TEST_comb_port.4.timenegativelogdijkstra_2.30..._SEED=3/Epsmerged_0_0.5_1_Tm1'
    # experiment_path = 'Evaluation/Results/EPSILON_TEST_comb_port.4.timenegativelogdijkstra_2.30..._SEED=4/Eps_0.0_EpsTm_0_comb_port.4.timenegativelogdijkstra_2.30_2.19_1.86_4.10'
    # experiment_path = 'Evaluation/Results/EPSILON_TEST_comb_port.4.timenegativelogdijkstra_2.30..._SEED=3_EPSEED=4/Eps_0.0_EpsTm_0_comb_port.4.timenegativelogdijkstra_2.30_2.19_1.86_4.10'
    # experiment_path = 'Evaluation/Results/EPSILON_TEST_greedydijkstra_2.30..._SEED=3/Eps_0.0_EpsTm_0_comb_port.4.negativedijkstra_2.3_2.19_1.86_4.1'
    # experiment_path = 'Evaluation/Results/EPSILON_TEST_PSO_2.30..._SEED=3/Eps_0.0_EpsTm_0_comb_port.4.negativedistance_2.3_2.19_1.86_4.1'
    # experiment_path = 'Evaluation/Results/EPSILON_TEST_PSOdijkstra_2.30..._SEED=3/Eps_0.0_EpsTm_0_comb_port.4.negativedijkstra_2.3_2.19_1.86_4.1'
    # experiment_path = 'Evaluation/Results/EPSILON_TEST_levywalksdijkstra_2.30..._SEED=3/Eps_0.0_EpsTm_0_comb_port.4.negativedijkstra_2.3_2.19_1.86_4.1'
    experiment_path = 'Evaluation/Results/FINAL_ALGORITHMS_comb_port.4.negativedijkstra_2.12_5.91_2.23_1.7'
    # experiment_path = 'Evaluation/Results/FINAL_ALGORITHMS_comb_port.2s1f.negativedijkstra_2.12_5.91_2.23_1.7'
    # experiment_path = 'Evaluation/Results/FINAL_ALGORITHMS_comb_port.1s2f.negativedijkstra_2.12_5.91_2.23_1.7'


    # Load the experiment data #
    df_experiment = load_experiment_pd(experiment_name, experiment_path)
    map_name = df_experiment['map_name'].unique()[0]
    scenario_map = np.genfromtxt(f'Environment/Maps/{map_name}.csv', delimiter=',')
    team_id_of_each_agent = df_experiment['team_id_of_each_agent'][0]
    teams_name = df_experiment['teams_name'][0]
    # If 'Explorers' and 'Cleaners' are in teams_name, rename them to 'Scouts' and 'Foragers'
    for team_id, team_name in teams_name.items():
        if team_name == 'Explorers':
            teams_name[team_id] = 'Scouts'
        elif team_name == 'Cleaners':
            teams_name[team_id] = 'Foragers'
    df_experiment = df_experiment.drop(columns=['team_id_of_each_agent', 'teams_name'])
    team_name_of_each_agent = {i: teams_name[team_id_of_each_agent[i]] for i in range(len(team_id_of_each_agent))}
    df_experiment['initial_trash'] = df_experiment.groupby('run')['trash_remaining'].transform('max')

    # Process the DataFrame to prepare for analysis #
    df_processed = process_dataframe(df_experiment)

    ''' Calls to analysis functions '''
    if False:
        calculate_average_metrics(df_processed, show_plots=False, experiment_path=experiment_path, save_data=True)
        # derived_accumulated_rewards(df_processed, show_plots=False, experiment_path=experiment_path, save_data=True)
        # mean_instant_rewards(df_processed, show_plots=False, experiment_path=experiment_path, save_data=True)
        # calculate_secondary_metrics(df_processed, show_plots=False, experiment_path=experiment_path, save_data=True)
        # gini_study(df_processed, show_plots=False, experiment_path=experiment_path, save_data=True)

    # A.a. Global Metrics
    if False:
        calculate_global_metrics(df_processed, show_plots=False, experiment_path=experiment_path, save_data=True)

    # B.b. Inter-team Metrics
    if False:
        calculate_inter_team_metrics(df_processed, show_plots=False, experiment_path=experiment_path, save_data=True)

    # C.c - Intra-team Metrics
    if False:
        calculate_intra_team_metrics(df_processed, show_plots=False, experiment_path=experiment_path, save_data=True)


    ''' THIS IS INDEPENDENT OF THE ABOVE EXPERIMENT '''
    ''' Epsilon study '''
    # epsilon_study_path = 'Evaluation/Results/EPSILON_TEST_comb_port.4.timenegativelogdijkstra_2.30..._SEED=3'
    # epsilon_study_path = 'Evaluation/Results/EPSILON_TEST_comb_port.4.timenegativelogdijkstra_2.30..._SEED=4'
    # epsilon_study_path = 'Evaluation/Results/EPSILON_TEST_comb_port.4.timenegativelogdijkstra_2.30..._SEED=3_EPSEED=4'
    # epsilon_study_path = 'Evaluation/Results/EPSILON_TEST_greedydijkstra_2.30..._SEED=3'
    # epsilon_study_path = 'Evaluation/Results/EPSILON_TEST_PSO_2.30..._SEED=3'
    # epsilon_study_path = 'Evaluation/Results/EPSILON_TEST_PSOdijkstra_2.30..._SEED=3'
    # epsilon_study_path = 'Evaluation/Results/EPSILON_TEST_levywalksdijkstra_2.30..._SEED=3'
    # Final paths:
    # epsilon_study_path = 'Evaluation/Results/FINAL_ALGORITHMS_EPSILON_TEST/EPSILON_TEST_comb_port.4.negativedijkstra_2.12..._SEED=3'
    # epsilon_study_path = 'Evaluation/Results/FINAL_ALGORITHMS_EPSILON_TEST/EPSILON_TEST_greedydijkstra_2.12..._SEED=3'
    # epsilon_study_path = 'Evaluation/Results/FINAL_ALGORITHMS_EPSILON_TEST/EPSILON_TEST_levywalksdijkstra_2.12..._SEED=3'

    if False:
        epsilon_study(epsilon_study_path=epsilon_study_path, show_plots=False, save_data=True)

    # Exit the script
    sys.exit()




