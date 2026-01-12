import os
import sys
sys.path.append('.')
import pandas as pd
import numpy as np

from sylegendarium import Legendarium
from sylegendarium import load_experiment_pd
from sylegendarium import load_experiments

def save_dataframe_as_legendarium(df, experiment_name, path):
    """
    Save a pandas DataFrame in the Legendarium format
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to save
    experiment_name : str
        Name of the experiment
    path : str
        Path where to save the files
    """
    import os
    import lzma
    import pickle
    import yaml
    
    # Create directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Convert DataFrame to list of lists (format used by Legendarium)
    metrics_list = df.values.tolist()
    
    # Get column names for the meta file
    data_names = df.columns.tolist()
    
    print(f"Dumping metrics...")
    # Save the metrics
    with lzma.open(os.path.join(path, f"{experiment_name}.metrics.xz"), "wb") as f:
        pickle.dump(metrics_list, f)
    
    # Save the metrics meta
    with open(os.path.join(path, f"{experiment_name}.meta.yaml"), "w") as f:
        yaml.dump(data_names, f)
    
    print(f"Saved experiment '{experiment_name}' to {path}")

def legendarium_merge(parent_folder, experiment_folder, output_name):
    """
    Merge multiple Legendarium experiments located in subfolders of a parent folder.
    
    Parameters:
    -----------
    parent_folder : str
        The parent folder containing subfolders with Legendarium experiments
    experiment_folder : str
        The specific experiment folder name to look for in each subfolder
    output_name : str
        The name for the merged experiment output
    """
    paths_to_merge = [os.path.join(parent_folder, d) for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))]
    
    # Empty dataframe to hold all data
    all_data = pd.DataFrame()
    
    for path in paths_to_merge:
        df_experiment = load_experiment_pd('evaluation', path)
        print(f'Loaded {path} with shape {df_experiment.shape}')
        all_data = pd.concat([all_data, df_experiment], ignore_index=True)
    
    # Save the merged dataframe in Legendarium format
    save_dataframe_as_legendarium(all_data, 'merged_experiment', output_name)
    print(f'Merged and saved to {output_name}')

if __name__ == "__main__":
    parent_folder = 'Evaluation/Results/EPSILON_TEST_comb_port.4.timenegativelogdijkstra_2.30..._/0_0.5_1_Tm1'
    experiment_folder = 'comb_port.4.timenegativelogdijkstra_2.30_2.19_1.86_4.10'
    output_name = f'{parent_folder}/EpsmergedTm1_{experiment_folder}'
    legendarium_merge(parent_folder, experiment_folder, output_name)
    sys.exit(0)

