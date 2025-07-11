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


map_name = 'acoruna_port'
# experiment_name = f'{map_name}_legendarium'
experiment_name = f'{map_name}.4.negativedistance_1_50_2_0'

df_experiment = load_experiment_pd('legendarium', f'Evaluation/Results/{experiment_name}/')

# Save the DataFrame to a CSV file
# output_csv = f'Evaluation/Results/{experiment_name}/df_experiment.csv'
# df_experiment.to_csv(output_csv, index=False)

print(df_experiment)