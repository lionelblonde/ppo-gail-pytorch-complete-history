from collections import defaultdict
from copy import deepcopy  # noqa
import glob
import argparse
import os
import hashlib
import time
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.signal import savgol_filter  # noqa
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import patches as mpatches  # noqa
from matplotlib import pyplot as plt  # noqa


parser = argparse.ArgumentParser(description="aggregate and visualize")
parser.add_argument('--style', type=str, default="dark", choices=['dark', 'light', 'paper'])
parser.add_argument('--dir', type=str, default=None, help='csv files location')
parser.add_argument('--uuid', type=str, default=None, help='uuid')
args = parser.parse_args()

# Create unique destination dir name
hash_ = hashlib.sha1()
hash_.update(str(time.time()).encode('utf-8'))
dest_dir = "plots/batchplots_{}".format(hash_.hexdigest()[:20])
os.makedirs(dest_dir, exist_ok=False)

# Colors
palette = sns.color_palette()
palette = [palette[i] for i in [2, 6, 0, 1]]

style = {'dark': 'dark_background',
         'light': 'seaborn-darkgrid',
         'paper': 'seaborn-paper'}

plt.style.use(style[args.style])

# DPI
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
# Suptitle
plt.rcParams['figure.titlesize'] = 24  # suptitle
plt.rcParams['figure.titleweight'] = 'bold'  # suptitle
# Title
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.titleweight'] = 'bold'
# X and Y axes
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
# Legend
plt.rcParams['legend.fontsize'] = 12
# Lines
plt.rcParams['lines.linewidth'] = 0.8
plt.rcParams['lines.markersize'] = 1
# Grid
if args.style == 'dark':
    plt.rcParams['grid.linewidth'] = 0.4
    plt.rcParams['grid.alpha'] = 0.15
    plt.rcParams['grid.linestyle'] = '-'

print("\n>>>>>>>>>>>>>>>>>>>> Aggregating.")

matplotlib.font_manager._rebuild()
plt.rcParams['font.family'] = 'Basier Square'

experiment_map = defaultdict(list)
ts_dump = defaultdict(list)
col_dump = defaultdict(list)
color_map = defaultdict(str)

dirs = [d.split('/')[-1] for d in glob.glob("{}/*".format(args.dir))]
print("pulling logs from sub-directories: {}".format(dirs))
dirs.sort()
dnames = deepcopy(dirs)
dirs = ["{}/{}/logs".format(args.dir, d) for d in dirs]
assert len(dirs) <= 4, "4 experiments per plot max!"

colors = {d: palette[i] for i, d in enumerate(dirs)}

raise ValueError()

for d in dirs:

    path = "{}/*/progress.csv".format(d)

    for fname in glob.glob(path):
        # Extract the expriment name from the file's full path
        experiment_name = fname.split('/')[-2]
        # Remove the 'seed' potion of the experiment name
        experiment_key = experiment_name.split('.seed')[0]
        env = experiment_name.split('.')[1]
        experiment_map[env].append(experiment_key)
        # Load data from the CSV file
        data = pd.read_csv(fname, skipinitialspace=True, usecols=["ep_env_ret"])
        # Retrieve the desired column from the data
        col = data['ep_env_ret'].to_numpy()
        # Craft an associated arbitrary time index structure
        ts = np.arange(0., np.amax(col.shape))
        # Add the experiment's data to the dictionary
        ts_dump[experiment_key].append(ts)
        col_dump[experiment_key].append(col)

        color_map[experiment_key] = colors[d]

# Remove duplicate
experiment_map = {k: list(set(v)) for k, v in experiment_map.items()}

# Display summary of the extracted data
print("summary -> {} different keys.".format(len(col_dump.keys())))
for i, key in enumerate(col_dump.keys()):
    print(">>>> [key #{}] {} | #values: {}".format(i, key, len(col_dump[key])))

print("\n>>>>>>>>>>>>>>>>>>>> Visualizing.")

texts = deepcopy(dnames)
texts.sort()
texts = [text.split('__')[-1] for text in texts]
print(texts)

patches = [plt.plot([], [], marker="o", ms=10, ls="", color=palette[i],
           label="{:s}".format(texts[i]))[0] for i in range(len(texts))]

# Calculate the x axis upper bound
maxes = defaultdict(int)
for env in experiment_map.keys():
    max_ = np.infty
    for i, key in enumerate(experiment_map[env]):
        if len(col_dump[key]) > 1:
            for col in col_dump[key]:
                max_ = len(col) if max_ > len(col) else max_
    maxes[env] = max_

# Plot mean and standard deviation
for env in experiment_map.keys():

    max_ = deepcopy(maxes[env])

    fig, ax = plt.subplots()

    for i, key in enumerate(experiment_map[env]):

        print("Mean >>>> {} in {}".format(key, color_map[key]))

        if len(col_dump[key]) > 1:

            mean = np.mean(np.column_stack([col_[0:max_] for col_ in col_dump[key]]), axis=-1)
            std = np.std(np.column_stack([col_[0:max_] for col_ in col_dump[key]]), axis=-1)

            timesteps = ts_dump[key][0][0:max_] * 1024 * 10

            #exp1: 308
            #exp4: 299
            if 'Walker2d' in key:
                print('Walker2d', len(timesteps))
                limit = 299
                timesteps = timesteps[:limit]
            #exp1: 163
            #exp4: 145
            elif 'Hopper' in key:
                print('Hopper', len(timesteps))
                limit = 145
                timesteps = timesteps[:limit]
            mean = mean[:limit]
            std = std[:limit]

            ax.plot(timesteps, mean, color=color_map[key])

            ax.fill_between(timesteps,
                            mean - 0.5 * std,
                            mean + 0.5 * std,
                            alpha=0.15, facecolor=color_map[key])

        else:
            ax.plot(ts_dump[key][0], col_dump[key][0])

    # Axis labels
    plt.xlabel("Timesteps")
    plt.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    plt.ylabel("Mean Episodic Return")
    # Indicate title
    plt.title("Mean, {}".format(env))
    # Display legend
    plt.legend(handles=patches, ncol=2, loc='upper left')
    # plt.legend(handles=patches, ncol=2, loc='lower right')

    # Save figure to disk
    plt.savefig("{}/plot_{}_{}.pdf".format(dest_dir, env, "mean"),
                format='pdf', bbox_inches='tight')
    print("mean plot done for env {}.".format(env))

# Plot CDF
for env in experiment_map.keys():

    max_ = deepcopy(maxes[env])

    fig, ax = plt.subplots()

    for i, key in enumerate(experiment_map[env]):

        print("CCDF >>>> {} in {}".format(key, color_map[key]))

        if len(col_dump[key]) > 1:

            #exp1: 308
            #exp4: 299
            if 'Walker2d' in key:
                print('Walker2d', len(timesteps))
                limit = 299
                timesteps = timesteps[:limit]
            #exp1: 163
            #exp4: 145
            elif 'Hopper' in key:
                print('Hopper', len(timesteps))
                limit = 145
                timesteps = timesteps[:limit]

            cat = np.concatenate(np.column_stack([col_[0:max_] for col_ in col_dump[key]])[:limit])

            cats = np.sort(cat)
            p = 1. * np.arange(len(cats)) / (len(cats) - 1)
            ax.plot(cats, 1 - p, color=color_map[key])

            print(key, cats.sum())

        else:
            raise ValueError()

    # Axis labels
    plt.xlabel("Episodic Return")
    plt.ylabel("Probability")
    # Indicate title
    plt.title("CCDF, {}".format(env))
    # Display legend
    plt.legend(handles=patches, ncol=1, loc='upper right')

    # Save figure to disk
    plt.savefig("{}/plot_{}_{}.pdf".format(dest_dir, env, "cdf"),
                format='pdf', bbox_inches='tight')
    print("cdf plot done for env {}.".format(env))

print(">>>>>>>>>>>>>>>>>>>> Bye.")
