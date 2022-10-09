import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path

x = list(range(50, 1600, 100))
x[-1] = 1500

parser = argparse.ArgumentParser(description='Data downloading and preprocessing')
parser.add_argument('--policy', type=str, default="ego_nets",
                    help='Subgraph selection policy in {ego_nets, ego_nets_plus}'
                            ' (default: ego_nets)')
args = parser.parse_args()
name = {"ego_nets": "EGO", "ego_nets_plus": "EGO+"}

metric = "Metric/test_mean"
outdir = Path(f"out/{args.policy}")

labels = {"GNN": "GNN", "NESTED": "NGNN", 'deepsets': 'DS', "dss": "DSS", "sun":"SUN", "GNN-AK": "GNN-AK", "GNN-AK-ctx": "GNN-AK-ctx"}

for model in ["GNN", "NESTED", "deepsets", "dss", "GNN-AK", "GNN-AK-ctx", "sun"]:
    df_list = []
    for seed in range(1, 4):
        # plot lines
        df_list.append(pd.read_csv(outdir / f"{model}-{seed}.csv"))

    df = pd.concat(df_list)
    y = df.groupby(["num_train_data"]).agg("mean")[metric]
    error = df.groupby(["num_train_data"]).agg(lambda x: np.std(x, 0))[metric]

    y = y.to_numpy()
    error = error.to_numpy()

    plt.plot(x, y, label = labels[model], marker='.')
    plt.fill_between(x, y-error, y+error, alpha=0.5)

#plt.yscale("log")

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('# training examples', fontsize=16)
plt.ylabel('Test MAE', fontsize=16)

plt.title(f"4-Cycles ({name[args.policy]})", fontsize=16, y=1.27)

plt.legend(bbox_to_anchor=(0,1.02,1,0.2),loc="lower left",
                mode="expand", borderaxespad=0, ncol=3, fontsize=14)
plt.savefig(f"{args.policy}-plot.pdf", bbox_inches='tight')
