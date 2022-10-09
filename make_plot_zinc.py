import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path

x = list(range(1_000, 11_000, 1_000))


policy = "ego_nets"
metric = "Metric/test_mean"
outdir = Path(f"out/ZINC/{policy}")

labels = {"GNN": "GNN", "NESTED": "NGNN", 'deepsets': 'DS', "dss": "DSS", "sun":"SUN", "GNN-AK-ctx": "GNN-AK-ctx", "GNN-AK":"GNN-AK", "SUN": "sun"}

parser = argparse.ArgumentParser(description='Data downloading and preprocessing')
parser.add_argument('--models', type=str, nargs='+', help='which models to plot (default: all)')
args = parser.parse_args()

models = args.models
if models is None:
    models = ["GNN", "NESTED", "deepsets", "dss", "GNN-AK", "GNN-AK-ctx", "sun"]

for model in models:
    df_list = []
    for seed in range(1, 11):
        # plot lines
        df_list.append(pd.read_csv(outdir / f"{model}-{seed}.csv"))

    df = pd.concat(df_list)
    y = df.groupby(["num_train_data"]).agg("mean")[metric]
    error = df.groupby(["num_train_data"]).agg(lambda x: np.std(x))[metric]

    y = y.to_numpy()
    error = error.to_numpy()

    print(f"{model}:", y[-1], error[-1])

    plt.plot(x, y, label = labels[model], marker='.')
    plt.fill_between(x, y-error, y+error, alpha=0.5)


#plt.yscale("log")

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('# training examples', fontsize=16)
plt.ylabel('Test MAE', fontsize=16)

plt.title("ZINC (EGO)", fontsize=16, y=1.27)

#plt.legend(loc=1, fontsize=11)
plt.legend(bbox_to_anchor=(0,1.02,1,0.2),loc="lower left",
                mode="expand", borderaxespad=0, ncol=3, fontsize=14)
plt.savefig(f"{policy}-ZINC-plot.pdf", bbox_inches='tight')
