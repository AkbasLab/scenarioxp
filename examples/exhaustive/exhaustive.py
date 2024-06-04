import warnings
warnings.filterwarnings("ignore")

import scenarioxp as sxp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class BlankScenario(sxp.Scenario):
    def __init__(self, params : pd.Series):
        super().__init__(params)
        self._score = pd.Series({"color" : 0})
        return

def main():
    params_df = pd.read_csv("params.csv")
    
    manager = sxp.ScenarioManager(params_df)
    scenario = BlankScenario
    tsc = lambda s: False

    exp = sxp.ExhaustiveExplorer(manager, scenario, tsc)
    
    fig = plt.figure(figsize=(5,2))
    ax = plt.gca()

    # Add vertical lines every 0.05
    for xc in np.arange(0.025, 1.05, 0.05):
        ax.axvline(x=xc, color='grey', linestyle='-', linewidth=0.5, zorder=1)
        ax.axhline(y=xc, color='grey', linestyle='-', linewidth=0.5, zorder=1)

    points = np.array(exp.all_combinations).T
    ax.scatter(*points,color="black",marker=".",zorder=2)

    ax.set_xlim([0,1])
    ax.set_ylim([0,0.5])
    ax.set_yticks([0,.1,.2,.3,.4,.5])
    ax.set_aspect('equal')

    # Use LaTeX for text rendering
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax.set_xlabel("$P_1$")
    ax.set_ylabel("$P_2$")


    fig.tight_layout()
    plt.savefig("exhaustive.pdf",bbox_inches="tight")
    return

if __name__ == "__main__":
    main()