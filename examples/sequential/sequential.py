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
    

def plot_2d():
    params_df = pd.read_csv("params.csv")

    manager = sxp.ScenarioManager(params_df)
    scenario = BlankScenario
    tsc = lambda s: False

    strategies = [
        sxp.SequenceExplorer.MONTE_CARLO,
        sxp.SequenceExplorer.HALTON,
        sxp.SequenceExplorer.SOBOL
    ]

    explorers = {}
    for strategy in strategies:
        exp = sxp.SequenceExplorer(
            strategy = strategy,
            seed = 4444,
            scenario_manager = manager,
            scenario = scenario,
            target_score_classifier = tsc,
            scramble = False
        )
        [exp.step() for i in range(1000)]
        explorers[strategy] = exp
    
    # for strategy, exp in explorers.items():
    #     break
    #     continue
  
    # Create the figure and GridSpec object
    fig = plt.figure(figsize=(5, 2))
    gs = fig.add_gridspec(1, 3, wspace=0)  # 3 rows, 1 column, no horizontal space between

    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)

    


    for i, items in enumerate(explorers.items()):
        ax = [ax1, ax2, ax3][i]
        # marker = [".","+","*"][i]
        strategy, exp = items
        if strategy == "random":
            strategy = "monte-carlo"
        x = exp.params_history["x"]
        y = exp.params_history["y"]
        ax.scatter(x,y,color="black",marker=".",zorder=2, s=4)
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_xlabel(strategy)
        ax.xaxis.set_label_position("top")
        ax.set_aspect('equal')
        continue
        

   
    

    
    # Set the x labels to avoid overlap
    ax1.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax2.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax3.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))

   # Set custom x-ticks to avoid overlap
    ax1.set_xticks(np.linspace(0, 1, 6))  # Remove the last x tick label
    ax2.set_xticks(np.linspace(0, 1, 6)[1:])  # Remove the last x tick label
    ax3.set_xticks(np.linspace(0, 1, 6)[1:])  # Remove the last x tick label
    
    # Optionally, remove y-axis ticks and labels
    # ax1.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax3.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    # Use LaTeX for text rendering
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    # ax2.set_xlabel("$P_1$")
    ax1.set_ylabel("$P_2$")
    fig.suptitle("$P_1$",x=.56, y=0.00)

    fig.tight_layout()
    plt.savefig("sequential.pdf",bbox_inches="tight")
    return

def main():
    plot_2d()
    return

if __name__ == "__main__":
    main()