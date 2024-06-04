import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scenarioxp as sxp
import random

class HalfCircleScenario(sxp.Scenario):
    def __init__(self, params : pd.Series):
        super().__init__(params)

        

        self.center = np.array([0.25,0])
        self.radius = 0.75

        point = np.array([params["x"], params["y"]])

        dist2center = np.linalg.norm(point - self.center)
     
        inside = (point[0] <= self.center[0]) \
            and (dist2center <= self.radius) \
            and (np.abs(point[1]) <= 0.5)
        
        

        self._score = pd.Series({"inside" : int(inside)})
        return


def plot(history : list[sxp.FindSurfaceExplorer]):

    #


    fig = plt.figure(figsize = (5,2))
    ax = fig.gca()


    # Plot the points
    for exp in history:
        x = exp.params_history["x"]
        y = exp.params_history["y"]
        colors = ["red","black"]
        color = [colors[i] for i in exp.score_history["inside"]]
        # print(color)
        ax.scatter(
            x,y, 
            color="black",
            marker = ".",
            # facecolors="none",
            alpha = 1,
            zorder = 2
        )
        # ax.plot(x,y, 
        #         color="black", 
        #         linewidth = 2.5,
        #         )
        continue
    
    """
    Plot the circle
    """


    # Define the angle range for half circle
    theta = np.linspace(0, np.pi, 100)

    # Define the x and y coordinates for the half circle
    radius = 0.75
    x = radius * -np.sin(theta) + 0.25
    y = radius * np.cos(theta)

    # Plot the half circle
    ax.plot(x, y, color='black', zorder = 2)

    # Add a line to close the shape
    ax.plot([x[-1], x[0]], [y[-1], y[0]], color='black', zorder=2)

    # Fill the area under the half circle with gray hatching
    ax.fill_between(
        x, y, 
        hatch='\\\\',
        edgecolor="gray", 
        facecolor="white",
        alpha = 0.4,
        zorder = 1
    )

    # Set aspect ratio to equal to get a perfect circle
    ax.set_aspect('equal', adjustable='box')

    # Set axis limits
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.5, 0.5)

    # Y ticks
    ax.set_yticks([-0.5,-0.25,0,.25,.5])

    # Equal Aspect
    ax.set_aspect('equal')

    # Use LaTeX for text rendering
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax.set_xlabel("$P_1$")
    ax.set_ylabel("$P_2$")

    plt.tight_layout()
    plt.savefig("find-surface.pdf",bbox_inches="tight")
    return

def generate_data() -> list[sxp.FindSurfaceExplorer]:
    params_df = pd.read_csv("params.csv")
    manager = sxp.ScenarioManager(params_df)
    tsc = lambda s: s["inside"] == 1
    scenario = HalfCircleScenario
    amt = 10
    
    
    history = []

    for i in range(amt):
        """
        Use a sequence explorer to get into the performance mode
        """
        seq_exp = sxp.SequenceExplorer(
            strategy = sxp.SequenceExplorer.MONTE_CARLO,
            seed = random.randint(0,9999),
            scenario_manager = manager,
            scenario = scenario,
            target_score_classifier = tsc
        )

        while not seq_exp.step():
            continue
        
        """
        Now find the surface
        """
        fs_exp = sxp.FindSurfaceExplorer(
            root = seq_exp.arr_history[-1],
            seed = random.randint(0,9999),
            scenario_manager = manager,
            scenario = scenario,
            target_score_classifier = tsc
        )

        while not fs_exp.step():
            continue

        history.append(fs_exp)
        
        continue
    
    return history

def main():
    history = generate_data()
    plot(history)
    return

if __name__ == "__main__":
    main()