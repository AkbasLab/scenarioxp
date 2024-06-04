import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import scenarioxp as sxp
import random

from shapely.geometry import Polygon, Point

class FollowLineScenario(sxp.Scenario):
    def __init__(self, params):
        super().__init__(params)
        self._polygon = create_shape()
        point = Point(params["x"], params["y"])

        inside = int(self.polygon.contains(point)) \
            and point.x >= 0 and point.x <= 1

        self._score = pd.Series({
            "inside" : int(self.polygon.contains(point))
        })
        return

    @property
    def polygon(self) -> Polygon:
        return self._polygon

def create_shape() -> Polygon:
     # Define the radius
    width = 0.25
    height = 0.25

    # Create an array of angles from 0 to pi (180 degrees) to draw the half circle
    angles = np.linspace(0, np.pi, 100)

    # Calculate x and y coordinates of the half circle
    x = list(width * np.cos(angles) + .5)
    y = list(height * np.sin(angles) + .1)


    # Add triangle to the front. 
    for xy in [
        [1.0,.1],
        # [1,0]
    ]:
        x.insert(0,xy[0])
        y.insert(0,xy[1])
        continue

    #  Add to the back
    for xy in [
        [0,.4],
        [0,0]
    ]:
        x.append(xy[0])
        y.append(xy[1])
        continue

    # Close itself
    x.append(x[0])
    y.append(y[1])
    
    verticies = [(x[i],y[i]) for i in range(len(x))]
    polygon = Polygon(verticies)
    return polygon
    

def main():

    """
    Setup Plots
    """
    fig = plt.figure(figsize=(5,2.3))
    ax = fig.gca()


    """
    Collect info
    """ 
    params_df = pd.read_csv("params.csv")
    manager = sxp.ScenarioManager(params_df)
    scenario = FollowLineScenario
    tsc = lambda s: s["inside"] == 1

    kwargs = {
        "scenario_manager" : manager,
        "scenario" : scenario,
        "target_score_classifier" : tsc
    }


    """
    Start right on the edge.
    """
    params = pd.Series({"x":0.01, "y" : .35})
    root = np.array([params["x"], params["y"]*2])
    root_scenario = scenario(params)


    """
    Plot the Performance Boundary
    """

    x,y = root_scenario.polygon.exterior.xy
    x = list(x)[:-2]
    y = list(y)[:-2]

    # Plot the half circle
    ax.plot(x, y, color="black")

    # Fill the area under the half circle with gray hatching
    ax.fill_between(
        x, y, 
        hatch='\\\\',
        edgecolor="gray", 
        facecolor="white",
        alpha = 0.4,
        zorder = 0
    )

    
    """
    Find the surface
    """
    fs_exp = sxp.FindSurfaceExplorer(
        root = root,
        seed = 4,
        **kwargs
    )

    while not fs_exp.step():
        # print(fs_exp.stage)
        pass


    """
    Plot 
    """
    # print(fs_exp.params_history)
    # print(fs_exp.score_history)
    # x = fs_exp.params_history["x"]
    # y = fs_exp.params_history["y"]
    # ax.scatter(
    #     x,y,
    #     color = "red",
    #     marker = "+",
    #     zorder = 2
    # )



    """
    Follow the surface
    """
    root = fs_exp._arr_history[-1]
    brrt_exp = sxp.BoundaryRRTExplorer(
        root = root,
        root_n = sxp.orthonormalize(root, fs_exp.v)[0],
        delta_theta = 90 * np.pi / 180,
        scale = 2.,
        **kwargs
    )
    while True:
        brrt_exp.step()
        if len(brrt_exp._arr_history) >= 1000:
            break

    x = brrt_exp.params_history["x"]
    y = brrt_exp.params_history["y"]
    ax.scatter(
        x,y,
        color = "black",
        marker = ".",
        zorder = 1,
        alpha = 0.5
    )
    
    ax.grid(False)

    ax.set_xlim(0,1)
    ax.set_ylim(0,.5)

    # Use LaTeX for text rendering
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax.set_xlabel("$P_1$")
    ax.set_ylabel("$P_2$")

    # Custom legend elements
    custom_patches = [
        Patch(
            facecolor='white', 
            edgecolor='black', 
            label='Solid White'
        ),
        Patch(
            facecolor='white', 
            edgecolor='gray', 
            hatch='\\\\', 
            label='Gray Hash "\\" pattern'
        )
    ]

    plt.legend(
        custom_patches, 
        ['$A\'$', '$A$'], 
        loc='upper right'
    )

    # Set aspect ratio to equal to ensure the circle is not distorted
    # ax.axis('equal')
    plt.tight_layout()
    plt.savefig("boundary.pdf", bbox_inches = "tight")

    print("A : A'")
    inside = (brrt_exp.score_history["inside"] == 1).sum()
    print("%3d : %3d" % (inside, 1000-inside))

    return

if __name__ == "__main__":
    main()