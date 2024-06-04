import warnings
warnings.filterwarnings("ignore")

import scenarioxp as sxp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from scipy.stats import qmc

class BlankScenario(sxp.Scenario):
    def __init__(self, params : pd.Series):
        super().__init__(params)
        self._score = pd.Series({"color" : 0})
        return




def one_experiment(n_dim : int, n_samples : int) -> dict:
    print("### %d-D %d Tests ###" % (n_dim, n_samples))

    dimensions = []
    for i in range(n_dim):
        s = pd.Series({
            "feat" : "P_%d" % i,
            "min" : 0,
            "max" : 1,
            "inc" : 0.01
        })
        dimensions.append(s)
        continue

    params_df = pd.DataFrame(dimensions)
    


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
        start = time.process_time()
        exp = sxp.SequenceExplorer(
            strategy = strategy,
            seed = 4444,
            scenario_manager = manager,
            scenario = scenario,
            target_score_classifier = tsc,
            scramble = False
        )

        for i in range(n_samples):
            if i % 1000 == 0:
                print(
                    "[%6s] step: %.2f%%" % ( strategy , (i+1)/n_samples * 100),
                    end="\r"
                )
            exp.step()
        print()
        end = time.process_time()
        print("CPU Process Time: %.3fs" % (end-start))
        
        explorers[strategy] = exp
    return explorers

def generate_date():
    data = []
    for n_dim in range(3,5+1):
        n_tests = int(float("1e%d" % (2*n_dim - 4)))
        
        explorers = one_experiment(n_dim, n_tests)
        for exp in explorers.values():
            exp : sxp.SequenceExplorer
            points = exp.params_history.to_numpy()
            s = pd.Series({
                "n_dim" : n_dim,
                "n_samples" : n_tests,
                "strategy" : exp._strategy,
                "points" : points
            })
            data.append(s)
            continue
        # break

    df = pd.DataFrame(data)
    df.to_pickle("discrepency.pkl")
    return

def graph_discrepency():
    df = pd.read_pickle("discrepency.pkl")
    discrepencies = []
    for points in df["points"]:
        print(points.shape)
        disc = qmc.discrepancy(points,workers = -1)
        print(disc)
        discrepencies.append(disc)
        continue
    df["discrepency"] = discrepencies
    df.to_pickle("d2.pkl")
    
    return

def main():
    graph_discrepency()
    return

if __name__ == "__main__":
    main()