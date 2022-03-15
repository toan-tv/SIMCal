import random
import numpy as np
from modestga import minimize

class GA:
    def __init__(self, costFunc, algo_cfg, initial, parameter, sim_cfg, verbose=False) -> None:
        def callback(x, fx, ng, *args):
            print(args)
            print(f"\nx={x}\nf(x)={fx}\n")

        options = {
            'generations': algo_cfg['number_iters'],    # Max. number of generations
            'pop_size': algo_cfg['hyperparameter']['pop_size'],        # Population size
            'mut_rate': algo_cfg['hyperparameter']['mut_rate'],       # Initial mutation rate (adaptive mutation)
            'trm_size': algo_cfg['hyperparameter']['trm_size'],         # Tournament size
            'tol': algo_cfg['hyperparameter']['tol']             # Solution tolerance
        }

        bounds = []
        for veh_type in parameter:
            for param_type in parameter[veh_type]:
                bounds.append(parameter[veh_type][param_type])

        minimize(costFunc, bounds, callback=callback, options=options, workers=algo_cfg['number_processors'])
    