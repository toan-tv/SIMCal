from __future__ import division
from re import S
import numpy as np
from numpy import argmin
from numpy.random import default_rng
from multiprocessing import Process, Manager
import sys

def evaluate_in_parallel(fireflies, costFunc, j, parameter, sim_cfg, ret_value):
    ret_value[j] = costFunc(fireflies[j], parameter, sim_cfg)


class FireflyAlgorithm:
    def minimizer(self, function, dim, lb, ub, max_evals, parameter=None, sim_cfg=None, pop_size=15, alpha=1, betamin=1.0, gamma=0.01, seed=None):
        rng = default_rng(seed)
        lb = np.array(lb)
        ub = np.array(ub)
        fireflies = rng.uniform(lb, ub, (pop_size, dim))
        intensity = np.zeros(pop_size)

        manager = Manager()
        Processes = {}
        ret_value = manager.list([0]*pop_size)
        for i in range(pop_size):
            Processes[i] = Process(target=evaluate_in_parallel, args=(fireflies, function, i, parameter, sim_cfg, ret_value))
            Processes[i].start()
        for i in range(pop_size):
            Processes[i].join()
        for i in range(pop_size):
            intensity[i] = ret_value[i]

        best = np.min(intensity)

        evaluations = pop_size
        new_alpha = alpha
        search_range = ub - lb

        for _ in range(50):
            new_alpha *= 0.97

            Processes = {}
            ret_value = manager.list([0]*pop_size)
            for i in range(pop_size):
                Processes[i] = Process(target=evaluate_in_parallel, args=(fireflies, function, i, parameter, sim_cfg, ret_value))
                Processes[i].start()
            for i in range(pop_size):
                Processes[i].join()
            for i in range(pop_size):
                intensity[i] = ret_value[i]
            #     intensity[i] = function(fireflies[i])

            # for i in range(pop_size):
            #     intensity[i] = function(fireflies[i])
            for i in range(pop_size):
                for j in range(i):
                    if intensity[i] >= intensity[j]:
                        r = np.sum(np.square(fireflies[i] - fireflies[j]), axis=-1)
                        beta = betamin * np.exp(-gamma * r)
                        steps = new_alpha * (rng.random(dim) - 0.5) * search_range
                        fireflies[i] += beta * (fireflies[j] - fireflies[i]) + steps
                        fireflies[i] = np.clip(fireflies[i], lb, ub)
                        best = min(intensity[i], best)

        return best, fireflies[np.argmin(intensity)]
