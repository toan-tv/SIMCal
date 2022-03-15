import numpy as np
from src.FA import FireflyAlgorithm


def sphere(x):
    return (x[0] - 1)**2 + x[1]**2 + (x[2] + 1)**2
 

best = FireflyAlgorithm().minimizer(function=sphere, dim=3, lb=-5, ub=5, max_evals=1000)

print(best)
