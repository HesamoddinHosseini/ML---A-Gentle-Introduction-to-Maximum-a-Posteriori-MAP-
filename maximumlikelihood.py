from scipy import stats
import numpy as np
from scipy.optimize import minimize

my_sample = np.random.normal(loc=0, scale=3, size=20)

def gaussian(params):
    mean = params[0]   
    sd = params[1]

    # Calculate negative log likelihood
    sum_log = -np.sum(stats.norm.logpdf(my_sample, loc=mean, scale=sd))

    return sum_log#sum of logarithms


initParams = [1, 1]# start quess

results = minimize(gaussian, initParams)
print(results.x)