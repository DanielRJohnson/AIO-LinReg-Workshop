'''
# Name: Daniel Johnson
# File: costs.py
# Date: 1/5/2021
# Brief: this file contains the cost function for 
#        linear regresssion, root mean squared error   
'''

import numpy as np

def rmse(y, hx):
    return np.sqrt( np.sum(np.square(y - hx)) / y.shape[0])