import pandas as pd
import numpy as np

from src.config import config
import src.preprocessing.preprocessors as pp
from src.preprocessing import data_management

import pipeline as p

z = [None]*config.NUM_LAYERS
h = [None]*config.NUM_LAYERS


del_fl_by_del_z = [None]*config.NUM_LAYERS
del_hl_by_del_theta0 = [None]*config.NUM_LAYERS
del_hl_by_del_theta = [None]*config.NUM_LAYERS
del_L_by_del_h = [None]*config.NUM_LAYERS
del_L_by_del_theta0 = [None]*config.NUM_LAYERS
del_L_by_del_theta = [None]*config.NUM_LAYERS


def layer_neurons_weighted_sum(previous_layer_neurons_outputs, current_layer_neurons_biases, current_layer_neurons_weights):
    return current_layer_neurons_biases + np.matmul(previous_layer_neurons_outputs,current_layer_neurons_weights)



def layer_neurons_output(current_layer_neurons_weighted_sums, current_layer_neurons_activation_function):

  if current_layer_neurons_activation_function == "linear":
    return current_layer_neurons_weighted_sums

  elif current_layer_neurons_activation_function == "sigmoid":
    return 1/(1 + np.exp(-current_layer_neurons_weighted_sums))

  elif current_layer_neurons_activation_function == "tanh":
    return (np.exp(current_layer_neurons_weighted_sums) - np.exp(-current_layer_neurons_weighted_sums))/ \
            (np.exp(current_layer_neurons_weighted_sums) + np.exp(-current_layer_neurons_weighted_sums))

  elif current_layer_neurons_activation_function == "relu":
    return current_layer_neurons_weighted_sums * (current_layer_neurons_weighted_sums > 0)
  


def del_layer_neurons_outputs_wrt_weighted_sums(current_layer_neurons_activation_function, current_layer_neurons_weighted_sums):

    if current_layer_neurons_activation_function == "linear":
        return np.ones_like(current_layer_neurons_weighted_sums)

    elif current_layer_neurons_activation_function == "sigmoid":
        current_layer_neurons_outputs = 1/(1 + np.exp(-current_layer_neurons_weighted_sums))
        return current_layer_neurons_outputs * (1 - current_layer_neurons_outputs)

    elif current_layer_neurons_activation_function == "tanh":
        return (2/(np.exp(current_layer_neurons_weighted_sums) + np.exp(-current_layer_neurons_weighted_sums)))**2

    elif current_layer_neurons_activation_function == "relu":
        return (current_layer_neurons_weighted_sums > 0)
    


def del_layer_neurons_outputs_wrt_biases(current_layer_neurons_outputs_dels):
  return current_layer_neurons_outputs_dels



def del_layer_neurons_outputs_wrt_weights(previous_layer_neurons_outputs,current_layer_neurons_outputs_dels):
  return np.matmul(previous_layer_neurons_outputs.T,current_layer_neurons_outputs_dels)



def run_training(tol,epsilon):
   
    epoch_counter = 0
    mse = 1
    loss_per_epoch = list()

    pass



if __name__ == "__main__":
   run_training(10**(-4),10**(-6))


    

