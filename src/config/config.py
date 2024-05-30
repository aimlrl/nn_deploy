import os
import pathlib
import src

NUM_INPUTS = 2
NUM_LAYERS = 3
P = [NUM_INPUTS,2,1]

f = [None,"Linear","Sigmoid"]

LOSS_FUNCTION = "Mean Squared Error"
MINI_BATCH_SIZE = 1

PACKAGE_ROOT = pathlib.Path(src.__file__).resolve().parent
DATAPATH = os.path.join(PACKAGE_ROOT,"datasets") 
#"/src/datasets"
SAVED_MODEL_PATH = os.path.join(PACKAGE_ROOT,"trained_models")
#"/src/trained_models"

#theta0 = [None]
#theta = [None]

#z = [None]*NUM_LAYERS
#h = [None]*NUM_LAYERS

"""
del_fl_by_del_z = [None]*NUM_LAYERS
del_hl_by_del_theta0 = [None]*NUM_LAYERS
del_hl_by_del_theta = [None]*NUM_LAYERS
del_L_by_del_h = [None]*NUM_LAYERS
del_L_by_del_theta0 = [None]*NUM_LAYERS
del_L_by_del_theta = [None]*NUM_LAYERS
"""