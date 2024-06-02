import os
import pathlib
import src

NUM_INPUTS = 2
NUM_LAYERS = 3
P = [NUM_INPUTS,2,1]

f = [None,"linear","sigmoid"]

LOSS_FUNCTION = "Mean Squared Error"
MINI_BATCH_SIZE = 1

PACKAGE_ROOT = pathlib.Path(src.__file__).resolve().parent
DATAPATH = os.path.join(PACKAGE_ROOT,"datasets") 
#"/src/datasets"
SAVED_MODEL_PATH = os.path.join(PACKAGE_ROOT,"trained_models")
#"/src/trained_models"