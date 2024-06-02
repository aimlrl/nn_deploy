import os
import pandas as pd
import pickle

from src.config import config


def load_dataset(file_name):

    file_path = os.path.join(config.DATAPATH,file_name)
    #"/src/datasets/file_name"

    data = pd.read_csv(file_path)

    return data


def save_model(theta0,theta):

    pkl_file_path = os.path.join(config.SAVED_MODEL_PATH,"two_input_xor_nn.pkl")

    with open(pkl_file_path,"wb") as file_handle:
        file_handle.dump({"params":{"biases":theta0,"weights":theta},"activations":config.f})

    print("Saved model with file name {} at {}".format("two_input_xor_nn.pkl",config.SAVED_MODEL_PATH))


def load_model(file_name):

    pkl_file_path = os.path.join(config.SAVED_MODEL_PATH,file_name)

    with open(pkl_file_path,"rb") as file_handle:
        loaded_model = file_handle.load()

    return loaded_model


    