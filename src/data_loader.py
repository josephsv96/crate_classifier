from numpy import load as np_load
from json import load as json_load


def load_npy(npy_file):
    npy_arr = np_load(npy_file)
    return npy_arr


def load_json(json_file):
    with open(json_file, 'r') as fp:
        file_dict = json_load(fp)

    return file_dict
