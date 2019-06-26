import pickle
import os

def load_pickle(path, file_name):
    with open(os.path.join(path, file_name), "rb") as f:
        data = pickle.load(f)
    return data

def save_pickle(path,file_name, data):
    with open(os.path.join(path, file_name), "wb") as f:
        pickle.dump(data, f)