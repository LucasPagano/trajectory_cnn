import os
import pickle

def get_dset_path(dset_name, dset_type):
    _dir = os.path.dirname(__file__)
    _dir = os.path.abspath(os.path.join(_dir, os.pardir))
    return os.path.join(_dir, 'datasets', dset_name, dset_type)

def get_obstacle_maps():
    datasets = ["eth", "hotel", "univ", "zara1", "zara2"]
    obstacle_maps = {}
    for dataset in datasets:
        dset_path = os.path.abspath(os.path.join(get_dset_path(dataset, "test"), os.pardir))
        obstacle_map_path = os.path.join(dset_path, "obs_map.pkl")
        with open(obstacle_map_path, "rb") as obs_map:
            obstacle_maps[dataset] = pickle.load(obs_map)

    return obstacle_maps
