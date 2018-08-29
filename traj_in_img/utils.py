import os
import pickle
import numpy as np
from PIL import Image


def get_dset_path(dset_name, dset_type):
    _dir = os.path.dirname(__file__)
    _dir = os.path.abspath(os.path.join(_dir, os.pardir))
    return os.path.join(_dir, 'datasets', dset_name, dset_type)

def get_resized_obstacle_maps_and_dimensions(obs_map_dir ="../datasets/trajnet_stanford/img_plane_val_as_test/imgs_and_maps"):
    '''
    returns a dictionnary of resized obstacle_maps and their pre-resizing shape

    :param obs_map_dir: directory where the obstacle maps are
    :return:
    '''
    obs_map_files = [os.path.join(obs_map_dir, file) for file in os.listdir(obs_map_dir) if file.endswith("_obs_map.pkl")]
    obstacle_maps = {}
    for file in obs_map_files:
        with open(file, "rb") as obs_map:
            key = "_".join(os.path.basename(file).split("_")[:2])
            obstacle_maps[key] = pickle.load(obs_map)

    for key, map in obstacle_maps.items():
        #resize to (100,100) and clip to 0
        obstacle_maps[key] = (np.array(Image.fromarray(map).resize([100, 100], Image.ANTIALIAS)).clip(min=0), map.shape)
    return obstacle_maps
