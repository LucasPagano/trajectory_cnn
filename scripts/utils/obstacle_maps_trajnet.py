import os
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt

def convert_to_obstacle_map(img):
    '''
    Function to create an obstacle map from the annotated image
    params:
    img : Image file path
    '''
    print("Processing file : {}".format(img))
    im = plt.imread(img)
    # im is a numpy array of shape (w, h, 4)
    w = im.shape[0]
    h = im.shape[1]

    obs_map = np.ones((w, h))
    im = np.around(im, decimals=3)
    R = im[:,:,0]
    np.place(obs_map, R == 0.082, 0.5)
    np.place(obs_map, R == 0.137, 0)
    return obs_map

def main():
    img_dir = "../../datasets/trajnet_stanford/img_plane_val_as_test/imgs"
    files = [os.path.join(img_dir, file) for file in os.listdir(img_dir) if file.endswith("watershed_mask.png")]
    for file in files:
        obs_map = convert_to_obstacle_map(file)
        file_name = file.split("/")[-1]
        obs_map_name = "_".join(file_name.split("_")[:2]) + "_obs_map.pkl"
        obs_map_path = os.path.join(img_dir, obs_map_name)
        f = open(obs_map_path, 'wb')
        pickle.dump(obs_map, f, protocol=2)
        f.close()

if __name__ == '__main__':
    main()
