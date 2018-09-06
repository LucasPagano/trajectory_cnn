import logging
import os
import math

import numpy as np
import pandas as pd

import torch
from PIL import Image
from scipy.interpolate import interp1d
from torch.utils.data import Dataset
from traj_in_img.utils import get_resized_obstacle_maps_and_dimensions

logger = logging.getLogger(__name__)


def seq_collate(data):
    (obs_seq_list, pred_seq_list) = zip(*data)
    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [obs_traj, pred_traj, seq_start_end]

    return tuple(out)

def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def trajs_to_map(obs_map_and_shape, obs_traj, pred_traj, step=12):
    obstacle_map, old_obs_map_shape = obs_map_and_shape
    boundaries = np.reshape(obstacle_map.shape, (2,1))
    obs_len = obs_traj.shape[1]
    seq = np.hstack((obs_traj, pred_traj))
    rescale_fact = np.divide(old_obs_map_shape, obstacle_map.shape).reshape((2,1))
    #x and y are not in the same order in array and trajectories
    temp = np.copy(seq[0, :])
    seq[0, :] = seq[1, :]
    seq[1, :] = temp
    seq /= rescale_fact
    seq_len = seq.shape[1]

    # interpolate and clip
    frames = np.arange(start=0, stop=seq_len*step , step=step)
    full_frames = np.arange(start=0, stop=frames[-1])
    interp = interp1d(frames, seq, kind="cubic")
    full_traj = np.rint(interp(full_frames)).astype(int)

    #add pixel values
    pixval = np.array([i//step + 2 for i in full_frames]).reshape(1, full_traj.shape[1])
    traj_pixval = np.vstack((full_traj, pixval))

    #remove duplicates
    #unique sorts, we have to recreate array using indexes
    obs_stop = (obs_len-1)*step
    traj_pixval_obs = traj_pixval[:, :obs_stop]
    traj_pixval_pred = traj_pixval[:, obs_stop:]
    _, pixval_obs_index = np.unique(traj_pixval_obs, axis=1, return_index=True)
    traj_pixval_obs = traj_pixval_obs[:, sorted(pixval_obs_index)]
    _, pixval_pred_index = np.unique(traj_pixval_pred, axis=1, return_index=True)
    traj_pixval_pred = traj_pixval_pred[:, sorted(pixval_pred_index)]
    #we want it to start from 2 so the rescaling doesn't darken the image too much
    traj_pixval_pred[-1, :] -= 7

    #remove predictions or observations that go out of image
    obs_mask = traj_pixval_obs[:-1, :] < boundaries
    pred_mask = traj_pixval_pred[:-1, :] < boundaries
    if not obs_mask.all():
        traj_pixval_obs = traj_pixval_obs[:, np.logical_and(*obs_mask)]
    if not pred_mask.all():
        traj_pixval_pred = traj_pixval_pred[:, np.logical_and(*pred_mask)]

    obs_map, pred_map = np.copy(obstacle_map), np.copy(obstacle_map)
    #put uses flattened array index
    flattened_obs_idx = np.ravel_multi_index(traj_pixval_obs[:-1, :], dims=obstacle_map.shape)
    flattened_pred_idx = np.ravel_multi_index(traj_pixval_pred[:-1, :], dims=obstacle_map.shape)

    #put the pixel values where we want in array
    np.put(obs_map, flattened_obs_idx, traj_pixval_obs[-1, :])
    np.put(pred_map , flattened_pred_idx, traj_pixval_pred[-1, :])

    return obs_map, pred_map

class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1,min_ped=1, delim='\t'):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        - load_preprocess: If True, load preprocessed images else reprocess everything to create images
        """
        super(TrajectoryDataset, self).__init__()


        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        dsets = []
        total_peds = 0
        for path in all_files:
            data = read_file(path, delim)
            try :
                frames = np.unique(data[:, 0]).tolist()
            except IndexError:
                print("IndexError: too many indices for array")
                print("File {} may be empty.".format(path))
                continue
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))
            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                num_peds_considered = 0

                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==  ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    if len(curr_ped_seq[:, 0]) != self.seq_len:
                        continue
                    dset_name = "_".join(os.path.basename(path).split("_")[:2])
                    dsets.append(dset_name)
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    _idx = num_peds_considered
                    curr_seq[_idx, :, 0:self.seq_len] = curr_ped_seq
                    num_peds_considered += 1

                if num_peds_considered >= min_ped:
                    total_peds += num_peds_considered
                    num_peds_in_seq.append(num_peds_considered)
                    seq_list.append(curr_seq[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)

        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]

        obstacle_maps_dict = get_resized_obstacle_maps_and_dimensions()
        obs_maps = []
        dims = []
        target_maps = []
        obs_trajs = seq_list[:, :, :self.obs_len]
        pred_trajs = seq_list[:, :, self.obs_len:]
        trainA_path = os.path.join(os.path.dirname(self.data_dir), "trainA")
        trainB_path = os.path.join(os.path.dirname(self.data_dir), "trainB")
        if not os.path.exists(trainA_path):
            os.mkdir(trainA_path)
        if not os.path.exists(trainB_path):
            os.mkdir(trainB_path)
        dims_path = os.path.join(os.path.dirname(trainA_path), "dims.txt")
        dims_file = open(dims_path, "w+")

        for key in obstacle_maps_dict.keys():
            dims_file.write("{}:{}\n".format(key, obstacle_maps_dict[key][1]))
        dims_file.close()

        counter = {key:0 for key in obstacle_maps_dict.keys()}

        for ped in range(total_peds):
            dset = dsets[ped]
            obstacle_map = obstacle_maps_dict[dset]
            obs_traj = obs_trajs[ped, :, :]
            pred_traj = pred_trajs[ped, :, :]
            obs_map, target_map = trajs_to_map(obstacle_map, obs_traj, pred_traj)
            dims.append(obstacle_map[1])
            obs_maps.append(obs_map)
            target_maps.append(target_map)

            file_name = dset + "_" + str(counter[dset]) + ".png"
            obs_file_path = os.path.join(trainA_path, file_name)
            target_file_path = os.path.join(trainB_path, file_name)
            counter[dset] += 1

            #rescale to [0,255] range
            obs_scaling = 255.0 / obs_map.max()
            target_scaling = 255.0 / target_map.max()

            obs_map = np.uint8(obs_map * obs_scaling)
            target_map = np.uint8(target_map * target_scaling)
            #save to image
            obs_map = Image.fromarray(obs_map, "L")
            target_map = Image.fromarray(target_map, "L")
            obs_map.save(obs_file_path, "PNG")
            target_map.save(target_file_path, "PNG")

        self.obs_maps = torch.from_numpy(np.array(obs_maps))
        self.target_maps = torch.from_numpy(np.array(target_maps))

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [self.obs_maps[start:end, :, :], self.target_maps[start:end, :, :]]
        return out
