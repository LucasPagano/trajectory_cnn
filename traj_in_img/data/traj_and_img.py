import logging
import os
import math

import numpy as np

import torch
from torch.utils.data import Dataset
from traj_in_img.utils import get_obstacle_maps

logger = logging.getLogger(__name__)


def seq_collate(data):
    (obs_seq_list, pred_seq_list) = zip(*data)
    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
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


def trajs_to_img(obstacle_map, obs_traj, pred_traj):
    x_obs, y_obs = obs_traj[0, :], obs_traj[1, :]
    x_pred, y_pred = pred_traj[0, :], pred_traj[1, :]
    obs_img, pred_img = np.copy(obstacle_map), np.copy(obstacle_map)


    pass


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
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
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
        #link file name to dataset
        file_name_dict = {
            "biwi_eth.txt" : "eth",
            "biwi_hotel.txt" : "hotel",
            "crowds_zara01.txt" : "zara1",
            "crowds_zara02.txt" : "zara2",
            #since it's for annotation file, we actually don't care about which zara we point to
            "crowds_zara03.txt" : "zara2",
            "uni_examples.txt" : "univ",
            "students003.txt" : "univ",
            "students001.txt" : "univ"
        }

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
            total_peds = 0
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

                    key = path.split("/")[-1].replace("_train", "").replace("_val", "")
                    dsets.append(file_name_dict[key])
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

        obstacle_imgs_dict = get_obstacle_maps()
        obs_imgs = []
        target_imgs = []
        obs_trajs = seq_list[:, :, :self.obs_len]
        pred_trajs = seq_list[:, :, self.obs_len:]
        for ped in range(total_peds):
            dset = dsets[ped]
            obstacle_map = obstacle_imgs_dict[dset]
            obs_traj = obs_trajs[ped, :, :]
            pred_traj = pred_trajs[ped, :, :]
            obs_img, target_img = trajs_to_img(obstacle_map, obs_traj, pred_traj)


        self.obstacle_maps = torch.from_numpy(np.array(obs_imgs))

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [self.obs_traj[start:end, :], self.pred_traj[start:end, :],]
        return out
