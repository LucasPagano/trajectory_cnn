import torch
import torch.nn as nn
import math
import numpy as np
import collections

def Conv1d(in_channels, out_channels, kernel_size, padding, dropout=0):
    m = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
    std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)

def distance(positions):
    '''
    compute the distance between the first and last position in the positions array

    :param positions: array of shape obs_len * 2
    :return: distance between first and last position
    '''
    x1 = positions[0][0]
    x2 = positions[-1][0]
    y1 = positions[0][1]
    y2 = positions[-1][1]
    return ((positions[0][0] - positions[-1][0] )**2 + (positions[0][1] - positions[-1][1] )**2)**0.5


class TrajEstimator(nn.Module):
    def __init__(self, obs_len, pred_len, embedding_dim=16, encoder_h_dim=16, num_layers=3, dropout=0.0):
        super(TrajEstimator, self).__init__()
        #params
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.encoder_h_dim = encoder_h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout

        #layers
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.relu = nn.ReLU()
        conv_dict = collections.OrderedDict()
        if num_layers >= 2:
            for i in range(0, num_layers * 2, 2):
                conv_dict[str(i)] = Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1, dropout=0)
                conv_dict[str(i+1)] = nn.ReLU()
        self.convs = nn.Sequential(conv_dict)
        conv_out_size = embedding_dim * self.obs_len
        self.hidden2pos = nn.Sequential(
            nn.Linear(conv_out_size + 1, int((conv_out_size+1)/2)),
            nn.Linear(int((conv_out_size+1)/2), 2 * self.pred_len))


    def forward(self, obs_traj, obs_traj_rel, seq_start_end, epoch=0):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        batch_size = obs_traj.size(1)
        #compute ade and fde for later
        view_obs = np.swapaxes(obs_traj.detach().cpu().numpy(), 0, 1)
        final_dist = torch.from_numpy(np.fromiter((distance(x) for x in view_obs), view_obs.dtype)).cuda().view(batch_size, -1)

        obs_traj_embedding = self.spatial_embedding(obs_traj_rel.view(-1, 2))
        debug_obs = obs_traj.detach().cpu().numpy()
        obs_traj_embedding = obs_traj_embedding.view(-1, batch_size, self.embedding_dim
        ).permute(1, 2, 0)
        state = self.convs(obs_traj_embedding).view(batch_size, self.obs_len * self.embedding_dim)
        #add ade and fde to state
        state_fde = torch.cat((state, final_dist), dim=1)
        rel_pos = self.hidden2pos(state_fde)
        pred_traj_fake_rel = rel_pos.reshape(batch_size, self.pred_len, 2).permute(1, 0, 2)
        return pred_traj_fake_rel


