import torch
import torch.nn as nn

import torch.nn.functional as F
import math

def Conv1d(in_channels, out_channels, kernel_size, padding, dropout=0):
    m = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
    std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


class Encoder(nn.Module):
    """Encoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""
    def __init__(
        self, embedding_dim=64, h_dim=64, num_layers=1,
        dropout=0.0
    ):
        super(Encoder, self).__init__()

        self.h_dim = embedding_dim # TODO
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        #self.kernel_size = kernel_size # TODO
        # self.pad = self.kernel_size - 1
        if (num_layers == 3):
            self.convs.append(Conv1d(embedding_dim, embedding_dim, 3,  padding=1,dropout= 0)) # out = 8-3+1 = 6
            self.convs.append(Conv1d(embedding_dim, embedding_dim, 3,  padding=1,dropout= 0)) # out = 6-3+1 = 4
            self.convs.append(Conv1d(embedding_dim, embedding_dim, 3,  padding=1,dropout= 0)) # out = 4-3+1 = 2
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(embedding_dim*8, 2)
        self.relu = nn.ReLU()

    def forward(self, obs_traj, last_pos, last_pos_rel, seq_start_end, seq_len):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - last_pos: Tensor of shape (batch, 2)
        - last_pos_rel: Tensor of shape (batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - pred_traj: tensor of shape (self.seq_len, batch, 2)
        """
        # Encode observed Trajectory
        batch = obs_traj.size(1)
        obs_traj_embedding = self.spatial_embedding(obs_traj.view(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(
             -1, batch, self.embedding_dim
        ).permute(1,2,0)

        pred_traj_fake_rel = []
        for _ in range(seq_len):

            for i, conv in enumerate(self.convs):
                if (i == 0):
                    state = self.relu(conv(obs_traj_embedding))
                else:
                    state = self.relu(conv(state))

            rel_pos = self.hidden2pos(state.view(batch, -1))
            curr_pos = rel_pos + last_pos
            rel_pos_embedding = self.spatial_embedding(rel_pos)
            
            obs_traj_embedding = obs_traj_embedding.permute(2,0,1)
            obs_traj_embedding = torch.cat((obs_traj_embedding[1:], rel_pos_embedding.reshape(1, batch, self.embedding_dim)))
            obs_traj_embedding = obs_traj_embedding.permute(1,2,0)

            pred_traj_fake_rel.append(rel_pos.view(batch, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel

# TODO batch_norm
class TrajectoryGenerator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=16, encoder_h_dim=16, num_layers=3,
        dropout=0.0, batch_norm=True
    ):
        super(TrajectoryGenerator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.encoder_h_dim = encoder_h_dim
        self.embedding_dim = embedding_dim
        
        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        input_dim = encoder_h_dim

 
    def forward(self, obs_traj, obs_traj_rel, seq_start_end, user_noise=None):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        batch = obs_traj_rel.size(1)
        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]
        encoder_out = self.encoder(obs_traj_rel, last_pos,
            last_pos_rel,
            seq_start_end, self.pred_len)
        pred_traj_fake_rel = encoder_out
        return pred_traj_fake_rel

