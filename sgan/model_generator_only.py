import torch
import torch.nn as nn

import torch.nn.functional as F
import math


def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


class PoolHiddenNet(nn.Module):
    """Pooling module as proposed in our paper"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0
    ):
        super(PoolHiddenNet, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim

        mlp_pre_dim = embedding_dim + h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim]

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout)

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]
            # Repeat -> H1, H2, H1, H2
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)
            # Repeat position -> P1, P2, P1, P2
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
            # Repeat position -> P1, P1, P2, P2
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h



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
        if (num_layers >= 2):
            for i in range(num_layers):
            	self.convs.append(Conv1d(embedding_dim, embedding_dim, 3,  padding=1,dropout= 0)) # out = 8-3+1 = 6
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(embedding_dim*8, 2*12) #TODO change with seq_len
        self.relu = nn.ReLU()
        # self.pool_net = PoolHiddenNet(
        #         embedding_dim=self.embedding_dim,
        #         h_dim=embedding_dim*8,
        #         mlp_dim=64,
        #         bottleneck_dim=32,
        #         activation='relu'
        #     )

    def forward(self, obs_traj, last_pos, last_pos_rel, seq_start_end, seq_len, epoch):
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
        end_pos = obs_traj[-1, :, :]
        batch = obs_traj.size(1)
        obs_traj_embedding = self.spatial_embedding(obs_traj.view(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(
             -1, batch, self.embedding_dim
        ).permute(1,2,0)
        # pred_traj_fake_rel = []
        # for _ in range(seq_len):

        for i, conv in enumerate(self.convs):
            if (i == 0):
                state = self.relu(conv(obs_traj_embedding))
            else:
                state = self.relu(conv(state))
        state = state.view(batch, -1)
        # mlp_decoder_context_input = torch.cat(
        #         [state, pool_h], dim=1)
        rel_pos = self.hidden2pos(state)
        rel_pos = rel_pos.reshape(batch, 12, 2).permute(1,0,2)
            # curr_pos = rel_pos + last_pos
            # rel_pos_embedding = self.spatial_embedding(rel_pos)
            
            # obs_traj_embedding = obs_traj_embedding.permute(2,0,1)
            # obs_traj_embedding = torch.cat((obs_traj_embedding[1:], rel_pos_embedding.reshape(1, batch, self.embedding_dim)))
            # obs_traj_embedding = obs_traj_embedding.permute(1,2,0)

        # pred_traj_fake_rel.append(rel_pos.view(batch, -1))
        #     last_pos = curr_pos

        # pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return rel_pos

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

 
    def forward(self, obs_traj, obs_traj_rel, seq_start_end, epoch = 50, user_noise=None):
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
            seq_start_end, self.pred_len, epoch)
        pred_traj_fake_rel = encoder_out
        return pred_traj_fake_rel

