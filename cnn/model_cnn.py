import torch.nn as nn
import math

def Conv1d(in_channels, out_channels, kernel_size, padding, dropout=0):
    m = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
    std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)

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
        self.convs = nn.ModuleList()
        if num_layers >= 2:
            for i in range(num_layers):
                self.convs.append(Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1, dropout=0))
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(embedding_dim * self.obs_len, 2 * self.pred_len)
        self.relu = nn.ReLU()

    def forward(self, obs_traj, obs_traj_rel, seq_start_end):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        batch_size = obs_traj.size(1)
        obs_traj_embedding = self.spatial_embedding(obs_traj_rel.view(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(
            -1, batch_size, self.embedding_dim
        ).permute(1, 2, 0)

        for i, conv in enumerate(self.convs):
            if i == 0:
                state = self.relu(conv(obs_traj_embedding))
            else:
                state = self.relu(conv(state))
        state = state.view(batch_size, -1)
        rel_pos = self.hidden2pos(state)
        rel_pos = rel_pos.reshape(batch_size, self.pred_len, 2).permute(1, 0, 2)
        pred_traj_fake_rel = rel_pos
        return pred_traj_fake_rel