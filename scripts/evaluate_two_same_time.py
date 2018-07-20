import argparse
import os
import torch
import pickle

torch.manual_seed(42)

torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


from attrdict import AttrDict

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.model_generator_only import TrajectoryGenerator as TrajectoryGenerator_only
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--model_path2', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)


def get_generator2(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator_only(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        num_layers=args.num_layers,
        dropout=args.dropout)
    generator.load_state_dict(checkpoint['g_best_state'])
    generator.cuda()
    generator.train()
    return generator


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def evaluate(args, loader, generator, generator2, num_samples):
    trajs = []
    times = []
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            for _ in range(num_samples):
                start = time.time()
                pred_traj_fake_rel = generator(
                    obs_traj, obs_traj_rel, seq_start_end
                )
                end = time.time()
                times.append(end-start)
                pred_traj_fake_rel2 = generator2(
                    obs_traj, obs_traj_rel, seq_start_end
                )
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )
                pred_traj_fake2 = relative_to_abs(
                    pred_traj_fake_rel2, obs_traj[-1]
                )

                trajs.append([obs_traj.cpu().numpy(), pred_traj_fake.cpu().numpy(), pred_traj_fake2.cpu().numpy(), pred_traj_gt.cpu().numpy(), seq_start_end.cpu().numpy()])
                ade.append(displacement_error(
                    pred_traj_fake, pred_traj_gt, mode='raw'
                ))
                fde.append(final_displacement_error(
                    pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                ))

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        return ade, fde, trajs, times



def main(args):
    
    path = args.model_path
    path2 = args.model_path2

    checkpoint = torch.load(path)
    generator = get_generator(checkpoint)
    checkpoint2 = torch.load(path2)
    generator2 = get_generator2(checkpoint2)
    
    _args = AttrDict(checkpoint['args'])
    path = get_dset_path(_args.dataset_name, args.dset_type)
    _, loader = data_loader(_args, path)
    
    ade, fde, trajs, times = evaluate(_args, loader, generator, generator2, args.num_samples)
    
    print (times, np.mean(times))
    
    print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(
        _args.dataset_name, _args.pred_len, ade, fde))
    with open("trajs_dumped/" + _args.dataset_name + "_" + args.dset_type + "_trajs.pkl", 'wb') as f:
        pickle.dump(trajs, f)
    print ("trajs dumped at ", _args.dataset_name + "_" + args.dset_type + "_trajs.pkl")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
