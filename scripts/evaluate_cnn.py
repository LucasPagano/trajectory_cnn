import argparse
import os
import pathlib
import time
import numpy as np
import pickle

import torch
from attrdict import AttrDict

from cnn.model_cnn import TrajEstimator
from cnn.model_cnn_moving_threshold import TrajEstimatorThreshold
from sgan.data.loader import data_loader
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default="save/eth_50epoch_with_model.pt", type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)
parser.add_argument('--new_moving_threshold', default=False, type=bool)
parser.add_argument('--threshold', default=0, type=int)

args = parser.parse_args()

def get_generator(checkpoint):
    _args = AttrDict(checkpoint['args'])
    if args.new_moving_threshold:
        generator = TrajEstimatorThreshold(
            obs_len=_args.obs_len,
            pred_len=_args.pred_len,
            embedding_dim=_args.embedding_dim,
            encoder_h_dim=_args.encoder_h_dim_g,
            threshold=args.threshold,
            num_layers=_args.num_layers,
            dropout=_args.dropout)

    elif _args.moving_threshold != 0:
        generator = TrajEstimatorThreshold(
            obs_len=_args.obs_len,
            pred_len=_args.pred_len,
            embedding_dim=_args.embedding_dim,
            encoder_h_dim=_args.encoder_h_dim_g,
            threshold=_args.moving_threshold,
            num_layers=_args.num_layers,
            dropout=_args.dropout)
    else:
        generator = TrajEstimator(
            obs_len=_args.obs_len,
            pred_len=_args.pred_len,
            embedding_dim=_args.embedding_dim,
            encoder_h_dim=_args.encoder_h_dim_g,
            num_layers=_args.num_layers,
            dropout=_args.dropout)
    generator.load_state_dict(checkpoint['g_best_state'])
    generator.cuda()
    generator.eval()
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


def evaluate(args, loader, generator):
    trajs = []
    ade_outer, fde_outer = [], []
    total_traj = 0
    times = []
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch

            total_traj += pred_traj_gt.size(1)

            start = time.time()
            pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end)

            end = time.time()
            times.append(end - start)
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

            trajs.append([obs_traj.cpu().numpy(), pred_traj_fake.cpu().numpy(), pred_traj_gt.cpu().numpy(), seq_start_end.cpu().numpy()])
            ade_traj = displacement_error(pred_traj_fake, pred_traj_gt, mode='sum')

            fde_traj = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode='sum')

            ade_outer.append(ade_traj)
            fde_outer.append(fde_traj)
        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / total_traj
        return ade, fde, trajs, times


def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    for path in paths:
        checkpoint = torch.load(path)
        _args = AttrDict(checkpoint['args'])
        path = get_dset_path(_args.dataset_name, args.dset_type)

        generator = get_generator(checkpoint)

        _, loader = data_loader(_args, path)

        ade, fde, trajs, times = evaluate(_args, loader, generator)

        print(np.mean(times))
        print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(
            _args.dataset_name, _args.pred_len, ade, fde))
        if _args.dataset_name.split("/")[0] == "split_moving":
            path = "trajs_dumped/" + "/".join(_args.dataset_name.split("/")[:-1])
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        with open("trajs_dumped/" + args.model_path.split("/")[-1].split(".")[0] + "_" + args.dset_type + "_trajs.pkl", 'wb+') as f:
            pickle.dump(trajs, f)
        print("trajs dumped at ", args.model_path.split("/")[-1].split(".")[0] + "_" + args.dset_type + "_trajs.pkl")

    return ade.item(), fde.item()

if __name__ == '__main__':
    main(args)
