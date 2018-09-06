import argparse
import os
import torch
import pickle
import pathlib

from attrdict import AttrDict

from sgan.data.loader import data_loader
from cnn.model_cnn import TrajEstimator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default="save/eth_50epoch_with_model.pt", type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)

import time
import numpy as np

def get_generator(checkpoint):
    args_ = AttrDict(checkpoint['args'])
    generator = TrajEstimator(
        obs_len=args_.obs_len,
        pred_len=args_.pred_len,
        embedding_dim=args_.embedding_dim,
        encoder_h_dim=args_.encoder_h_dim_g,
        num_layers=args_.num_layers,
        dropout=args_.dropout)
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


def evaluate(args, loader, generator, num_samples):
    trajs = []
    ade_outer, fde_outer = [], []
    total_traj = 0
    times = []
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            for _ in range(num_samples):
                start = time.time()
                pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end)
                end = time.time()
                times.append(end - start)
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

                trajs.append([obs_traj.cpu().numpy(), pred_traj_fake.cpu().numpy(), pred_traj_gt.cpu().numpy(), seq_start_end.cpu().numpy()])
                ade_ = displacement_error(pred_traj_fake, pred_traj_gt, mode='raw')
                ade.append(ade_)
                fde_ = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode='raw')
                fde.append(fde_)

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
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

        ade, fde, trajs, times = evaluate(_args, loader, generator, args.num_samples)

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
    args = parser.parse_args()
    main(args)
