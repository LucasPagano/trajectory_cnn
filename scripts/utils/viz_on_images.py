import os
import sys
import numpy as np
import cv2 as cv
import pandas as pd
from attrdict import AttrDict
from cnn.model_cnn import TrajEstimator
from cnn.model_cnn_moving_threshold import TrajEstimatorThreshold
from sgan.models import TrajectoryGenerator as TrajectoryGenerator_sgan
from sgan.utils import relative_to_abs
import torch
import skvideo.io

def get_generator_sgan(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator_sgan(
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
    generator.eval()
    return generator

def get_generator_cnn(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajEstimator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        num_layers=args.num_layers,
        dropout=args.dropout)
    generator.load_state_dict(checkpoint['g_best_state'])
    generator.cuda()
    generator.eval()
    return generator

def get_generator_cnn_threshold(checkpoint):
    args = AttrDict(checkpoint['args'])
    if not hasattr(args, "threshold"):
        threshold = 0.5
    else:
        threshold=args.threshold

    generator = TrajEstimatorThreshold(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        num_layers=args.num_layers,
        threshold=threshold,
        dropout=args.dropout)
    generator.load_state_dict(checkpoint['g_best_state'])
    generator.cuda()
    generator.eval()
    return generator

def world_to_img(world_coordinates, hom_matrix):
    scaled_trajs = []

    inv_matrix = np.linalg.inv(hom_matrix)

    # if several sequences
    if len(world_coordinates.shape) > 2:
        # easier to iterate over them
        world_coordinates = np.swapaxes(world_coordinates, 0, 1)

        for traj in world_coordinates:
            ones = np.ones((len(traj), 1))
            P = np.hstack((traj, ones))
            R = np.dot(inv_matrix, P.transpose()).transpose()
            y = (R[:, 0]/R[:, 2]).reshape(-1, 1)
            x = (R[:, 1]/R[:, 2]).reshape(-1, 1)
            scaled_trajs.append(np.hstack((x, y)))
    else:
        ones = np.ones((len(world_coordinates), 1))
        P = np.hstack((world_coordinates, ones))
        R = np.dot(inv_matrix, P.transpose())
        y = (R[0, :]/R[2, :]).reshape(-1, 1)
        x = (R[1, :]/R[2, :]).reshape(-1, 1)
        scaled_trajs.append(np.hstack((x, y)))
    return scaled_trajs


def img_to_world(input, matrix):
    return world_to_img(input, np.linalg.inv(matrix))

def get_frame(video_path, frame):
    cap = cv.VideoCapture(video_path)
    cap.set(cv.CAP_PROP_POS_FRAMES, frame)
    _, img = cap.read()
    return img

def print_to_img(trajs, video_path, matrix_path, frame):
    img = get_frame(video_path, frame)
    if trajs is not None:
        matrix = np.loadtxt(matrix_path, dtype=float)
        heigth, width, _ = img.shape

        scaled_trajs = {}
        for ped_id, ped in trajs.items():
            scaled_trajs[ped_id] = {}
            for traj_name, traj in ped.items():
                scaled_traj = []
                if traj.size != 0:
                    scaled_traj = world_to_img(traj, matrix)[0]
                scaled_trajs[ped_id][traj_name] = scaled_traj

        for ped_id, ped in scaled_trajs.items():
            for ped_seq_name, ped_sequence in ped.items():
                color = color_dict[ped_seq_name]
                if len(ped_sequence) > 0:
                    #draw pred_gt thicker if we can compute ade/fde on it
                    thick = 3 if ped_seq_name == "pred_gt" and len(ped_sequence) == 12 else 1

                    for index, point in enumerate(ped_sequence[:-1, :]):
                        real_pt_1 = tuple([int(round(x)) for x in point])
                        real_pt_2 = tuple([int(round(x)) for x in ped_sequence[index + 1]])
                        cv.line(img, real_pt_1, real_pt_2, color, thick)
    return img

def get_trajs(frame, step=10):
    '''
    :param frame: last observed frame
    :param step: step between each frame
    :returns None if no prediction can be made, or trajs_, a dictionnary containing trajectories for each pedestrian
    '''

    trajs_ = {}

    # -1 because we include in selection
    seq_range = [frame - (obs_len - 1) * step, frame + pred_len * step]
    obs_range = [frame - (obs_len - 1) * step, frame]

    raw_obs_seq = data.loc[data["frameID"].between(obs_range[0], obs_range[1], inclusive=True)]
    raw_pred_seq = data.loc[data["frameID"].between(obs_range[1] + step, seq_range[1], inclusive=True)]
    peds_in_seq = raw_obs_seq.pedID.unique()

    curr_seq = np.zeros((len(peds_in_seq), 2, obs_len))
    curr_seq_rel = np.zeros((len(peds_in_seq), 2, obs_len))
    id_list = []
    considered_ped = 0

    for ped_id in peds_in_seq:
        obs_ped_seq = raw_obs_seq.loc[raw_obs_seq.pedID == ped_id]
        # seq has to have at least obs_len length
        if len(obs_ped_seq.frameID) == obs_len:
            id_list.append(ped_id)

            pred_ped_seq = raw_pred_seq.loc[raw_pred_seq.pedID == ped_id]
            trajs_[ped_id] = {}

            obs_traj = obs_ped_seq[["x", "y"]].values.transpose()
            obs_traj_rel = np.zeros(obs_traj.shape)
            obs_traj_rel[:, 1:] = obs_traj[:, 1:] - obs_traj[:, :-1]

            curr_seq[considered_ped, :, 0:obs_len] = obs_traj
            curr_seq_rel[considered_ped, :, 0:obs_len] = obs_traj_rel

            trajs_[ped_id]["obs"] = obs_traj.transpose()
            trajs_[ped_id]["pred_gt"] = pred_ped_seq[["x", "y"]].values

            considered_ped += 1

    if considered_ped > 0:
        obs_list_tensor = torch.from_numpy(curr_seq[:considered_ped, :]).permute(2, 0, 1).cuda().float()
        obs_list_rel_tensor = torch.from_numpy(curr_seq_rel[:considered_ped, :]).permute(2, 0, 1).cuda().float()
        seq_start_end_tensor = torch.tensor([[0, considered_ped]])

        for model_name, model in models.items():
            pred_rel = model(obs_list_tensor, obs_list_rel_tensor, seq_start_end_tensor)
            pred_abs = relative_to_abs(pred_rel, obs_list_tensor[-1]).detach().cpu().numpy()
            pred_abs_reorder = np.swapaxes(pred_abs, 0, 1)
            key = "pred_" + model_name
            for i in range(considered_ped):
                ped_id = id_list[i]
                trajs_[ped_id][key] = pred_abs_reorder[i]
        return trajs_
    else:
        return None


def get_paths(dset_):
    paths_ = {}

    if dset_.split("/")[0] == "split_moving":
        dset = dset_.split("/")[1]
        model_path_us = os.path.join("scripts/save/", (dset_ + "_50epoch_with_model.pt"))
        model_path_sgan = "models/sgan-p-models/" + dset + "_12_model.pt"
        if model_path_sgan.split("/")[1] == "sgan-p-models":
            out_vid_path = "visualization/" + dset + "_" + dset_.split("/")[-1] + "_sgan-p.mp4"
        else:
            out_vid_path = "visualization/" + dset + dset_.split("/")[-1] + ".mp4"

        test_dataset_path = os.listdir("datasets/split_moving/" + dset +"/" + dset_.split("/")[-1] + "/test")
        if len(test_dataset_path) > 1:
            print("Several test datasets found : {}".format(test_dataset_path))
            while True:
                to_keep = input("Enter the name of the dataset you want to use :")
                if to_keep in test_dataset_path:
                    test_dataset_path = "datasets/" + dset + "/test/" + to_keep
                    break
        else:
            test_dataset_path = "datasets/split_moving/" + dset +"/" + dset_.split("/")[-1] + "/test/" + test_dataset_path[0]


    else:
        dset = dset_
        model_path_us = "scripts/save/" + dset + "_50epoch_with_model.pt"
        model_path_sgan = "models/sgan-p-models/" + dset + "_12_model.pt"
        if model_path_sgan.split("/")[1] == "sgan-p-models":
            out_vid_path = "visualization/" + dset + "_sgan-p.mp4"
        else:
            out_vid_path = "visualization/" + dset + ".mp4"

        test_dataset_path = os.listdir("datasets/" + dset + "/test")
        if len(test_dataset_path) > 1:
            print("Several test datasets found : {}".format(test_dataset_path))
            while True:
                to_keep = input("Enter the name of the dataset you want to use :")
                if to_keep in test_dataset_path:
                    test_dataset_path = "datasets/" + dset + "/test/" + to_keep
                    break
        else:
            test_dataset_path = "datasets/" + dset + "/test/" + test_dataset_path[0]

    scenes_and_mat_path = "scenes_and_matrices/"
    mat_path = scenes_and_mat_path + dset + ".txt"
    vid_path = scenes_and_mat_path + dset + ".avi"

    paths_["vid"] = vid_path
    paths_["mat"] = mat_path
    paths_["model_us"] = model_path_us
    paths_["model_sgan"] = model_path_sgan
    paths_["test_dataset"] = test_dataset_path
    for key, item in paths_.items():
        if not os.path.exists(item):
            print("File not found : {}".format(item))
            sys.exit(0)
    #this file is created, not required
    paths_["out_vid"] = out_vid_path
    return paths_

if __name__ == "__main__":
    #paths are relative from sgan dir
    os.chdir("../../")
    dataset = "hotel"
    obs_len = 8
    pred_len = 12
    color_dict = {"obs": (0, 0, 0), "pred_cnn": (250, 250, 0), "pred_cnn_threshold": (250, 250, 250), "pred_gt": (0, 250, 0), "pred_sgan": (0,0,250)}

    paths = get_paths(dataset)

    print("Paths :")
    for key in sorted(paths.keys()):
        print("\t{}: {}".format(key, paths[key]))

    print("Loading models.")
    models = {}
    checkpoint_cnn = torch.load(paths["model_us"])
    models["cnn"] = get_generator_cnn(checkpoint_cnn)
    models["cnn_threshold"] = get_generator_cnn_threshold(checkpoint_cnn)
    checkpoint_sgan =  torch.load(paths["model_sgan"])
    models["sgan"] = get_generator_sgan(checkpoint_sgan)

    print("Loading data.")
    data = pd.read_csv(paths["test_dataset"], sep="\t", header=None)
    data.columns = ["frameID", "pedID", "x", "y"]
    data.sort_values(by=["frameID", "pedID"])
    data.reset_index(drop=True)
    writer = skvideo.io.FFmpegWriter(paths["out_vid"])

    frameList = data.frameID.unique()
    max = frameList[-1]
    #step every ten frame for watchable video
    for frame_number in range(0,max,10):
        if frame_number%1000 == 0:
            print("Frame {}/{}".format(frame_number, max))

        trajs = None
        if frame_number in frameList:
            trajs = get_trajs(frame_number)
        img = print_to_img(trajs, paths["vid"], paths["mat"], frame_number)
        writer.writeFrame(img)