import numpy as np
import os
import pandas as pd
import pathlib

def distance(positions):
    '''
    compute the distance between the first and last position in the positions dataframe

    :param positions: array of shape obs_len * 2
    :return: distance between first and last position
    '''
    return ((positions.x.iloc[0] - positions.x.iloc[-1])**2 + (positions.y.iloc[0] - positions.y.iloc[-1])**2)**0.5

def split(dataset_path):
    acceptable_dirs = ["test", "val", "train"]
    for dir_ in os.listdir(dataset_path):
        if dir_ in acceptable_dirs:
            print("\t\t{}".format(dir_))

            move_dir = os.path.join(output_dir, os.path.basename(dataset_path), "moving", dir_)
            not_move_dir = os.path.join(output_dir, os.path.basename(dataset_path), "not_moving", dir_)
            pathlib.Path(move_dir).mkdir(parents=True, exist_ok=True)
            pathlib.Path(not_move_dir).mkdir(parents=True, exist_ok=True)
            for file in os.listdir(os.path.join(dataset_path, dir_)):
                if file.endswith(".txt"):
                    move_file = os.path.join(move_dir, file)
                    not_move_file = os.path.join(not_move_dir, file)
                    pathlib.Path(move_file).touch(exist_ok=True)
                    pathlib.Path(not_move_file).touch(exist_ok=True)
                    data = pd.read_csv(os.path.join(dataset_path, dir_, file), sep="\t", header=None)
                    data.columns = ["frameID", "pedID", "x", "y"]
                    data.sort_values(by=["frameID", "pedID"])
                    data.reset_index(drop=True)
                    frames = data.frameID.unique()
                    move_df = pd.DataFrame(columns=["frameID", "pedID", "x", "y"])
                    not_move_df = pd.DataFrame(columns=["frameID", "pedID", "x", "y"])
                    for frame in frames:
                        seq_range = [frame - (obs_len - 1) * step, frame + pred_len * step]
                        raw_seq = data.loc[data["frameID"].between(seq_range[0], seq_range[1], inclusive=True)]
                        peds_in_seq = raw_seq.pedID.unique()

                        for ped_id in peds_in_seq:
                            ped_seq = raw_seq.loc[raw_seq.pedID == ped_id]
                            # seq has to have at least seq_len length
                            if len(ped_seq.frameID) == seq_len:
                                obs_seq = ped_seq.iloc[:obs_len, :]
                                dist = distance(obs_seq[["x", "y"]])
                                if dist < distance_threshold:
                                    not_move_df = not_move_df.append(ped_seq)
                                else:
                                    move_df = move_df.append(ped_seq)
                    move_df.drop_duplicates().to_csv(move_file, header=False, index=False, sep='\t')
                    not_move_df.drop_duplicates().to_csv(not_move_file, header=False, index=False, sep='\t')




if __name__ == "__main__":
    obs_len = 8
    pred_len = 12
    # step between each frame
    step = 10
    datasets_dir = "../../datasets/"
    output_dir = "../../datasets/split_moving"
    distance_threshold = 0.5

    seq_len = obs_len + pred_len

    print("Processing dataset :")
    for dset in os.listdir(datasets_dir):
        print("\t{}".format(dset))
        split(os.path.join(datasets_dir, dset))
