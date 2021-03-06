import argparse
import pandas as pd
from scripts.evaluate_cnn import main
import math
import os

parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)
parser.add_argument('--force_new_moving_threshold', default=False, type=bool)
parser.add_argument('--threshold', default=0, type=int)

if __name__ == "__main__":

    columns = ["Dataset", "ADE50", "FDE50", "ADE100", "FDE100"]
    datasets = ["eth", "hotel", "univ", "zara1", "zara2"]
    args = parser.parse_args()
    results = []
    for dataset in datasets:
        print("Evaluating {}".format(dataset))
        args.model_path = "save/" + dataset + "_50epoch_with_model.pt"
        if os.path.isfile(args.model_path):
            ade50, fde50 = main(args)
        else:
            print("File not found : {}, please check that you trained a model on dataset {}".format(args.model_path, dataset))
            ade50, fde50 = math.nan, math.nan

        args.model_path = "save/" + dataset + "_100epoch_with_model.pt"
        if os.path.isfile(args.model_path):
            ade100, fde100 = main(args)
        else:
            print("File not found : {}, please check that you trained a model on dataset {}".format(args.model_path, dataset))
            ade100, fde100 = math.nan, math.nan

        results.append([dataset, ade50, fde50, ade100, fde100])

    df = pd.DataFrame(results, columns=columns)
    avg = df.mean()
    avg.loc["Dataset"] = "AVG"
    final = df.append(avg, ignore_index=True)
    final = final.round(decimals=2)
    save_file_path = "ade_fde_cnn_all.csv"
    final.to_csv(save_file_path, index=False)
    print("Results saved to {}".format(save_file_path))