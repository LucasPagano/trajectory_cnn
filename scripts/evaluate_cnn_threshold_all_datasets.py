import argparse
import pandas as pd
from scripts.evaluate_cnn import main
import math
import os

parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)
parser.add_argument('--force_new_moving_threshold', default=True, type=bool)

if __name__ == "__main__":

    columns = ["Threshold", "Dataset", "ADE50", "FDE50", "ADE100", "FDE100"]
    final = pd.DataFrame(columns=columns)
    datasets = ["eth", "hotel", "univ", "zara1", "zara2"]
    thresholds = [0, 0.5, 1, 1.5]
    args = parser.parse_args()
    for threshold in thresholds:
        print("Using threshold {}".format(threshold))
        results = []
        args.threshold = threshold
        for dataset in datasets:
            print("\tEvaluating {}".format(dataset))
            args.model_path = "save/" + dataset + "_50epoch_with_model.pt"
            if os.path.isfile(args.model_path):
                ade50, fde50 = main(args)
            else:
                print("\tFile not found : {}, please check that you trained a model on dataset {}".format(args.model_path, dataset))
                ade50, fde50 = math.nan, math.nan

            args.model_path = "save/" + dataset + "_100epoch_with_model.pt"
            if os.path.isfile(args.model_path):
                ade100, fde100 = main(args)
            else:
                print("\tFile not found : {}, please check that you trained a model on dataset {}".format(args.model_path, dataset))
                ade100, fde100 = math.nan, math.nan

            results.append([threshold, dataset, ade50, fde50, ade100, fde100])

        temp_df = pd.DataFrame(results, columns=columns)
        avg = temp_df.mean()
        avg.loc["Dataset"] = "AVG"
        temp_df = temp_df.append(avg, ignore_index=True)
        final = final.append(temp_df)

    final = final.round(decimals=2)
    save_file_path = "ade_fde_cnn_all_threshold.csv"
    final.to_csv(save_file_path, index=False)
    print("Results saved to {}".format(save_file_path))