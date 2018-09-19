import argparse
import pandas as pd
from scripts.evaluate_model import main

parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)

if __name__ == "__main__":
    columns = ["Threshold", "Dataset", "ADE", "FDE"]
    final = pd.DataFrame(columns=columns)
    datasets = ["eth", "hotel", "univ", "zara1", "zara2"]
    thresholds = [0, 0.5, 1, 1.5]
    args = parser.parse_args()
    models_dir = "../models/sgan-models/"
    for threshold in thresholds:
        results = []
        for dataset in datasets:
            print("Evaluating {}".format(dataset))
            args.model_path = models_dir + dataset + "_12_model.pt"
            args.threshold = threshold
            ade, fde = main(args)
            results.append([threshold, dataset, ade, fde])
        temp_df = pd.DataFrame(results, columns=columns)
        avg = temp_df.mean()
        avg.loc["Dataset"] = "AVG"
        temp_df = temp_df.append(avg, ignore_index=True)
        final = final.append(temp_df)

    final = final.round(decimals=2)
    final.to_csv("ade_fde_sgan_all_threshold.csv", index=False)