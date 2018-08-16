import argparse
import pandas as pd
from scripts.evaluate_cnn import main

parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)

if __name__ == "__main__":

    results = []
    datasets = ["eth", "hotel", "univ", "zara1", "zara2"]
    args = parser.parse_args()
    for dataset in datasets:
        print("Evaluating {}".format(dataset))
        args.model_path = "save/" + dataset + "_50epoch_with_model.pt"
        ade50, fde50 = main(args)
        args.model_path = "save/" + dataset + "_100epoch_with_model.pt"
        ade100, fde100 = main(args)
        results.append([dataset, ade50, fde50, ade100, fde100])

    df = pd.DataFrame(results, columns=["Dataset", "ADE_50epochs", "FDE_50epochs", "ADE_100epochs", "FDE_100epochs"])
    avg = df.mean()
    avg.loc["Dataset"] = "AVG"
    final = df.append(avg, ignore_index=True)
    final.to_csv("ade_fde_all.csv", index=False)