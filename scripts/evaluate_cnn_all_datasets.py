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
        args.model_path = "save/" + dataset + "_100epoch_with_model.pt"
        ade, fde = main(args)
        results.append([dataset, ade, fde])

    df = pd.DataFrame(results, columns=["Dataset", "ADE", "FDE"])
    df.to_csv("evals.csv", index=False)