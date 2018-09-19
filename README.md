This repo is built off of the SGAN repository found <a href=https://github.com/agrimgupta92/sgan/tree/master/>here</a>.

## Model
### Sgan
Our model consists of three key components: Generator (G), Pooling Module (PM) and Discriminator (D). G is based on encoder-decoder framework where we link the hidden states of encoder and decoder via PM. G takes as input trajectories of all people involved in a scene and outputs corresponding predicted trajectories. D inputs the entire sequence comprising both input trajectory and future prediction and classifies them as “real/fake”.

<div align='center'>
  <img src='images/model.png' width='1000px'>
</div>

### Cnn
Cnn used for trajectory prediction, you can find the exact specifications <a href=https://arxiv.org/abs/1809.00696>here</a>.

## Setup
All code was developed and tested on Ubuntu 16.04 with Python 3.5 and PyTorch 0.4.

You can setup a virtual environment to run the code like this:

```bash
python3 -m venv env               # Create a virtual environment
source env/bin/activate           # Activate virtual environment
pip install -r requirements.txt   # Install dependencies
echo $PWD > env/lib/python3.5/site-packages/sgan.pth  # Add current directory to python path
# Work for a while ...
deactivate  # Exit virtual environment
```

## Pretrained Models
You can download pretrained sgan models by running the script `bash scripts/download_models.sh`. This will download the following models:

- `sgan-models/<dataset_name>_<pred_len>.pt`: Contains 10 pretrained models for all five datasets. These models correspond to SGAN-20V-20 in Table 1.
- `sgan-p-models/<dataset_name>_<pred_len>.pt`: Contains 10 pretrained models for all five datasets. These models correspond to SGAN-20VP-20 in Table 1.

Please refer to [Model Zoo](MODEL_ZOO.md) for results.

## Running Models
### Sgan
You can use the script `scripts/evaluate_model.py` to easily run any of the pretrained models on any of the datsets. For example you can replicate the Table 1 results for all datasets for SGAN-20V-20 like this:

```bash
python scripts/evaluate_model.py \
  --model_path models/sgan-models
```

### Cnn
You can use the script `scripts/evaluate_cnn.py` to run any of the cnn models.
Since the moving threshold is only active during eval time, you can change it by using the arguments `--force_new_moving_threshold` and `--threshold` if you want to change from the threshold specified during training.

You can use the script `scripts/evaluate_cnn_all_datasets.py` to run `scripts/evaluate_cnn.py` on every dataset and produce a CSV file with the results.
You can use the script `scripts/evaluate_cnn_threshold_all_datasets.py` to do the same with several moving_thresholds.

## Training new models
Instructions for training new models can be [found here](TRAINING.md).
