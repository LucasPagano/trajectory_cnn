# Util scripts

These scripts are utility scripts to be used when working on the models.

### Create_obstacle_map.py
Used to create obstacle maps from the `annotated.png` files in the datasets folders.

### Print_args.py
Tiny utility to print the command-line args used for a checkpoint.

### Split_moving_peds.py
Split datasets in non moving and moving pedestrians. A pedestrian is said to be not moving when the distance between his first and last observed positions is less than the distance threshold in the script (default is 0.5).

### viz.py
Utility to vizualize the trajectories stored in the `trajs_dumped/*.pkl` files.

### viz_on_images.py
Utility to draw the predicted trajectories from sgan/cnn on the video files available in `scenes_and_matrices`. Only works with eth and hotel datasets because of lacking gomography matrices for the other datasets.

