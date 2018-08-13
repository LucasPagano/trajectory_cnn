import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



def plot_person(start, end):
    x = obs_traj[:,start:end,0]
    y = obs_traj[:,start:end,1]
    plt.plot(x,y,color='r', linestyle='solid', marker='o')
    x_p = pred_traj_fake[:,start:end,0]
    y_p = pred_traj_fake[:,start:end,1]
    plt.plot(x_p,y_p,color='b', linestyle='solid', marker='o')
    x_g = pred_traj_gt[:,start:end,0]
    y_g = pred_traj_gt[:,start:end,1]
    plt.plot(x_g,y_g,color='g', linestyle='solid', marker='o')
    plt.plot([x[-1], x_p[0]],[y[-1], y_p[0]],color='c', linestyle='solid', marker='o')
    plt.plot([x[-1], x_g[0]],[y[-1], y_g[0]],color='c', linestyle='solid', marker='o')
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)

def plot_person_us(start, end):
    x = obs_traj[:,start:end,0]
    y = obs_traj[:,start:end,1]
    plt.plot(x,y,color='r', linestyle='solid', marker='o')
    x_p = pred_traj_fake_us[:,start:end,0]
    y_p = pred_traj_fake_us[:,start:end,1]
    plt.plot(x_p,y_p,color='b', linestyle='solid', marker='o')
    x_g = pred_traj_gt[:,start:end,0]
    y_g = pred_traj_gt[:,start:end,1]
    plt.plot(x_g,y_g,color='g', linestyle='solid', marker='o')
    plt.plot([x[-1], x_p[0]],[y[-1], y_p[0]],color='c', linestyle='solid', marker='o')
    plt.plot([x[-1], x_g[0]],[y[-1], y_g[0]],color='c', linestyle='solid', marker='o')
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)

def plot_person_both(start, end):
    x = obs_traj[:,start:end,0]
    y = obs_traj[:,start:end,1]    
    x_p = pred_traj_fake_us[:,start:end,0]
    y_p = pred_traj_fake_us[:,start:end,1]
    plt.plot(x_p,y_p,color='b', linestyle='solid', marker='o', label = 'Prediction from our model')
    x_p = pred_traj_fake[:,start:end,0]
    y_p = pred_traj_fake[:,start:end,1]
    plt.plot([x[-1], x_p[0]],[y[-1], y_p[0]],color='b', linestyle='solid', marker='o')
    plt.plot(x_p,y_p,color='m', linestyle='solid', marker='o')
    x_g = pred_traj_gt[:,start:end,0]
    y_g = pred_traj_gt[:,start:end,1]
    plt.plot(x_g,y_g,color='g', linestyle='solid', marker='o')
    r = mpatches.Patch(color='r', label = 'Observed Trajectory')
    b = mpatches.Patch(color='b', label = 'Prediction from our model')
    g = mpatches.Patch(color='g', label = 'Ground truth')
    m = mpatches.Patch(color='m', label = 'Prediction from SGAN')
    plt.legend(handles=[r, g, b, m])
    plt.plot([x[-1], x_p[0]],[y[-1], y_p[0]],color='m', linestyle='solid', marker='o')
    plt.plot([x[-1], x_g[0]],[y[-1], y_g[0]],color='g', linestyle='solid', marker='o')
    plt.plot(x,y,color='r', linestyle='solid', marker='o')
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)

if __name__ == "__main__":
    trajs_dir = "../trajs_dumped"
    for file in os.listdir(trajs_dir):
        path = os.path.join(trajs_dir, file)
        if os.path.isfile(path):
            with open(path, "rb") as input_file:
                data = pickle.load(input_file)

            datum = data[0]
            seqs = datum[:-1]
            for index, seq in enumerate(seqs):
                seqs[index] = np.swapaxes(seq, 0, 2)

            min_x = min([seq[0].min() for seq in seqs])
            max_x = max([seq[0].max() for seq in seqs])
            min_y = min([seq[1].min() for seq in seqs])
            max_y = max([seq[1].max() for seq in seqs])

            try:
                obs_traj, pred_traj_fake, pred_traj_fake_us, pred_traj_gt, seq_start_end = datum
            except ValueError:
                # pickle file only has predictions from 1 model (most probably ours)
                obs_traj, pred_traj_fake_us, pred_traj_gt, seq_start_end = datum

            total_peds = obs_traj.shape[1]
            print(file)
            print(total_peds)

            plot_person_us(20, 50)
            plt.title(file)
            plt.show()

