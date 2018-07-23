import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2 as cv

trajectories_data = "scripts/trajs_dumped/univ_test_trajs.pkl"
with open(trajectories_data, "rb") as input_file:
    data = pickle.load(input_file)

datum = data[0]
obs_traj, pred_traj_fake_us, pred_traj_gt, seq_start_end = datum

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
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.show()

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
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.show()

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
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.show()

#TODO : should we denormalize? How?
#TODO : figure out how apply homography matrix
def world_to_img(world_coordinates, hom_matrix):
    scaled_trajs = []

    inv_matrix = np.linalg.inv(hom_matrix)

    #if several sequences
    if len(world_coordinates.shape) > 2:
        #easier to iterate over them
        world_coordinates = np.swapaxes(world_coordinates, 0, 1)

        for traj in world_coordinates:
            ones = np.ones((len(traj), 1))
            P = np.hstack((traj, ones))
            R = np.dot(P, inv_matrix)
            x = (R[:, 0] ).reshape(-1, 1)
            y = (R[:, 1] ).reshape(-1, 1)
            scaled_trajs.append(np.hstack((x, y)))
    else:
        ones = np.ones((len(world_coordinates), 1))
        P = np.hstack((world_coordinates, ones))
        R = np.dot(P, inv_matrix)
        x = (R[:, 0]  ).reshape(-1, 1)
        y = (R[:, 1] ).reshape(-1, 1)
        scaled_trajs.append(np.hstack((x, y)))
    return scaled_trajs

def img_to_world(input, matrix):
    return world_to_img(input, np.linalg.inv(matrix))


def print_to_img(start, end, image_path, matrix_path):
    img = cv.imread(image_path)
    matrix = np.loadtxt(matrix_path)
    trajs = {}
    trajs["obs"] = obs_traj[:, start:end, :]
    trajs["pred_us"] = pred_traj_fake_us[:, start:end, :]
    trajs["pred_gt"] = pred_traj_gt[:,start:end, :]
    scaled_trajs = {}
    for key, traj in trajs.items():
        scaled_trajs[key] = world_to_img(traj, matrix)

    heigth, width, _ = img.shape
    center = (int(heigth / 2), int(width / 2))

    print("Heigth x Width : {} x {}".format(heigth, width))

    for key, sequences in scaled_trajs.items():
        color = color_dict[key]
        for sequence in sequences:
            for point in sequence:
                real_pt = tuple(map(int, center + point))
                # real_pt = tuple(map(int, point))
                cv.circle(img, real_pt, 1, color, thickness=5)

    cv.imshow(image_path.split(".")[0], img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    color_dict = {"obs": (0,0,0), "pred_us":(250,0,0), "pred_gt":(0,250,0)}

    print_to_img(0, len(obs_traj[0]), "scenes_and_matrices/univ.png", "scenes_and_matrices/univ.txt")



