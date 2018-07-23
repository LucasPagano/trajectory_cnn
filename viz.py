import pickle
with open(r"univ_test_trajs.pkl", "rb") as input_file:
    data = pickle.load(input_file)


datum = data[0]
obs_traj, pred_traj_fake, pred_traj_fake_us, pred_traj_gt, seq_start_end = datum


import matplotlib.pyplot as plt

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


import matplotlib.patches as mpatches

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

