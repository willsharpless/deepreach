import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import configargparse
from tqdm.autonotebook import tqdm

p = configargparse.ArgumentParser()
p.add_argument('--solve', default=False, action='store_true')

p.add_argument('--save_path_PDE', type=str, default="./plots/losses_toy/L_PDE_vals_-33_x2.npy")
p.add_argument('--save_path_LSS', type=str, default="./plots/losses_toy/L_LSS_vals_-33_x2.npy")
p.add_argument('--save_path_plot', type=str, default="./plots/losses_toy/test.png")

p.add_argument('--load_path_PDE', type=str, default="./plots/losses_toy/L_PDE_vals_-11_x2.npy")
p.add_argument('--load_path_LSS', type=str, default="./plots/losses_toy/L_LSS_vals_-11_x2.npy")

p.add_argument('--fname', type=str, default='fx2')

p.add_argument('--max_ab_param_val', type=float, default=1.)
p.add_argument('--grid_res_states', type=int, default=50)
p.add_argument('--grid_res_params', type=int, default=50)
opt = p.parse_args()

if opt.solve:
    opt.save_path_PDE = "./plots/losses_toy/L_PSE_vals_p" + f"{int(opt.max_ab_param_val):1d}" + f"{int(opt.max_ab_param_val):1d}" + "_" + opt.fname + ".npy"
    opt.save_path_LSS = "./plots/losses_toy/L_LSS_vals_p" + f"{int(opt.max_ab_param_val):1d}" + "_" + opt.fname + ".npy"
    opt.save_path_plot = "./plots/losses_toy/Toy_Problem_Losses_Plot_p" + f"{int(opt.max_ab_param_val):1d}" + "_" + opt.fname + ".png"
else:
    opt.load_path_PDE = "./plots/losses_toy/L_PSE_vals_p" + f"{int(opt.max_ab_param_val):1d}" + f"{int(opt.max_ab_param_val):1d}" + "_" + opt.fname + ".npy"
    opt.load_path_LSS = "./plots/losses_toy/L_LSS_vals_p" + f"{int(opt.max_ab_param_val):1d}" + "_" + opt.fname + ".npy"

# # Define the function f(x)
# def f(x):
#     return x**2

# Define the function f(x)
def f(x):
    return x**3

# Define the integrand for L_PDE
def integrand_L_PDE(x, t, theta1, theta2):
    term1 = np.sin(theta1 * t + theta2 * x)
    term2 = (1 - theta2 * t * f(x)) * np.cos(theta1 * t + theta2 * x)
    term3 = f(x) * x
    return np.abs(term1 + term2 - term3)

# Define the integrand for L_LSS
def integrand_L_LSS(x, t, theta1, theta2, alpha=1):
    term1 = t * np.sin(theta1 * t + theta2 * x)
    term2 = 0.5 * x**2 * (1 - np.exp(t**alpha)**2)
    return np.abs(term1 + term2)

# Discretize the parameter space
theta1_vals = np.linspace(-opt.max_ab_param_val, opt.max_ab_param_val, opt.grid_res_params)
theta2_vals = np.linspace(-opt.max_ab_param_val, opt.max_ab_param_val, opt.grid_res_params)
theta1_grid, theta2_grid = np.meshgrid(theta1_vals, theta2_vals)

# Integration limits
x_vals = np.linspace(-1, 1, opt.grid_res_states)
t_vals = np.linspace(0, 1, opt.grid_res_states)

def compute_L_PDE(theta1_grid, theta2_grid):
    L_PDE_vals = np.zeros_like(theta1_grid)
    with tqdm(total=opt.grid_res_params ** 2, desc="L_PDE Computation") as pbar:
        for i in range(theta1_grid.shape[0]):
            for j in range(theta1_grid.shape[1]):
                theta1 = theta1_grid[i, j]
                theta2 = theta2_grid[i, j]
                integrand_values = np.zeros((len(x_vals), len(t_vals)))
                for k, x in enumerate(x_vals):
                    for l, t in enumerate(t_vals):
                        integrand_values[k, l] = integrand_L_PDE(x, t, theta1, theta2)
                pbar.update(1)
                L_PDE_vals[i, j] = np.trapz(np.trapz(integrand_values, t_vals, axis=1), x_vals)
    return L_PDE_vals

# Function to compute L_LSS
def compute_L_LSS(theta1_grid, theta2_grid):
    L_LSS_vals = np.zeros_like(theta1_grid)
    with tqdm(total=opt.grid_res_params ** 2, desc="L_LSS Computation") as pbar:
        for i in range(theta1_grid.shape[0]):
            for j in range(theta1_grid.shape[1]):
                theta1 = theta1_grid[i, j]
                theta2 = theta2_grid[i, j]
                integrand_values = np.zeros((len(x_vals), len(t_vals)))
                for k, x in enumerate(x_vals):
                    for l, t in enumerate(t_vals):
                        integrand_values[k, l] = integrand_L_LSS(x, t, theta1, theta2)
                pbar.update(1)
                L_LSS_vals[i, j] = np.trapz(np.trapz(integrand_values, t_vals, axis=1), x_vals)
    return L_LSS_vals

def compute_L_LSS_PDE_lambda(theta1_grid, theta2_grid, lambda_val=0.):
    return (1-lambda_val) * compute_L_LSS(theta1_grid, theta2_grid) + lambda_val * compute_L_LSS(theta1_grid, theta2_grid)

if opt.solve:
    ## Solve
    L_LSS_vals, L_PDE_vals = compute_L_LSS(theta1_grid, theta2_grid), compute_L_PDE(theta1_grid, theta2_grid)

    # Save L_PDE_vals
    np.save(opt.save_path_PDE, L_PDE_vals)
    np.save(opt.save_path_LSS, L_LSS_vals)
else:
    ## Load
    L_PDE_vals = np.load(opt.load_path_PDE)
    L_LSS_vals = np.load(opt.load_path_LSS)

## Plot
fig = plt.figure(figsize=(20,10), facecolor='white')
plt.rcParams['text.usetex'] = False
pad_label = 6
max_v = max(L_LSS_vals.max(), L_PDE_vals.max())
elev, azim = 15, 60

# Plot L_LSS
ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(theta1_grid, theta2_grid, L_LSS_vals, cmap='plasma', vmin=0, vmax=max_v)

ax1.view_init(elev=elev, azim=azim)
ax1.set_xlabel(r"$\theta_1$", fontsize=12) #, labelpad=pad_label)
ax1.set_ylabel(r"$\theta_2$", fontsize=12) #, labelpad=pad_label)
ax1.set_title(r"$\mathcal{L}_{LS}$", fontsize=40) #, labelpad=10)
ax1.set_xticks([-1, 0, 1])
ax1.set_xticklabels([r'$-1$', r'$0$', r'$1$'])
ax1.set_yticks([-1, 0, 1])
ax1.set_yticklabels([r'$-1$', r'$0$', r'$1$'])
ax1.set_zticks([])
ax1.set_zlim(0, max_v)
ax1.zaxis.label.set_position((-0.1, 0.5))
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False
ax1.contour(theta1_grid, theta2_grid, L_LSS_vals, zdir='z', offset=ax1.get_zlim()[0], colors='k')

cbar1 = fig.colorbar(surf1, ax=ax1, fraction=0.02, pad=0.0)
cbar1.set_ticks([0., max_v])  # Define custom tick locations
cbar1.set_ticklabels(['0', f'{max_v:1.1f}'])  # Define custom tick labels

# Plot L_PDE
ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(theta1_grid, theta2_grid, L_PDE_vals, cmap='plasma', vmin=0, vmax=max_v)

ax2.view_init(elev=elev, azim=azim)
ax2.set_xlabel(r"$\theta_1$", fontsize=12) #, labelpad=pad_label)
ax2.set_ylabel(r"$\theta_2$", fontsize=12) #, labelpad=pad_label)
ax2.set_title(r"$\mathcal{L}_{PDE}$", fontsize=40) #, labelpad=10)
ax2.set_xticks([-1, 0, 1])
ax2.set_xticklabels([r'$-1$', r'$0$', r'$1$'])
ax2.set_yticks([-1, 0, 1])
ax2.set_yticklabels([r'$-1$', r'$0$', r'$1$'])
ax2.set_zticks([])
ax2.set_zlim(0, max_v)
ax2.zaxis.label.set_position((-0.1, 0.5))
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False
ax2.contour(theta1_grid, theta2_grid, L_PDE_vals, zdir='z', offset=ax1.get_zlim()[0], colors='k')

cbar2 = fig.colorbar(surf2, ax=ax2, fraction=0.02, pad=0.0)
cbar2.set_ticks([0., max_v])  # Define custom tick locations
cbar2.set_ticklabels(['0', f'{max_v:1.1f}'])  # Define custom tick labels

fig.savefig(opt.save_path_plot)

def update_rotate(frame):
    angle = frame % 360
    ax1.view_init(elev=30, azim=angle)
    ax2.view_init(elev=30, azim=angle)
    return fig

def update_lambda(frame):

    lambda_val = frame # Vary lambda from 0 to 1

    ## Plot
    fig = plt.figure(facecolor='white')
    plt.rcParams['text.usetex'] = False
    pad_label = 6
    L_LSS_vals, L_PDE_vals = compute_L_LSS(theta1_grid, theta2_grid), compute_L_PDE(theta1_grid, theta2_grid)
    max_v = max(L_LSS_vals.max(), L_PDE_vals.max())
    L_LSS_PDE_vals = compute_L_LSS_PDE_lambda(theta1_grid, theta2_grid, lambda_val=0.)

    # Plot L_LSS
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_surface(theta1_grid, theta2_grid, L_LSS_PDE_vals, cmap='plasma', vmin=0, vmax=max_v)
    ax1.contour(theta1_grid, theta2_grid, L_LSS_PDE_vals, zdir='z', offset=ax1.get_zlim()[0], colors='k')

    ax1.set_xlabel(r"$\theta_1$", fontsize=12, labelpad=pad_label)
    ax1.set_ylabel(r"$\theta_2$", fontsize=12, labelpad=pad_label)
    ax1.set_title(r"$\mathcal{L}_{LS}$, $\lambda$ =" + f"{lambda_val:1.1f}", fontsize=12, labelpad=10)
    ax1.set_xticks([-1, 0, 1])
    ax1.set_xticklabels([r'$-1$', r'$0$', r'$1$'])
    ax1.set_yticks([-1, 0, 1])
    ax1.set_yticklabels([r'$-1$', r'$0$', r'$1$'])
    ax1.set_zticks([])
    ax1.set_zlim(0, max_v)
    ax1.zaxis.label.set_position((-0.1, 0.5))
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False

    # Rotate
    # angle = frame % 360
    # ax1.view_init(elev=30, azim=angle)

    return fig

# # Create animation
# frames = 360  # Number of frames for a full rotation
# anim = FuncAnimation(fig, update_rotate, frames=frames, interval=50, blit=True)

# # Save the animation as a GIF
# anim.save('./plots/losses_toy/plot_losses_toy_rotation.gif', writer=PillowWriter(fps=20))