from dynamics import dynamics
from utils import modules
import inspect
import configargparse
import matplotlib.pyplot as plt
import torch
import os
import pickle
import numpy as np
import matplotlib
from tqdm import tqdm
import matplotlib.patches as mpatches
matplotlib.use("Agg")


def load_experiment_setting(experiment_dir_specific, deepreach_model):
    # load original experiment settings
    with open(os.path.join(experiment_dir_specific, 'orig_opt.pickle'), 'rb') as opt_file:
        orig_opt = pickle.load(opt_file)
    dynamics_class = getattr(dynamics, orig_opt.dynamics_class)
    dynamics_ = dynamics_class(**{argname: getattr(orig_opt, argname)
                              for argname in inspect.signature(dynamics_class).parameters.keys() if argname != 'self'})
    dynamics_.deepreach_model=deepreach_model

    model = modules.SingleBVPNet(in_features=dynamics_.input_dim, out_features=1, type=orig_opt.model, mode=orig_opt.model_mode,
                                     final_layer_factor=1., hidden_features=orig_opt.num_nl, num_hidden_layers=orig_opt.num_hl)
    model.cuda()

    model.load_state_dict(torch.load(os.path.join(
        experiment_dir_specific, "training/checkpoints/model_final.pth"))["model"])

    model.eval()
    return model, dynamics_

def overlay_border(image, set, color):
    thickness = max(0, image.shape[0] // 120 - 1)
    for x in range(set.shape[0]):
        for y in range(set.shape[1]):
            if not set[x, y]:
                continue
            neighbors = [
                (x, y+1),
                (x, y-1),
                (x+1, y+1),
                (x+1, y),
                (x+1, y-1),
                (x-1, y+1),
                (x-1, y),
                (x-1, y-1),
            ]
            for neighbor in neighbors:
                if neighbor[0] >= 0 and neighbor[1] >= 0 and neighbor[0] < set.shape[0] and neighbor[1] < set.shape[1]:
                    if not set[neighbor]:
                        image[x-thickness:x+1, y -
                                thickness:y+1+thickness] = color
                        break

def plot_recovery_fig(dynamics_, model, delta_level, tMax, z_res, use_prestored_init_states):
    # 1. for ground truth slices (if available), record (higher-res) grid of learned values
    # plot (with ground truth) learned BRTs, recovered BRTs
    z_res = z_res
    plot_config = dynamics_.plot_config()
    
    fig = plt.figure()
    fig.suptitle(plot_config['state_slices'], fontsize=8)
    x_min, x_max = dynamics_.state_test_range()[
        plot_config['x_axis_idx']]
    y_min, y_max = dynamics_.state_test_range()[
        plot_config['y_axis_idx']]

    fig2 = plt.figure()

    resolution = 32
    xs = np.linspace(*dynamics_.state_test_range()
                        [plot_config['x_axis_idx']], resolution)
    ys = np.linspace(*dynamics_.state_test_range()
                        [plot_config['y_axis_idx']], resolution)
    zs = np.linspace(*dynamics_.state_test_range()
                        [plot_config['z_axis_idx']], z_res)

    xys = torch.cartesian_prod(torch.tensor(xs), torch.tensor(ys))

    xs2 = np.linspace(*dynamics_.state_test_range()
                        [plot_config['x_axis_idx']], 9)
    ys2 = np.linspace(*dynamics_.state_test_range()
                        [plot_config['y_axis_idx']], 9)/5.0


    value_grids = np.zeros((len(zs), len(xs), len(ys)))

    all_state_trajs = []
    for i in tqdm(range(len(zs)), desc='Zs', position=0, leave=False):
        coords = torch.zeros(xys.shape[0], dynamics_.state_dim + 1)
        coords[:, 0] = tMax
        coords[:, 1:] = torch.tensor(plot_config['state_slices'])
        coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
        coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]

        coords[:, 1 + plot_config['z_axis_idx']] = zs[i]

        model_results = model(
            {'coords': dynamics_.coord_to_input(coords.cuda())})
        values_ = dynamics_.io_to_value(model_results['model_in'].detach(
        ), model_results['model_out'].detach().squeeze(dim=-1)).detach().cpu()
        value_grids[i] = values_.reshape(len(xs), len(ys))

        values = value_grids[i]
        # learned BRT and recovered BRT
        ax = fig.add_subplot(1, len(zs), (i+1))
        ax.set_title('%s = %0.2f' % (
            plot_config['state_labels'][plot_config['z_axis_idx']], zs[i]), fontsize=8)

        image = np.full((*values.shape, 3), 255, dtype=int)
        BRT = values < 0
        recovered_BRT = values < delta_level

        if dynamics_.set_mode == 'reach':
            image[BRT] = np.array([252, 227, 152])
            overlay_border(image, BRT, np.array([249, 188, 6]))
            image[recovered_BRT] = np.array([155, 241, 249])
            overlay_border(image, recovered_BRT,
                                np.array([15, 223, 240]))

        else:
            image[recovered_BRT] = np.array([155, 241, 249])
            image[BRT] = np.array([252, 227, 152])
            overlay_border(image, BRT, np.array([249, 188, 6]))
            # overlay recovered border over learned BRT
            overlay_border(image, recovered_BRT,
                                np.array([15, 223, 240]))


        ax.imshow(image.transpose(1, 0, 2), origin='lower',
                    extent=(x_min, x_max, y_min, y_max))

        ############## Using initial states sampled from a grid ##############
        # batch_scenario_states = torch.zeros(xys2.shape[0], dynamics_.state_dim)
        # batch_scenario_states[:, 0:] = torch.tensor(plot_config['state_slices'])
        # batch_scenario_states[:, plot_config['x_axis_idx']] = xys2[:, 0]
        # batch_scenario_states[:, plot_config['y_axis_idx']] = xys2[:, 1]
        # batch_scenario_states[:, plot_config['z_axis_idx']] = zs[i]

        if use_prestored_init_states:
            batch_scenario_states=torch.from_numpy(np.load("./runs/prestored_trajs/state_traj%d.npy"%i))[:,0,:]
        else:
            batch_scenario_states=coords[torch.logical_and(torch.abs(values_)<0.01, dynamics_.boundary_fn(coords[...,1:])>0.2),1:]
            if batch_scenario_states.shape[0]>20:
                idx = torch.randperm(batch_scenario_states.size(0))[:20]
                batch_scenario_states=batch_scenario_states[idx]

        state_trajs, batch_scenario_costs = rollout_trajs(tMax, 0.0025, batch_scenario_states, dynamics_, model)
        
        # print(batch_scenario_states,state_trajs.shape)
        for j in range(state_trajs.shape[0]):
            ax.plot(state_trajs[j,:,plot_config['x_axis_idx']],state_trajs[j,:,plot_config['y_axis_idx']],color='blue',linestyle='--', linewidth = 0.1)

        circle = plt.Circle((0, 0), 0.5,color='darkblue', 
                         lw=0.5, 
                         fill=False)
        ax.add_artist(circle)

        ax.set_xlabel(plot_config['state_labels']
                        [plot_config['x_axis_idx']])
        ax.set_ylabel(plot_config['state_labels']
                        [plot_config['y_axis_idx']])
        ax.set_xticks([x_min, x_max])
        ax.set_yticks([y_min, y_max])
        ax.tick_params(labelsize=6)
        if i != 0:
            ax.set_yticks([])
        all_state_trajs.append(state_trajs)

        ax2= fig2.add_subplot(1, len(zs), (i+1), projection='3d')
        for j in range(state_trajs.shape[0]):
            ax2.plot(state_trajs[j,:,plot_config['x_axis_idx']],state_trajs[j,:,plot_config['y_axis_idx']],zs=state_trajs[j,:,2], 
                     color='blue',linestyle='--', linewidth = 0.1)

        
    return fig, fig2, all_state_trajs, value_grids[0]

def plot_result(experiment_dir, dynamics_, model, tMax, use_prestored_init_states):
    with open(os.path.join(experiment_dir, 'basic_logs.pickle'), 'rb') as f:
        logs = pickle.load(f)
    # 0.
    print('N:', str(logs['N']))
    print('M:', str(logs['M']))
    print('beta:', str(logs['beta']))
    print('epsilon:', str(logs['epsilon']))
    print('S:', str(logs['S']))
    print('delta level', str(logs['delta_level']))
    delta_level = logs['delta_level']
    print('learned volume', str(logs['learned_volume']))
    print('recovered volume', str(logs['recovered_volume']))
    print('theoretically recoverable volume', str(
        logs['theoretically_recoverable_volume']))
    print('recovered violation rate', str(
        logs['recovered_violation_rate']))
    z_res =5
    fig, fig2, all_state_trajs, values = plot_recovery_fig(
        dynamics_, model, delta_level, tMax, z_res, use_prestored_init_states)

    plt.tight_layout()
    fig.savefig(os.path.join(
        experiment_dir, f'traj_plots.png'), dpi=800)
    fig2.savefig(os.path.join(
        experiment_dir, f'traj_plots3D.png'), dpi=800)
    for z in range(z_res):
        np.save(os.path.join(experiment_dir, f'state_traj%d'%z),
                            all_state_trajs[z].detach().cpu().numpy())
    
    return values, delta_level

def rollout_trajs(tMax, dt, batch_scenario_states, dynamics_, model):
    # propagate scenarios
    scenario_batch_size=batch_scenario_states.shape[0]
    state_trajs = torch.zeros(scenario_batch_size, int(
        tMax/dt + 1), dynamics_.state_dim)
    ctrl_trajs = torch.zeros(scenario_batch_size, int(
        tMax/dt), dynamics_.control_dim)
    dstb_trajs = torch.zeros(scenario_batch_size, int(
        tMax/dt), dynamics_.disturbance_dim)
    ham_trajs = torch.zeros(scenario_batch_size, int(tMax/dt))

    if dynamics_.name == 'Quadrotor10DLambda':
        state_trajs[:, 0, :-1] = batch_scenario_states
        state_trajs[:, 0, -1] = 1.0
    else:
        state_trajs[:, 0, :] = batch_scenario_states
    for k in tqdm(range(int(tMax/dt)), desc='Trajectory Propagation', position=1, leave=False):

        traj_time = tMax - k*dt
        traj_times = torch.full((scenario_batch_size, ), traj_time)
        
        traj_coords = torch.cat(
            (traj_times.unsqueeze(-1), state_trajs[:, k]), dim=-1)
        traj_policy_results = model(
            {'coords': dynamics_.coord_to_input(traj_coords.cuda())})
        traj_dvs = dynamics_.io_to_dv(
            traj_policy_results['model_in'], traj_policy_results['model_out'].squeeze(dim=-1)).detach()

        # TODO: I do not think there is actually any reason to store these trajs? Could save space by removing these.

        ctrl_trajs[:, k] = dynamics_.optimal_control(
            traj_coords[:, 1:].cuda(), traj_dvs[..., 1:].cuda())
        dstb_trajs[:, k] = dynamics_.optimal_disturbance(
            traj_coords[:, 1:].cuda(), traj_dvs[..., 1:].cuda())
        ham_trajs[:, k] = dynamics_.hamiltonian(
            traj_coords[:, 1:].cuda(), traj_dvs[..., 1:].cuda())

        
        next_state_ = dynamics_.equivalent_wrapped_state(state_trajs[:, k].cuda(
        ) + dt*dynamics_.dsdt(state_trajs[:, k].cuda(), ctrl_trajs[:, k].cuda(), dstb_trajs[:, k].cuda()))
        next_state_ = torch.clamp(next_state_, torch.tensor(dynamics_.state_test_range(
        )).cuda()[..., 0], torch.tensor(dynamics_.state_test_range()).cuda()[..., 1])
        state_trajs[:, k+1] = next_state_

    batch_scenario_costs = dynamics_.cost_fn(state_trajs.cuda())
    return state_trajs, batch_scenario_costs


def generate_overlay_plot(all_values, titles, levels, fname):
    resolution = 512
    xs = np.linspace(-4,4, resolution)
    ys = np.linspace(-4,4, resolution)
    X, Y = np.meshgrid(xs, ys)

    # Target set
    theta = np.linspace(0, 2 * np.pi, 500)  # Parameter for the circle
    r = (0.5)
    circle_x = r * np.cos(theta)
    circle_y = r * np.sin(theta)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # Plot target set
    obs=ax.plot(circle_x, circle_y, color='#006d6f', linewidth=2, label='obstacle')
    i=0
    legend_handles=[]
    
    for values, title, delta_level in zip(all_values, titles, levels):
        
        zero_contour = ax.contour(X, 
                                Y, 
                                values.T, 
                                levels=[delta_level],  
                                colors=colors[i],  
                                linewidths=3,    
                                linestyles='-')  
        
        
        # manually create label for the legend
        color_patch = mpatches.Patch(color=colors[i], label=title)
        legend_handles.append(color_patch)
        # Set x and y limits
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        
        # Remove xticks and yticks
        ax.set_xticks([])
        ax.set_yticks([])
        fig.patch.set_linewidth(10)
        fig.patch.set_edgecolor('black')
        ax.get_xaxis().set_visible(False) 
        ax.get_yaxis().set_visible(False) 
        plt.tight_layout()
        i+=1
    ax.set_facecolor("#DBE2E6")
    plt.legend(handles=legend_handles)
        
    plt.savefig(fname, bbox_inches='tight',pad_inches = 0.06)

if __name__ == "__main__":
    p = configargparse.ArgumentParser()
    experiments_dir = './runs'
    exp_names = ['quadrotor10D_baseline_nrange', 'quadrotor10D_lindecay_nrange', 'lin_{0.6}_{10.0}_{5.0}_nrange','quadrotor10D_lambda_exact']
    titles = ['Baseline', 'Adaptive', 'LinDecayParamSearch','Lambda_exact']
    colors=["#243B6A","#92AFD7","#CF8E80","#550C18"]
    p.add_argument('--tMax', type=float, required=True, help='Time horizon.')
    p.add_argument('--use_prestored_init_states', default=False, action='store_true', help='use prestored initial states for plotting')
    opt = p.parse_args()
    use_prestored_init_states = opt.use_prestored_init_states
    
    # generate BRT+trajs plots for all experiments
    all_values=[]
    delta_levels=[]
    for experiment_name in exp_names:
        experiment_dir = os.path.join(
            experiments_dir, experiment_name)
        model, dynamics_ = load_experiment_setting(
            experiment_dir, deepreach_model="exact")
    
        values, delta_level=plot_result(experiment_dir, dynamics_, model, opt.tMax, use_prestored_init_states)
        all_values.append(values)
        delta_levels.append(delta_level)
    
    # plot overlay verified BRTs and overlay learned BRTs
    generate_overlay_plot(all_values,titles,delta_levels,"runs/overlay_verified_BRTs.png")
    generate_overlay_plot(all_values,titles,[0.0 for n in range(5)],"runs/overlay_learned_BRTs.png")
    

