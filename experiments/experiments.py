import wandb
import torch
import os
import shutil
import time
import math
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.io as spio

from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from collections import OrderedDict
from datetime import datetime
from sklearn import svm 
from utils import diff_operators
from utils.error_evaluators import scenario_optimization, ValueThresholdValidator, MultiValidator, MLPConditionedValidator, target_fraction, MLP, MLPValidator, SliceSampleGenerator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch.profiler as profiler

class Experiment(ABC):
    def __init__(self, model, dataset, experiment_dir, use_wandb):
        self.model = model
        self.dataset = dataset
        self.experiment_dir = experiment_dir
        self.use_wandb = use_wandb

    @abstractmethod
    def init_special(self):
        raise NotImplementedError

    def _load_checkpoint(self, epoch):
        if epoch == -1:
            model_path = os.path.join(self.experiment_dir, 'training', 'checkpoints', 'model_final.pth')
            self.model.load_state_dict(torch.load(model_path, weights_only=True)['model'])
        else:
            model_path = os.path.join(self.experiment_dir, 'training', 'checkpoints', 'model_epoch_%04d.pth' % epoch)
            self.model.load_state_dict(torch.load(model_path, weights_only=True)['model'])

    def validate(self, epoch, save_path, x_resolution, y_resolution, z_resolution, time_resolution):
        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)

        plot_config = self.dataset.dynamics.plot_config()

        state_test_range = self.dataset.dynamics.state_test_range()
        x_min, x_max = state_test_range[plot_config['x_axis_idx']]
        y_min, y_max = state_test_range[plot_config['y_axis_idx']]
        z_min, z_max = state_test_range[plot_config['z_axis_idx']]

        times = torch.linspace(0, self.dataset.tMax, time_resolution)
        xs = torch.linspace(x_min, x_max, x_resolution)
        ys = torch.linspace(y_min, y_max, y_resolution)
        zs = torch.linspace(z_min, z_max, z_resolution)
        xys = torch.cartesian_prod(xs, ys)
        
        fig = plt.figure(figsize=(5*len(zs), 5*len(times)))
        for i in range(len(times)):
            for j in range(len(zs)):
                coords = torch.zeros(x_resolution*y_resolution, self.dataset.dynamics.state_dim + 1)
                coords[:, 0] = times[i]
                coords[:, 1:] = torch.tensor(plot_config['state_slices'])
                coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
                coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
                coords[:, 1 + plot_config['z_axis_idx']] = zs[j]

                with torch.no_grad():
                    model_results = self.model({'coords': self.dataset.dynamics.coord_to_input(coords.cuda())})
                    values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach())
                
                ax = fig.add_subplot(len(times), len(zs), (j+1) + i*len(zs))
                ax.set_title('t = %0.2f, %s = %0.2f' % (times[i], plot_config['state_labels'][plot_config['z_axis_idx']], zs[j]))
                s = ax.imshow(1*(values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T <= 0), cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(s, cax=cax) 
        fig.savefig(save_path)
        if self.use_wandb:
            wandb.log({
                'step': epoch,
                'val_plot': wandb.Image(fig),
            })
        plt.close()

        if was_training:
            self.model.train()
            self.model.requires_grad_(True)
    
    def train(
            self, batch_size, epochs, lr, 
            steps_til_summary, epochs_til_checkpoint, 
            loss_fn, loss_fn_baseline, clip_grad, use_lbfgs, adjust_relative_grads, 
            val_x_resolution, val_y_resolution, val_z_resolution, val_time_resolution,
            use_CSL, CSL_lr, CSL_dt, epochs_til_CSL, num_CSL_samples, CSL_loss_frac_cutoff, max_CSL_epochs, CSL_loss_weight, CSL_batch_size,
            dual_lr=False, lr_decay_w=1., lr_hopf=2e-5, lr_hopf_decay_w=1., smoothing_factor=0.8, 
            hopf_loss='none', hopf_loss_decay_early = True, hopf_loss_decay=True, hopf_loss_decay_w=0.9998,
            reset_loss_w=False, reset_loss_period=0, 
            diff_con_loss_incr=False, hopf_loss_decay_type = 'exponential',
            nonlin_scale=False, nl_scale_epoch_step=10000, nl_scale_epoch_post=10000, 
            record_temporal_loss = False, use_sgd=False,
            deposit_blocking = True, deposit_blocking_period = 5000, # seg faults if nonblocking rn...,
            fin_diff = False, fd_alpha_scale = [2.5, 2., 1.5, 1.], 
            fd_delta_x_scale = [0.7, 0.5, 0.3, 0.1], fd_delta_t_scale = [0.05, 0.03, 0.02, 0.01],
        ):
        was_eval = not self.model.training
        self.model.train()
        self.model.requires_grad_(True)

        train_dataloader = DataLoader(self.dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=0)

        ## Define Optimizers and Schedulers ## TODO, would SGD be better than Adam?
        if dual_lr and self.dataset.super_pretrain:
            if use_sgd:
                optim_hopf = torch.optim.SGD(params=self.model.parameters(), lr=lr_hopf, momentum=0.2)
                optim_std = torch.optim.SGD(params=self.model.parameters(), lr=lr, momentum=0.2)
            else:
                optim_hopf = torch.optim.Adam(params=self.model.parameters(), lr=lr_hopf)
                optim_std = torch.optim.Adam(params=self.model.parameters(), lr=lr)
            lr_scheduler_hopf = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim_hopf, gamma=lr_hopf_decay_w)
            lr_scheduler_std = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim_std, gamma=lr_decay_w)
            optim = optim_hopf
            lr_scheduler = lr_scheduler_hopf
        else:
            if use_sgd:
                optim = torch.optim.SGD(params=self.model.parameters(), lr=lr, momentum=0.2)
            else:
                optim = torch.optim.Adam(params=self.model.parameters(), lr=lr)
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=lr_decay_w)

        # copy settings from Raissi et al. (2019) and here 
        # https://github.com/maziarraissi/PINNs
        if use_lbfgs:
            optim = torch.optim.LBFGS(lr=lr, params=self.model.parameters(), max_iter=50000, max_eval=50000,
                                    history_size=50, line_search_fn='strong_wolfe')

        training_dir = os.path.join(self.experiment_dir, 'training')
        
        summaries_dir = os.path.join(training_dir, 'summaries')
        if not os.path.exists(summaries_dir):
            os.makedirs(summaries_dir)

        checkpoints_dir = os.path.join(training_dir, 'checkpoints')
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        writer = SummaryWriter(summaries_dir)

        ## Params
        total_steps = 0
        JIp_s_max, JIp_s = 0., 0.
        total_pretrain_iters = self.dataset.pretrain_iters if hopf_loss=='none' else self.dataset.pretrain_iters + self.dataset.super_pretrain_iters
        self.total_pretrain_iters = total_pretrain_iters
        self.epochs = epochs
        nl_perc = 0.

        ## Dynamic Weighting
        loss_weights = {'dirichlet': 1., 'hopf': 1., 'diff_constraint_hom': 1., 
                        'dss_value_1_loss': 1., 'dss_value_2_loss': 1., 
                        'dss_grad_1_loss': 1., 'dss_vgrad_2_loss': 1., 
                        'lbss_value_loss': 1., 'lbss_grad_loss': 1.,
                        }
        if diff_con_loss_incr:
            loss_weights['diff_constraint_hom'] = 0.
        if hopf_loss_decay_type == 'negative_exponential': 
            loss_weights['hopf'] = 1 - (hopf_loss_decay_w ** (epochs - 1 - total_pretrain_iters))
        if hopf_loss == 'lin_val_grad_diff':
            loss_weights['hopf_grad'] = loss_weights['hopf']
        og_loss_weights = loss_weights.copy()

        ## Finite Differencing Weights
        fd_weight_scales = {'alpha': fd_alpha_scale, 'delta_x': fd_delta_x_scale, 'delta_t': fd_delta_t_scale}
        fd_scale_epoch_step = int((epochs - self.total_pretrain_iters) / len(fd_alpha_scale))
        fd_scale_i = 0
        diss = 0.

        ## Train
        with tqdm(total=len(train_dataloader) * epochs) as pbar:

            train_losses = []
            last_CSL_epoch = -1
            for epoch in range(0, epochs):

                if self.dataset.pretrain: # skip CSL
                    last_CSL_epoch = epoch
                time_interval_length = (self.dataset.counter/self.dataset.counter_end)*(self.dataset.tMax-self.dataset.tMin)
                CSL_tMax = self.dataset.tMin + int(time_interval_length/CSL_dt)*CSL_dt

                ## Reset Weights
                if reset_loss_w and epoch % reset_loss_period == 0 and not self.dataset.pretrain and not self.dataset.super_pretrain:
                    loss_weights = og_loss_weights.copy()

                ## If hopf-solving, Bank Deposits
                if self.dataset.solve_hopf and epoch > self.dataset.pretrain_iters and (loss_weights['hopf'] > 0. or reset_loss_w):
                    
                    ## Check Deposit Jobs
                    if not deposit_blocking and self.dataset.hjpool.jobs: # self.dataset.hjpool.jobs # FIXME: fix segf's
                        self.dataset.hjpool.check_jobs()
                    
                    ## Reorder Deposit Jobs
                    elif (not deposit_blocking and not self.dataset.hjpool.jobs) or (deposit_blocking and (epoch-self.dataset.pretrain_iters) % deposit_blocking_period == 0):

                        if self.dataset.hjpool.hopf_warm_start:
                            self.dataset.hjpool.solve_bank_deposit(model=self.model, n_splits=self.dataset.hopf_deposit_numsplits, blocking=deposit_blocking)
                        else:
                            self.dataset.hjpool.solve_bank_deposit(model=None, n_splits=self.dataset.hopf_deposit_numsplits, blocking=deposit_blocking)

                        self.dataset.solved_hopf_pts += self.dataset.hopf_bank_params["n_deposit"]
                        self.max_solved_ix = min(self.dataset.solved_hopf_pts, self.dataset.hopf_bank_params["n_total"])
                        print(f"In total, {self.dataset.solved_hopf_pts} hopf pts have been solved.\n")

                        # if reset_loss_w: #TODO
                        #     reset grad steps

                ## Parameter Scaling for Nonlinearity Curriculum
                not_pretraining = not(self.dataset.pretrain) and not(self.dataset.super_pretrain)
                if nonlin_scale:
                    if not_pretraining and ((epoch-total_pretrain_iters) % nl_scale_epoch_step) == 0 and epoch <= (epochs-nl_scale_epoch_post):
                        nl_perc = ((epoch - total_pretrain_iters)/((epochs - nl_scale_epoch_post) - total_pretrain_iters)) # instead of epoch/(epochs-post), this gives 1 step to switch from hopf to pde loss w/o changin dynamcis
                    elif epoch == (epochs - nl_scale_epoch_post) + 1: # jic epoch / nl_scale_epoch_step is not an integer
                        nl_perc = 1
                    self.dataset.dynamics.vary_nonlinearity(nl_perc)

                ## Learn
                for step, (model_input, gt) in enumerate(train_dataloader):
                    start_time = time.time()

                    ## Evaluate Sample with Learned Model
                    if self.timing: start_time_2 = time.time()
                    model_input = {key: value.cuda() for key, value in model_input.items()}
                    gt = {key: value.cuda() for key, value in gt.items()}
                    model_results = self.model({'coords': model_input['model_coords']})
                    if self.timing: print("Sample Evaluation took:", time.time() - start_time_2)

                    ## Pre-Loss Computation
                    if self.timing: start_time_2 = time.time()

                    ## Evaluate Model
                    results_coord = self.dataset.dynamics.input_to_coord(model_results['model_in'].detach())
                    state_times, states = results_coord[..., 0], results_coord[..., 1:]
                    values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1))
                    
                    ## Compute Gradients via Jacobian Backprop
                    if not fin_diff:
                        dvs = self.dataset.dynamics.io_to_dv(model_results['model_in'], model_results['model_out'].squeeze(dim=-1))
                        dvdt, dvdx = dvs[..., 0], dvs[..., 1:]
                    
                    ## Compute Gradients via Finite Diff
                    else:
                        # WAS TODO: move to fn like self.datasret.dynamics.io_to_dvfd(model_input['model_coords'], fd_weight_scales["delta_x"][fd_scale_i], fd_weight_scales["delta_t"][fd_scale_i])
                        delta_x_mat = fd_weight_scales["delta_x"][fd_scale_i] * torch.hstack((torch.zeros(self.dataset.dynamics.state_dim, 1), torch.eye(self.dataset.dynamics.state_dim))).cuda()
                        delta_t_mat = fd_weight_scales["delta_t"][fd_scale_i] * torch.hstack((torch.ones(1), torch.zeros(self.dataset.dynamics.state_dim))).unsqueeze(0).cuda()

                        coords_U = model_input['model_coords'].unsqueeze(-2) + delta_x_mat
                        coords_L = model_input['model_coords'].unsqueeze(-2) - delta_x_mat
                        coords_t = model_input['model_coords'] + delta_t_mat

                        model_results_U = self.model({'coords': coords_U})
                        model_results_L = self.model({'coords': coords_L})
                        model_results_t = self.model({'coords': coords_t})

                        values_U = self.dataset.dynamics.io_to_value(model_results_U['model_in'].detach(), model_results_U['model_out'].squeeze(dim=-1))
                        values_L = self.dataset.dynamics.io_to_value(model_results_L['model_in'].detach(), model_results_L['model_out'].squeeze(dim=-1))
                        values_t = self.dataset.dynamics.io_to_value(model_results_t['model_in'].detach(), model_results_t['model_out'].squeeze(dim=-1))

                        dvdx_fd_U = (values_U - values.unsqueeze(2).expand(-1,-1,self.dataset.dynamics.state_dim)) / fd_weight_scales["delta_x"][fd_scale_i]
                        dvdx_fd_L = (values.unsqueeze(2).expand(-1,-1,self.dataset.dynamics.state_dim) - values_L) / fd_weight_scales["delta_x"][fd_scale_i]
                        dvdt_fd   = (values_t - values) / fd_weight_scales["delta_t"][fd_scale_i]

                        dvdx_fd = 0.5 * (dvdx_fd_U + dvdx_fd_L)
                        diss_fd = fd_weight_scales["alpha"][fd_scale_i] * 0.5 * (dvdx_fd_U - dvdx_fd_L).sum(dim=-1)
                        # WAS TODO: what if instead of all this we just added a back-prop laplacian term?

                        dvdt, dvdx, diss = dvdt_fd, dvdx_fd, diss_fd

                        ## Diminish Viscosity for Better Approximation
                        if (epoch + 1 - self.total_pretrain_iters) % fd_scale_epoch_step == 0 and epoch + 1 > self.total_pretrain_iters and fd_scale_i < len(fd_weight_scales["alpha"])-1:
                            print(f'Finite Difference Learning, diminshing viscosity (a, dx, dt)=[{fd_weight_scales["alpha"][fd_scale_i]:2.2f}, {fd_weight_scales["delta_x"][fd_scale_i]:2.2f}, {fd_weight_scales["delta_t"][fd_scale_i]:2.2f}] -> [{fd_weight_scales["alpha"][fd_scale_i+1]:2.2f}, {fd_weight_scales["delta_x"][fd_scale_i+1]:2.2f}, {fd_weight_scales["delta_t"][fd_scale_i+1]:2.2f}]')
                            fd_scale_i += 1
                            
                    boundary_values = gt['boundary_values']
                    dirichlet_masks = gt['dirichlet_masks']

                    if self.dataset.dynamics.loss_type == 'brat_hjivi':
                        reach_values = gt['reach_values']
                        avoid_values = gt['avoid_values']

                    if self.timing: print("Pre-loss Computation took:", time.time() - start_time_2)

                    if self.dataset.memory_tracking:
                        print(f"Epoch {epoch} - init, torch.cuda.memory_allocated: {torch.cuda.memory_allocated()/1000000:2.2f} MB")
                        print(f"Epoch {epoch} - init, torch.cuda.memory_reserved:  {torch.cuda.memory_reserved()/1000000:2.2f} MB")
                        print()  

                    ## Compute Loss
                    if self.timing: start_time_2 = time.time()

                    ## Standard BRT
                    if self.dataset.dynamics.loss_type == 'brt_hjivi':
                        losses = loss_fn(states, values, dvdt, dvdx, boundary_values, dirichlet_masks, model_results['model_out'], diss=diss)
                    
                    ## Standard BRAT
                    elif self.dataset.dynamics.loss_type == 'brat_hjivi':
                        losses = loss_fn(states, values, dvdt, dvdx, boundary_values, reach_values, avoid_values, dirichlet_masks, model_results['model_out'], diss=diss)
                    
                    ## Multi-Objective (BRAT, BRAAT, BRRT)
                    elif self.dataset.dynamics.loss_type == 'mulob_hjivi':

                        if self.dataset.load_decomposed_models:
                            # note, decomposed values inferred on data wo lambda, and appropriately shifted if augmented for pde loss (but not ss losses)
                            
                            if not self.dataset.solve_grad:
                                decomposed_grads_1, decomposed_grads_2 = None, None
                                
                                with torch.inference_mode():
                                    load_coords = model_input['model_coords'] if not self.dataset.lambda_var else model_input['model_coords'][..., :-1]

                                    loaded_model_results_1 = self.dataset.loaded_model_1({'coords': load_coords})
                                    loaded_model_results_2 = self.dataset.loaded_model_2({'coords': load_coords})

                                    decomposed_values_1 = self.dataset.loaded_dynamics_1.io_to_value(loaded_model_results_1['model_in'], loaded_model_results_1['model_out'].squeeze(dim=-1)).detach()
                                    decomposed_values_2 = self.dataset.loaded_dynamics_2.io_to_value(loaded_model_results_2['model_in'], loaded_model_results_2['model_out'].squeeze(dim=-1)).detach()
                            
                            else:
                                load_coords = model_input['model_coords'] if not self.dataset.lambda_var else model_input['model_coords'][..., :-1]

                                loaded_model_results_1 = self.dataset.loaded_model_1({'coords': load_coords})
                                loaded_model_results_2 = self.dataset.loaded_model_2({'coords': load_coords})

                                decomposed_values_1 = self.dataset.loaded_dynamics_1.io_to_value(loaded_model_results_1['model_in'], loaded_model_results_1['model_out'].squeeze(dim=-1)).detach()
                                decomposed_values_2 = self.dataset.loaded_dynamics_2.io_to_value(loaded_model_results_2['model_in'], loaded_model_results_2['model_out'].squeeze(dim=-1)).detach()
                        
                                decomposed_grads_1 = self.dataset.loaded_dynamics_1.io_to_dv(loaded_model_results_1['model_in'], loaded_model_results_1['model_out'].squeeze(dim=-1)).detach() # NOTE: keeping time grad too now, add [..., 1:] otherwise
                                decomposed_grads_2 = self.dataset.loaded_dynamics_2.io_to_dv(loaded_model_results_2['model_in'], loaded_model_results_2['model_out'].squeeze(dim=-1)).detach() 

                            if hasattr(self.dataset.loaded_dynamics_2, 'avoid_only') and self.dataset.loaded_dynamics_2.avoid_only:
                                decomposed_values_2 = -1 * decomposed_values_2 # avoid_only defined positively in DR, so flip
                                if self.dataset.solve_grad:
                                    decomposed_grads_2[..., 1:] = -1 * decomposed_grads_2[..., 1:]

                        # elif self.dataset.use_gt_decomposed:
                        else:

                            if not hasattr(self.dataset, "V_DP"):
                                raise AssertionError("Must have ground truth solution if not loading decomposed solutions")
                            
                            decomposed_values_1, decomposed_values_2 = gt['gt_decomposed_values_1'], gt['gt_decomposed_values_2']

                            if self.dataset.solve_grad:
                                decomposed_grads_1, decomposed_grads_2 = gt['gt_decomposed_grads_1'], gt['gt_decomposed_grads_2']
                            else:
                                decomposed_grads_1, decomposed_grads_2 = None, None

                        bc_value_1, bc_value_2 = gt['bc_values_1'], gt['bc_values_2']
                        
                        # For fixed lambda slice supervision, infer model values on slices
                        if self.dataset.lam_slice_super:
                            
                            model_results_poslam = self.model({'coords': gt['model_coords_poslam']})
                            model_results_neglam = self.model({'coords': gt['model_coords_neglam']})

                            values_poslam = self.dataset.dynamics.io_to_value(model_results_poslam['model_in'].detach(), model_results_poslam['model_out'].squeeze(dim=-1))
                            values_neglam = self.dataset.dynamics.io_to_value(model_results_neglam['model_in'].detach(), model_results_neglam['model_out'].squeeze(dim=-1))
                            
                            dvs_poslam = self.dataset.dynamics.io_to_dv(model_results_poslam['model_in'], model_results_poslam['model_out'].squeeze(dim=-1))
                            dvs_neglam = self.dataset.dynamics.io_to_dv(model_results_neglam['model_in'], model_results_neglam['model_out'].squeeze(dim=-1))

                        else:
                            values_poslam, dvs_poslam, values_neglam, dvs_neglam = values, dvs, values, dvs

                        losses = loss_fn(states, values, dvs, boundary_values, dirichlet_masks, model_results['model_out'], 
                                        bc_value_1, bc_value_2, 
                                        decomposed_values_1, decomposed_values_2, 
                                        decomposed_grads_1, decomposed_grads_2,
                                        values_poslam, dvs_poslam, values_neglam, dvs_neglam)

                    ## Linear Supervision BRT (non-baseline)
                    elif hopf_loss != 'none':
                        hopf_values = gt['hopf_values']

                        if hopf_loss == 'lin_val_grad_diff':
                            hopf_grads = gt['hopf_grads']
                                            
                        ## Load from Reference Linear Model
                        if self.dataset.load_hopf_model:
                            with torch.inference_mode():
                                if not self.dataset.lambda_var:
                                    loaded_model_results = self.dataset.loaded_model({'coords': model_input['model_coords']})
                                    hopf_values = self.dataset.dynamics.io_to_value(loaded_model_results['model_in'], loaded_model_results['model_out'].squeeze(dim=-1)).detach()
                                else:
                                    if not self.dataset.zerolambda_LS or self.dataset.pretrain or self.dataset.super_pretrain or self.dataset.super_pretrain_counter == self.dataset.super_pretrain_iters:
                                        loaded_model_results = self.dataset.loaded_model({'coords': model_input['model_coords'][..., :-1]}) # remove lambda
                                        hopf_values = self.dataset.dynamics.io_to_value(model_input['model_coords'], loaded_model_results['model_out'].squeeze(dim=-1)).detach() 
                                    else: # use model coords with lambda=0 only for linear value
                                        loaded_model_results = self.dataset.loaded_model({'coords': gt['model_coords_hopf'][..., :-1]}) # remove lambda
                                        hopf_values = self.dataset.dynamics.io_to_value(gt['model_coords_hopf'], loaded_model_results['model_out'].squeeze(dim=-1)).detach() 

                        if self.dataset.load_hopf_model and hopf_loss == 'lin_val_grad_diff':
                            if not self.dataset.lambda_var:
                                loaded_model_results = self.dataset.loaded_model({'coords': model_input['model_coords']})
                                hopf_grads = self.dataset.dynamics.io_to_dv(loaded_model_results['model_in'], loaded_model_results['model_out'].squeeze(dim=-1))[..., 1:].detach()
                            
                            # If we want grads from a loaded linear model that does not have same dim as current model (eg lambda variation), need to use loaded model dynamics class
                            else:
                                if not self.dataset.zerolambda_LS or self.dataset.pretrain or self.dataset.super_pretrain or self.dataset.super_pretrain_counter == self.dataset.super_pretrain_iters:
                                    loaded_model_results = self.dataset.loaded_model({'coords': model_input['model_coords'][..., :-1]}) # remove lambda
                                    # loaded_results_w_lambda = torch.cat((loaded_model_results['model_in'], torch.zeros(1, self.dataset.numpoints, 1).cuda()), dim=2) # put lambda back just for next line (removed b4 loss)
                                    # hopf_grads = self.dataset.dynamics.io_to_dv(loaded_model_results['model_in'], loaded_model_results['model_out'].squeeze(dim=-1)).detach()
                                    hopf_grads = self.dataset.loaded_dynamics.io_to_dv(loaded_model_results['model_in'], loaded_model_results['model_out'].squeeze(dim=-1))[..., 1:].detach()
                                else:
                                    loaded_model_results = self.dataset.loaded_model({'coords': gt['model_coords_hopf'][..., :-1]}) # remove lambda
                                    hopf_grads = self.dataset.loaded_dynamics.io_to_dv(loaded_model_results['model_in'], loaded_model_results['model_out'].squeeze(dim=-1))[..., 1:].detach()    
                            
                        # if self.dataset.memory_tracking:
                        #     print(f"Epoch {epoch} - w/ losses, torch.cuda.memory_allocated: {torch.cuda.memory_allocated()/1000000:2.2f} MB")
                        #     print(f"Epoch {epoch} - w/ losses, torch.cuda.memory_reserved:  {torch.cuda.memory_reserved()/1000000:2.2f} MB")
                        #     print()

                        ## Seperate Hopf Coords (for the hopf loss to allow unrestricted sampling for PDE loss OR supervision on lambda=0 data only)
                        if (not self.dataset.use_bank or self.dataset.super_pretrain_counter == 0) and (not self.dataset.zerolambda_LS or (self.dataset.pretrain or self.dataset.super_pretrain or self.dataset.super_pretrain_counter == self.dataset.super_pretrain_iters)):
                            
                            learned_hopf_values = values
                            if hopf_loss == 'lin_val_grad_diff':
                                if not self.dataset.lambda_var:
                                    learned_hopf_grads = dvdx
                                else:
                                    learned_hopf_grads = dvdx[..., :-1] # remove lambda grad
                        else:
                            model_results_hopf = self.model({'coords': gt['model_coords_hopf']})
                            learned_hopf_values = self.dataset.dynamics.io_to_value(model_results_hopf['model_in'].detach(), model_results_hopf['model_out'].squeeze(dim=-1))
                            
                            if hopf_loss == 'lin_val_grad_diff':
                                if not self.dataset.lambda_var:
                                    learned_hopf_grads = self.dataset.dynamics.io_to_dv(model_results_hopf['model_in'], model_results_hopf['model_out'].squeeze(dim=-1))[..., 1:]
                                else:
                                    learned_hopf_grads = self.dataset.dynamics.io_to_dv(model_results_hopf['model_in'], model_results_hopf['model_out'].squeeze(dim=-1))[..., 1:-1] # remove lambda grad

                        if hopf_loss == 'lin_val_grad_diff':
                            losses = loss_fn(states, values, dvdt, dvdx, boundary_values, dirichlet_masks, model_results['model_out'], hopf_values, learned_hopf_values, hopf_grads, learned_hopf_grads, epoch, state_times)
                        else:
                            losses = loss_fn(states, values, dvdt, dvdx, boundary_values, dirichlet_masks, model_results['model_out'], hopf_values, learned_hopf_values, epoch, state_times)
                            # losses = loss_fn_baseline(states, values, dvdt, dvdx, boundary_values, dirichlet_masks, model_results['model_out'])
                            # print("\nUsing the loaded hopf values")                    
                    else:
                        raise NotImplementedError
                    
                    if self.dataset.memory_tracking:
                        print(f"Epoch {epoch} - after losses, torch.cuda.memory_allocated: {torch.cuda.memory_allocated()/1000000:2.2f} MB")
                        print(f"Epoch {epoch} - after losses, torch.cuda.memory_reserved:  {torch.cuda.memory_reserved()/1000000:2.2f} MB")
                        print()

                    if self.timing: print("Loss Computation took:", time.time() - start_time_2)

                    ## Compute & Record Temporal Loss Quartiles #FIXME doesn't work for baseline
                    if record_temporal_loss:
                        losses_t = {}
                        temporal_loss_times = [0., 0.25, 0.5, 0.75, 1.] 
                        for ti in range(len(temporal_loss_times)-1):
                            tp, tm = temporal_loss_times[ti + 1], temporal_loss_times[ti]
                            t_ix = (state_times < tp) * (state_times >= tm)
                            state_times_t, states_t, values_t, dvs_t, boundary_values_t = state_times[t_ix, ...].unsqueeze(0), states[t_ix, ...].unsqueeze(0), values[t_ix].unsqueeze(0), dvs[t_ix, ...].unsqueeze(0), boundary_values[t_ix].unsqueeze(0)
                            dirichlet_masks_t, model_results_t, hopf_values_t, learned_hopf_values_t = dirichlet_masks[t_ix].unsqueeze(0), model_results['model_out'][t_ix].unsqueeze(0), hopf_values[t_ix].unsqueeze(0), learned_hopf_values[t_ix].unsqueeze(0)
                            if hopf_loss == 'lin_val_diff':
                                losses_t[str(tp)] = loss_fn(states_t, values_t, dvs_t[..., 0], dvs_t[..., 1:], boundary_values_t, dirichlet_masks_t, model_results_t, hopf_values_t, learned_hopf_values_t, epoch, state_times_t)
                            elif hopf_loss == 'lin_val_grad_diff':
                                hopf_grads_t, learned_grads_t = hopf_grads[t_ix].unsqueeze(0), learned_hopf_grads[t_ix].unsqueeze(0)
                                losses_t[str(tp)] = loss_fn(states_t, values_t, dvs_t[..., 0], dvs_t[..., 1:], boundary_values_t, dirichlet_masks_t, model_results_t, hopf_values_t, learned_hopf_values_t, hopf_grads_t, learned_grads_t, epoch, state_times_t)

                    ## Switch Optimizers/Rates (after Hopf Pretraining)
                    if self.timing: start_time_2 = time.time()
                    if dual_lr and not(self.dataset.super_pretrain) and self.dataset.super_pretrained:
                        optim = optim_std
                        lr_scheduler = lr_scheduler_std
                    if self.timing: print("Loss Scheduler took:", time.time() - start_time_2)
                    
                    ## Decay Hopf Loss(es)
                    if hopf_loss_decay and hopf_loss != 'none': #                         
                        if epoch >= total_pretrain_iters or hopf_loss_decay_early:
                            if hopf_loss_decay_type == 'exponential' and epoch > total_pretrain_iters or hopf_loss_decay_early:
                                loss_weights['hopf'] = hopf_loss_decay_w * loss_weights['hopf']
                            elif hopf_loss_decay_type == 'linear':
                                loss_weights['hopf'] = 1 - hopf_loss_decay_w * (epoch - total_pretrain_iters)/(epochs - 1 - total_pretrain_iters)
                            elif hopf_loss_decay_type == 'negative_exponential' and epoch > total_pretrain_iters:
                                loss_weights['hopf'] = 1 - ((1 - loss_weights['hopf']) / hopf_loss_decay_w)
                            elif hopf_loss_decay_type not in ['exponential', 'linear', 'negative_exponential']:
                                raise NotImplementedError
                        loss_weights['hopf'] = min(max(loss_weights['hopf'], 0.), 1.)
                        
                        if hopf_loss == 'lin_val_grad_diff':
                            loss_weights['hopf_grad'] = loss_weights['hopf'] 

                        ## Incrementally Introduce Differential Constraint Loss (After All Pretraining)
                        if diff_con_loss_incr and epoch >= total_pretrain_iters:
                            loss_weights['diff_constraint_hom'] = 1 - loss_weights['hopf']

                    if self.dataset.memory_tracking:
                        print(f"Epoch {epoch} - after weight scheduler, torch.cuda.memory_allocated: {torch.cuda.memory_allocated()/1000000:2.2f} MB")
                        print(f"Epoch {epoch} - after weight scheduler, torch.cuda.memory_reserved:  {torch.cuda.memory_reserved()/1000000:2.2f} MB")
                        print()

                    ## Combine Losses
                    if self.timing: start_time_2 = time.time()
                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean() ## TODO: why did the prev authors put this here?
                        writer.add_scalar(loss_name, single_loss, total_steps)
                        train_loss += loss_weights[loss_name] * single_loss

                    train_losses.append(train_loss.item())
                    writer.add_scalar("total_train_loss", train_loss, total_steps)
                    if self.timing: print("Loss Combination took:", time.time() - start_time_2)

                    ## Save Checkpoint
                    if not total_steps % steps_til_summary:
                        torch.save(self.model.state_dict(),
                                os.path.join(checkpoints_dir, 'model_current.pth'))
                        # summary_fn(model, model_input, gt, model_output, writer, total_steps)

                    if self.dataset.memory_tracking:
                        print(f"Epoch {epoch} - loss combo, torch.cuda.memory_allocated: {torch.cuda.memory_allocated()/1000000:2.2f} MB")
                        print(f"Epoch {epoch} - loss combo, torch.cuda.memory_reserved:  {torch.cuda.memory_reserved()/1000000:2.2f} MB")
                        print()

                    ## Take Gradient Step
                    if not use_lbfgs:
                        if self.timing: start_time_2 = time.time()
                        optim.zero_grad()
                        train_loss.backward()
                        if self.timing: print("Grad Comp took:", time.time() - start_time_2)

                        if self.timing: start_time_2 = time.time()
                        if clip_grad:
                            if isinstance(clip_grad, bool):
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.)
                            else:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_grad)
                        if self.timing: print("Grad Clip took:", time.time() - start_time_2)

                        if self.timing: start_time_2 = time.time()
                        optim.step()
                        lr_scheduler.step()
                        if self.timing: print("Grad/Sched step took:", time.time() - start_time_2)

                    if self.dataset.memory_tracking:
                        print(f"Epoch {epoch} - grad step, torch.cuda.memory_allocated: {torch.cuda.memory_allocated()/1000000:2.2f} MB")
                        print(f"Epoch {epoch} - grad step, torch.cuda.memory_reserved:  {torch.cuda.memory_reserved()/1000000:2.2f} MB")
                        print()

                    ## Record Data Summary
                    if not total_steps % steps_til_summary:
                        iter_time = time.time() - start_time

                        if self.dataset.record_gt_metrics:   
                            JIp, FIp, FEp, Vmse, DVXmse = self.compute_gt_metrics()
                            JIp_s = smoothing_factor * JIp + (1 - smoothing_factor) * JIp_s
                            JIp_s_max = max(JIp_s, JIp_s_max)
                        
                            tqdm.write("Epoch %d, Total loss %0.6f, MSE %3.5e, IOU %0.6f, iter time %0.6f" % (epoch, train_loss, Vmse, JIp, iter_time))
                        else:
                            tqdm.write("Epoch %d, Total loss %0.6f, iter time %0.6f" % (epoch, train_loss, iter_time))
                        
                        if self.dataset.memory_tracking:
                            print(f"Epoch {epoch}-10, torch.cuda.memory_allocated: {torch.cuda.memory_allocated()/1000000:2.2f} MB")
                            print(f"Epoch {epoch}-10, torch.cuda.memory_reserved:  {torch.cuda.memory_reserved()/1000000:2.2f} MB")
                            print()

                        if self.use_wandb:
                            log_dict = {
                                'step': epoch,
                                'train_loss': train_loss,
                                'iter_time_sec': iter_time}
                            
                            for loss_name, loss in losses.items():
                                log_dict[loss_name + "_loss"] = loss

                            if nonlin_scale and not(self.dataset.pretrain) and not(self.dataset.super_pretrain):
                                log_dict["Nonlinearity Scale"] = nl_perc

                            if self.dataset.record_gt_metrics:

                                log_dict["Jaccard Index over Time"] = JIp
                                log_dict["Smooth Jaccard Index over Time"] = JIp_s
                                log_dict["Max Smooth Jaccard Index over Time"] = JIp_s_max
                                log_dict["Falsely Included percent over Time"] = FIp
                                log_dict["Falsely Excluded percent over Time"] = FEp
                                log_dict["Mean Absolute Spatial Gradient"] = torch.abs(dvdx).sum() / (self.dataset.numpoints * self.N)
                                log_dict["Mean Squared Error of Value"] = Vmse
                                if self.dataset.solve_grad:
                                    log_dict["Mean Squared Error of Spatial Gradient"] = DVXmse

                            if hopf_loss_decay and epoch >= total_pretrain_iters:
                                log_dict['hopf_weight'] = loss_weights['hopf']
                                log_dict['pde_weight'] = loss_weights['diff_constraint_hom']

                            if self.dataset.solve_hopf and self.dataset.record_gt_metrics:
                                log_dict['Hopf Value MSE'] = self.dataset.hjpool.alg_iter_MSE
                                log_dict['Hopf Gradient MSE'] = self.dataset.hjpool.alg_iter_grad_MSE
                                log_dict['Hopf Compute Time (ppt)'] = self.dataset.hjpool.alg_iter_comp_time

                            if record_temporal_loss:
                                for ti in range(len(temporal_loss_times)-1):
                                    tps = str(temporal_loss_times[ti+1])
                                    for loss_name, loss in losses_t[tps].items():
                                        log_dict[loss_name + "_loss_t" + tps] = loss

                            wandb.log(log_dict)

                    if self.dataset.memory_tracking:
                        print(f"Epoch {epoch} - end, torch.cuda.memory_allocated: {torch.cuda.memory_allocated()/1000000:2.2f} MB")
                        print(f"Epoch {epoch} - end, torch.cuda.memory_reserved:  {torch.cuda.memory_reserved()/1000000:2.2f} MB")
                        print()

                    pbar.update(1)
                    total_steps += 1

                ## cost-supervised learning (CSL) used to be here (removed because not using)

                if self.timing: start_time_2 = time.time()
                if not (epoch+1) % epochs_til_checkpoint:
                    # Saving the optimizer state is important to produce consistent results
                    checkpoint = { 
                        'epoch': epoch+1,
                        'model': self.model.state_dict(),
                        'optimizer': optim.state_dict()}
                    torch.save(checkpoint,
                        os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % (epoch+1)))
                    torch.save(checkpoint,
                        os.path.join(checkpoints_dir, 'model_final.pth'))
                    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % (epoch+1)),
                        np.array(train_losses))
                    self.validate(
                        epoch=epoch+1, save_path=os.path.join(checkpoints_dir, 'BRS_validation_plot_epoch_%04d.png' % (epoch+1)),
                        # epoch=epoch+1, save_path=os.path.join(checkpoints_dir, 'BRS_validation_plot.png'), # overwriting to save data
                        x_resolution = val_x_resolution, y_resolution = val_y_resolution, z_resolution=val_z_resolution, time_resolution=val_time_resolution)
                if self.timing: print("Checkpointing took:", time.time() - start_time_2)

        # print("\n PROFILER RESULTS \n")
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        # print()

        if was_eval:
            self.model.eval()
            self.model.requires_grad_(False)

    def test(self, current_time, last_checkpoint, checkpoint_dt, dt, num_scenarios, num_violations, set_type, control_type, data_step, val_x_resolution=None, val_y_resolution=None, val_z_resolution=None, val_time_resolution=None, checkpoint_toload=None):
        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)

        testing_dir = os.path.join(self.experiment_dir, 'testing_%s' % current_time.strftime('%m_%d_%Y_%H_%M'))
        if os.path.exists(testing_dir):
            overwrite = input("The testing directory %s already exists. Overwrite? (y/n)"%testing_dir)
            if not (overwrite == 'y'):
                print('Exiting.')
                quit()
            shutil.rmtree(testing_dir)
        os.makedirs(testing_dir)

        if checkpoint_toload is None:
            raise NotImplementedError
            print('running cross-checkpoint testing')

            # for i in tqdm(range(sidelen), desc='Checkpoint'):
            #     self._load_checkpoint(epoch=checkpoints[i])
            # raise NotImplementedError

        else:
            print('running specific-checkpoint testing')
            self._load_checkpoint(checkpoint_toload)

            # model = self.model
            # dataset = self.dataset
            # dynamics = dataset.dynamics
            # raise NotImplementedError

            checkpoint = str(checkpoint_toload) if not checkpoint_toload == -1 else 'final'
            self.validate(epoch=0, save_path=os.path.join(testing_dir, f'BRS_validation_plot_epoch_{checkpoint}.png'), # overwriting to save data
                        x_resolution = val_x_resolution, y_resolution = val_y_resolution, z_resolution=val_z_resolution, time_resolution=val_time_resolution)

        if was_training:
            self.model.train()
            self.model.requires_grad_(True)

class DeepReach(Experiment):
    def init_special(self):
        pass

class DeepReachHopf(Experiment):
    def init_special(self, N=2, timing=False):
        self.N = N
        self.timing = timing
        if N == 2:
            if hasattr(self.dataset.dynamics, 'name') and self.dataset.dynamics.name in ["Conveyor","Canoe"]:
                if not self.dataset.lambda_var:
                    self.validate = self.validate2D_mulob
                else:
                    self.validate = self.validate2Dlambda_mulob
            else:
                self.validate = self.validate2D
        elif N > 2:
            self.validate = self.validateND
            if self.dataset.lambda_var and self.dataset.dynamics.name == "LessLinear":
                self.validate = self.validateNDlambda
            elif self.dataset.lambda_var and self.dataset.dynamics.name in ["Conveyor","Canoe"]:
                self.validate = self.validateNDlambda_mulob

        pass        
    
    def validate2D(self, epoch, save_path, x_resolution, y_resolution, z_resolution, time_resolution):
        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)

        plot_config = self.dataset.dynamics.plot_config()

        state_test_range = self.dataset.dynamics.state_test_range()
        x_min, x_max = state_test_range[plot_config['x_axis_idx']]
        y_min, y_max = state_test_range[plot_config['y_axis_idx']]
        # z_min, z_max = state_test_range[plot_config['z_axis_idx']]

        times = torch.linspace(0, self.dataset.tMax, time_resolution)
        xs = torch.linspace(x_min, x_max, x_resolution)
        ys = torch.linspace(y_min, y_max, y_resolution)
        # zs = torch.linspace(z_min, z_max, z_resolution)
        xys = torch.cartesian_prod(xs, ys)
        
        fig = plt.figure(figsize=(5*len(times), 5*1))
        for i in range(len(times)):
            # for j in range(len(zs)):
            j = 0
            coords = torch.zeros(x_resolution*y_resolution, self.dataset.dynamics.state_dim + 1)
            coords[:, 0] = times[i]
            coords[:, 1:] = torch.tensor(plot_config['state_slices'])
            coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
            coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
            # coords[:, 1 + plot_config['z_axis_idx']] = zs[j]

            with torch.no_grad():
                model_results = self.model({'coords': coords.cuda()})
                values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach())
            
            ax = fig.add_subplot(1, len(times), (j+1) + i)
            ax.set_title('t = %0.2f' % (times[i])) #, plot_config['state_labels'][plot_config['z_axis_idx']], zs[j]))
            s = ax.imshow(1*(values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T <= 0), cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(s, cax=cax) 

        fig.savefig(save_path)
        if self.use_wandb:
            wandb.log({
                'step': epoch,
                'val_plot': wandb.Image(fig),
            })
        plt.close()

        # if self.dataset.record_gt_metrics:
        #     self.plot_set_metrics_eachtime(epoch, times)
        #     if self.use_wandb:
        #         wandb.log({'Time vs. Epoch vs. Set Accuracy compared to DP': wandb.Image(self.t_ep_acc_fig),})
        #     plt.close()

        if was_training:
            self.model.train()
            self.model.requires_grad_(True)

    def validate2D_mulob(self, epoch, save_path, x_resolution, y_resolution, z_resolution, time_resolution, testing=False):
        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)

        solve_times = torch.linspace(0, self.dataset.tMax, time_resolution)
        
        fig = plt.figure(figsize=(5*len(solve_times), 5*1))
        for i in range(len(solve_times)):
            j = 0

            states = self.dataset.model_states_grid
            times = torch.full((self.dataset.n_grid_pts_2d, 1), solve_times[i]).cuda()
            coords = torch.cat((times, states), dim=1) 
            n_grid_len = self.dataset.X1g.size()[0]

            values_gt = self.dataset.V_DP(self.dataset.dynamics.input_to_coord(coords).cpu().t()).reshape(n_grid_len, n_grid_len)

            states_scaled = self.dataset.dynamics.input_to_coord(coords)[:, 1:]

            if self.dataset.dynamics.name == "Conveyor":
                values_bc_1 = self.dataset.dynamics.reach_fn(states_scaled, times).reshape(n_grid_len, n_grid_len).cpu()
                values_bc_2 = self.dataset.dynamics.avoid_fn(states_scaled, times).reshape(n_grid_len, n_grid_len).cpu()
                values_bc_1_color = "blue"
                values_bc_2_color = "red"
                xlims, ylims = (-2.5, 0.5), (-1.5, 1.5)
            elif self.dataset.dynamics.name == "Canoe":
                values_bc_1 = self.dataset.dynamics.reach_fn_1(states_scaled, times).reshape(n_grid_len, n_grid_len).cpu()
                values_bc_2 = self.dataset.dynamics.reach_fn_2(states_scaled, times).reshape(n_grid_len, n_grid_len).cpu()
                values_bc_1_color = "cyan"
                values_bc_2_color = "blue"
                xlims, ylims = (-1.25, 1.25), (-0.5, 2.)

            with torch.no_grad():
                model_results = self.model({'coords': coords.cuda()})
                values_learned = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach()).reshape(n_grid_len, n_grid_len).cpu()
            
            ax = fig.add_subplot(1, len(solve_times), (j+1) + i)
            ax.set_title('t = %0.2f' % (solve_times[i])) #, plot_config['state_labels'][plot_config['z_axis_idx']], zs[j]))
            
            cmap_name = "RdBu_r"
            # vmin, vmax = -0.075, 0.075
            # vmin, vmax = -0.5, 0.5
            vmax = 0.5
            vmin = -0.5 if self.dataset.dynamics.name == "Conveyor" else -self.dataset.dynamics.goalR_2d
            levels = np.linspace(vmin, vmax)
            n_bins_high = round(256 * vmax/(vmax - vmin))
            offset=0
            scaled_colors = np.vstack((matplotlib.colormaps[cmap_name](np.linspace(0., 0.4, 256-n_bins_high+offset)), matplotlib.colormaps[cmap_name](np.linspace(0.6, 1., n_bins_high-offset))))
            RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', scaled_colors)
            
            plot_values = values_learned if not testing else values_gt

            ax.contourf(self.dataset.X1g * self.dataset.dynamics.state_var[0] + self.dataset.dynamics.state_mean[0], 
                    self.dataset.X2g * self.dataset.dynamics.state_var[1] + self.dataset.dynamics.state_mean[1],
                    plot_values,
                    levels=levels, extend="both", cmap=RdWhBl_vscaled)
            
            if values_gt.min() < 0. < values_gt.max():
                ax.contour(self.dataset.X1g * self.dataset.dynamics.state_var[0] + self.dataset.dynamics.state_mean[0], 
                            self.dataset.X2g * self.dataset.dynamics.state_var[1] + self.dataset.dynamics.state_mean[1],
                            values_gt, 
                            levels=0, colors="black", linewidths=3, alpha=0.7)
            
            if values_bc_1.min() < 0. < values_bc_1.max():
                ax.contour(self.dataset.X1g * self.dataset.dynamics.state_var[0] + self.dataset.dynamics.state_mean[0], 
                            self.dataset.X2g * self.dataset.dynamics.state_var[1] + self.dataset.dynamics.state_mean[1],
                            values_bc_1, 
                            levels=0, colors=values_bc_1_color, linewidths=4, alpha=0.7)
                
            if values_bc_2.min() < 0. < values_bc_2.max():
                ax.contour(self.dataset.X1g * self.dataset.dynamics.state_var[0] + self.dataset.dynamics.state_mean[0], 
                            self.dataset.X2g * self.dataset.dynamics.state_var[1] + self.dataset.dynamics.state_mean[1],
                            values_bc_2, 
                            levels=0, colors=values_bc_2_color, linewidths=4, alpha=0.7)
            
            ax.set_xlim(xlims)    
            ax.set_ylim(ylims)
            ax.set_aspect('equal')

        if testing:
            plt.savefig(f"plots/mulob_tests/{self.dataset.dynamics.name}_test_training_plot_gt.png")
        
        else:
            fig.savefig(save_path)
            # if was_training: fig.savefig('BRS_final_'+ '_'.join(save_path.split('_')[1:-1]) + '.png')
            if self.use_wandb:
                wandb.log({
                    'step': epoch,
                    'val_plot': wandb.Image(fig),
                })
        plt.close()

        # if self.dataset.record_gt_metrics:
        #     self.plot_set_metrics_eachtime(epoch, times)
        #     if self.use_wandb:
        #         wandb.log({'Time vs. Epoch vs. Set Accuracy compared to DP': wandb.Image(self.t_ep_acc_fig),})
        #     plt.close()

        if was_training:
            self.model.train()
            self.model.requires_grad_(True)

    def validate2Dlambda_mulob(self, epoch, save_path, x_resolution, y_resolution, z_resolution, time_resolution, testing=False):
        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)

        solve_times = torch.linspace(0, self.dataset.tMax, time_resolution)
        if self.dataset.dynamics.name == "Conveyor":
            if self.dataset.dynamics.avoid_type == "ball":
                lambda_vals = torch.tensor([1., 0.1, 0., -0.3, -1.])
            else:
                lambda_vals = torch.tensor([1., 0.05, 0., -0.2, -1.])
        else:
            lambda_vals = torch.tensor([1., 0.2, 0., -0.2, -1.])
        
        cmap_name = "RdBu_r"
        vmax = 0.5
        vmin = -0.5 if self.dataset.dynamics.name == "Conveyor" else -self.dataset.dynamics.goalR_2d
        levels = np.linspace(vmin, vmax)
        n_bins_high = round(256 * vmax/(vmax - vmin))
        offset = 0
        scaled_colors = np.vstack((matplotlib.colormaps[cmap_name](np.linspace(0., 0.4, 256-n_bins_high+offset)), matplotlib.colormaps[cmap_name](np.linspace(0.6, 1., n_bins_high-offset))))
        RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', scaled_colors)
        
        fig = plt.figure(figsize=(5*len(solve_times), 5*len(lambda_vals)))
        for i in range(len(solve_times)):
            for j in range(len(lambda_vals)):

                states_input = self.dataset.model_states_grid
                times = torch.full((self.dataset.n_grid_pts_2d, 1), solve_times[i]).cuda()
                input = torch.cat((times, states_input), dim=1) 
                input[..., -1] = lambda_vals[j]/self.dataset.dynamics.state_var[-1] + 0*input[..., -1] # certain lambda
                
                coords = self.dataset.dynamics.input_to_coord(input)
                states = coords[:, 1:]
                n_grid_len = self.dataset.X1g.size()[0]
                
                if lambda_vals[j] == 1.:
                    values_gt = self.dataset.V_DP_1(coords[..., :-1].cpu().t()).reshape(n_grid_len, n_grid_len)
                elif lambda_vals[j] == -1.:
                    values_gt = self.dataset.V_DP_2(coords[..., :-1].cpu().t()).reshape(n_grid_len, n_grid_len)
                elif lambda_vals[j] == 0.:
                    values_gt = self.dataset.V_DP(coords[..., :-1].cpu().t()).reshape(n_grid_len, n_grid_len)

                if self.dataset.dynamics.name == "Conveyor":
                    values_bc_1 = self.dataset.dynamics.reach_fn(states, times).reshape(n_grid_len, n_grid_len).cpu()
                    values_bc_2 = self.dataset.dynamics.avoid_fn(states, times).reshape(n_grid_len, n_grid_len).cpu()
                    values_bc_1_color = "blue"
                    values_bc_2_color = "red"
                    xlims, ylims = (-2.5, 0.5), (-1.5, 1.5)

                elif self.dataset.dynamics.name == "Canoe":
                    values_bc_1 = self.dataset.dynamics.reach_fn_1(states, times).reshape(n_grid_len, n_grid_len).cpu()
                    values_bc_2 = self.dataset.dynamics.reach_fn_2(states, times).reshape(n_grid_len, n_grid_len).cpu()
                    values_bc_1_color = "cyan"
                    values_bc_2_color = "blue"
                    xlims, ylims = (-1.25, 1.25), (-0.5, 2.)

                with torch.no_grad():
                    model_results = self.model({'coords': input.clone().cuda()}).copy()
                    values_learned = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach()).reshape(n_grid_len, n_grid_len).cpu()
                    
                    if testing:
                        if j == 0:
                            model_results_1 = self.dataset.loaded_model_1({'coords': input[..., :-1].clone().cuda()}).copy()
                            values_learned = self.dataset.loaded_dynamics_1.io_to_value(model_results_1['model_in'].detach(), model_results_1['model_out'].squeeze(dim=-1).detach()).detach().reshape(n_grid_len, n_grid_len).cpu()
                        elif j == 4:
                            model_results_2 = self.dataset.loaded_model_2({'coords': input[..., :-1].cuda()}).copy()
                            values_learned = self.dataset.loaded_dynamics_2.io_to_value(model_results_2['model_in'].detach(), model_results_2['model_out'].squeeze(dim=-1).detach()).detach().reshape(n_grid_len, n_grid_len).cpu()
                    
                ax = fig.add_subplot(len(lambda_vals), len(solve_times), j*len(lambda_vals) + i + 1)
                ax.set_title(f"t = {solve_times[i]:0.1f}, lambda = {lambda_vals[j]:1.1f}")
                
                plot_values = values_learned if not testing else values_gt

                ax.contourf(self.dataset.X1g * self.dataset.dynamics.state_var[0] + self.dataset.dynamics.state_mean[0], 
                        self.dataset.X2g * self.dataset.dynamics.state_var[1] + self.dataset.dynamics.state_mean[1],
                        plot_values,
                        levels=levels, extend="both", cmap=RdWhBl_vscaled)
                
                if (abs(lambda_vals[j]) in [0., 1.]) and values_gt.min() < 0. < values_gt.max():
                    ax.contour(self.dataset.X1g * self.dataset.dynamics.state_var[0] + self.dataset.dynamics.state_mean[0], 
                                self.dataset.X2g * self.dataset.dynamics.state_var[1] + self.dataset.dynamics.state_mean[1],
                                values_gt, 
                                levels=0, colors="black", linewidths=3, alpha=0.7)
                
                if values_bc_1.min() < 0. < values_bc_1.max():
                    ax.contour(self.dataset.X1g * self.dataset.dynamics.state_var[0] + self.dataset.dynamics.state_mean[0], 
                                self.dataset.X2g * self.dataset.dynamics.state_var[1] + self.dataset.dynamics.state_mean[1],
                                values_bc_1, 
                                levels=0, colors=values_bc_1_color, linewidths=4, alpha=0.7)
                    
                if values_bc_2.min() < 0. < values_bc_2.max():
                    ax.contour(self.dataset.X1g * self.dataset.dynamics.state_var[0] + self.dataset.dynamics.state_mean[0], 
                                self.dataset.X2g * self.dataset.dynamics.state_var[1] + self.dataset.dynamics.state_mean[1],
                                values_bc_2, 
                                levels=0, colors=values_bc_2_color, linewidths=4, alpha=0.7)
                
                ax.set_xlim(xlims)    
                ax.set_ylim(ylims)
                ax.set_aspect('equal')

        if testing:
            plt.savefig(f"plots/mulob_tests/{self.dataset.dynamics.name}_{self.dataset.dynamics.avoid_type}_lambda_test_training_plot_gt.png")
        
        else:
            fig.savefig(save_path)
            # if was_training: fig.savefig('BRS_final_'+ '_'.join(save_path.split('_')[1:-1]) + '.png')
            if self.use_wandb:
                wandb.log({
                    'step': epoch,
                    'val_plot': wandb.Image(fig),
                })
        plt.close()

        # if self.dataset.record_gt_metrics:
        #     self.plot_set_metrics_eachtime(epoch, times)
        #     if self.use_wandb:
        #         wandb.log({'Time vs. Epoch vs. Set Accuracy compared to DP': wandb.Image(self.t_ep_acc_fig),})
        #     plt.close()

        if was_training:
            self.model.train()
            self.model.requires_grad_(True)
    
    def validateND(self, epoch, save_path, x_resolution, y_resolution, z_resolution, time_resolution, plot_value=True):
        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)

        plot_config = self.dataset.dynamics.plot_config()

        state_test_range = self.dataset.dynamics.state_test_range()
        x_min, x_max = state_test_range[plot_config['x_axis_idx']]
        y_min, y_max = state_test_range[plot_config['y_axis_idx']]
        # z_min, z_max = state_test_range[plot_config['z_axis_idx']]

        times = torch.linspace(0, self.dataset.tMax, time_resolution)
        xs = torch.linspace(x_min, x_max, x_resolution)
        ys = torch.linspace(y_min, y_max, y_resolution)
        # zs = torch.linspace(z_min, z_max, z_resolution)
        xys = torch.cartesian_prod(xs, ys)
        Xg, Yg = torch.meshgrid(xs, ys)
        
        ## Plot Set and Value Fn
        
        # fig_set = plt.figure(figsize=(5*len(times), 2*5*1))
        # fig_val = plt.figure(figsize=(5*len(times), 2*5*1), facecolor='white')

        fig = plt.figure(figsize=(5*len(times), 2*5*1), facecolor='white')
        
        plt.rcParams['text.usetex'] = False

        # for i in range(3*len(times)):
        for i in range(2*len(times)):
            
            # ax_set = fig_set.add_subplot(2, len(times), 1+i)
            # ax_val = fig_val.add_subplot(2, len(times), 1+i, projection='3d')
            # ax_set.set_title('t = %0.2f' % (times[i % len(times)]))
            # ax_val.set_title('t = %0.2f' % (times[i % len(times)]))

            if i >= len(times):
                ax = fig.add_subplot(2, len(times), 1+i)
            else:
                ax = fig.add_subplot(2, len(times), 1+i, projection='3d')
            ax.set_title(r"t =" + "%0.2f" % (times[i % len(times)]))

            ## Define Grid Slice to Plot

            coords = torch.zeros(x_resolution*y_resolution, self.dataset.dynamics.state_dim + 1)
            coords[:, 0] = times[i % len(times)]
            coords[:, 1:] = torch.tensor(plot_config['state_slices']) # initialized to zero (nothing else to set!)

            # if i < len(times): # xN - xi plane
            #     ax_set.set_xlabel("xN"); ax_set.set_ylabel("xi")
            #     ax_val.set_xlabel("xN"); ax_val.set_ylabel("xi")
            #     coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
            #     coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]

            # elif i < 2*len(times): # xi - xj plane
            #     ax_set.set_xlabel("xi"); ax_set.set_ylabel("xj")
            #     ax_val.set_xlabel("xi"); ax_val.set_ylabel("xj")
            #     coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 0]
            #     coords[:, 1 + plot_config['z_axis_idx']] = xys[:, 1]

            # xN - (xi = xj) plane
            if i >= len(times):
                pad_label = 0
                ax.set_xlabel(r"$x_N$", fontsize=12, labelpad=pad_label); ax.set_ylabel(r"$x_i = x_j$", fontsize=12, labelpad=pad_label)
                ax.set_xticks([-1, 1])
                ax.set_xticklabels([r'$-1$', r'$1$'])
                ax.set_yticks([-1, 1])
                ax.set_yticklabels([r'$-1$', r'$1$'])

                ax_pad = 0
                ax.xaxis.set_tick_params(pad=ax_pad)
                ax.yaxis.set_tick_params(pad=ax_pad)

            else:
                pad_label = 6
                ax.set_xlabel(r"$x_N$", fontsize=12, labelpad=pad_label); 
                ax.set_ylabel(r"$x_i = x_j$", fontsize=12, labelpad=pad_label); 
                ax.set_zlabel(r"$V$", fontsize=12, labelpad=10) #, labelpad=pad_label)
                ax.set_xticks([-1, 0, 1])
                ax.set_xticklabels([r'$-1$', r'$0$', r'$1$'])
                ax.set_yticks([-1, 0, 1])
                ax.set_yticklabels([r'$-1$', r'$0$', r'$1$'])
                ax.set_zticks([])
                ax.zaxis.label.set_position((-0.1, 0.5))
                
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                
                ax_pad = 0
                ax.xaxis.set_tick_params(pad=ax_pad)
                ax.yaxis.set_tick_params(pad=ax_pad)
                ax.zaxis.set_tick_params(pad=ax_pad)

            coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
            coords[:, 2:] = (xys[:, 1] * torch.ones(self.N-1, xys.size()[0])).t()

            with torch.no_grad():
                model_results = self.model({'coords': self.dataset.dynamics.coord_to_input(coords.cuda())})
                values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach())
            
            learned_value = values.detach().cpu().numpy().reshape(x_resolution, y_resolution)

            n_grid_plane_pts = int(self.dataset.n_grid_pts/3)
            n_grid_len = int(n_grid_plane_pts ** 0.5)
            pix_start = (i // len(times)) * n_grid_plane_pts
            tix_start = (i % len(times)) * self.dataset.n_grid_pts
            # ix = pix_start + tix_start # plane and grid no longer synced
            ix = tix_start
            Vgt = self.dataset.values_DP_grid[ix:ix+n_grid_plane_pts].reshape(n_grid_len, n_grid_len).cpu()

            ## Make Value-Based Colormap
            # cmap_name = "coolwarm"
            cmap_name = "RdBu"

            if learned_value.min() > 0:
                # RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', [(1,1,1), (0.5,0.5,1), (0,0,1), (0,0,1)])
                scaled_colors = np.vstack((matplotlib.colormaps[cmap_name](np.linspace(0.6, 1., 256))))
                RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', scaled_colors)

            elif learned_value.max() < 0:
                # RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', [(1,0,0), (1,0,0), (1,0.5,0.5), (1,1,1)])
                scaled_colors = np.vstack((matplotlib.colormaps[cmap_name](np.linspace(0., 0.4, 256))))
                RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', scaled_colors)

            else:
                # n_bins_high = int(256 * (learned_value.max()/(learned_value.max() - learned_value.min())) // 1)
                n_bins_high = round(256 * learned_value.max()/(learned_value.max() - learned_value.min()))

                # RdWh = matplotlib.colors.LinearSegmentedColormap.from_list('RdWh', [(1,0,0), (1,0,0), (1,0.5,0.5), (1,1,1)])
                # WhBl = matplotlib.colors.LinearSegmentedColormap.from_list('WhBl', [(1,1,1), (0.5,0.5,1), (0,0,1), (0,0,1)])
                # RdWh = matplotlib.colors.LinearSegmentedColormap.from_list('RdWh', [(1,0,0), (1,0,0), (0.5, 0.,0.), (0,0,0)])
                # WhBl = matplotlib.colors.LinearSegmentedColormap.from_list('WhBl', [(0,0,0), (0.,0.,0.5), (0,0,1), (0,0,1)])
                # RdWh = matplotlib.colors.LinearSegmentedColormap.from_list('RdWh', [(153/255, 21/255, 39/255), (153/255, 21/255, 39/255), (153/255, 21/255, 39/255), (10/255,10/255,15/255)])
                # WhBl = matplotlib.colors.LinearSegmentedColormap.from_list('WhBl', [(10/255,10/255,15/255), (38/255, 69/255, 168/255), (38/255, 69/255, 168/255), (38/255, 69/255, 168/255)])
                                                                                    
                # colors = np.vstack((RdWh(np.linspace(0., 1, 256-n_bins_high)), WhBl(np.linspace(0., 1, n_bins_high))))
                # RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', colors)

                # gray_band_width = 4
                # Gray = matplotlib.colors.LinearSegmentedColormap.from_list('Gray', [(10/255,10/255,15/255),(10/255,10/255,15/255)])
                # colors = np.vstack((matplotlib.colormaps["RdBu"](np.linspace(0., 0.5, 256-n_bins_high)), Gray(np.linspace(0., 1., int(gray_band_width))), matplotlib.colormaps["RdBu"](np.linspace(0.5, 1., n_bins_high-gray_band_width))))
                # RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', colors)

                offset = 0
                scaled_colors = np.vstack((matplotlib.colormaps[cmap_name](np.linspace(0., 0.4, 256-n_bins_high+offset)), matplotlib.colormaps[cmap_name](np.linspace(0.6, 1., n_bins_high-offset))))
                RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', scaled_colors)
            
            if i >= len(times):

                ## Plot Zero-level Set of Learned Value
                
                # s = ax.imshow(1*(learned_value.T <= 0), cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
                s = ax.imshow(learned_value.T, cmap=RdWhBl_vscaled, origin='lower', extent=(-1., 1., -1., 1.))
                # s = ax.contourf(Xg, Yg, learned_value, cmap=RdWhBl_vscaled, levels=256)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(s, cax=cax)
                cbar.set_ticks([learned_value.min(), 0., learned_value.max()])  # Define custom tick locations
                cbar.set_ticklabels([f'{learned_value.min():1.1f}', '0', f'{learned_value.max():1.1f}'])  # Define custom tick labels

                ## Plot Ground-Truth Zero-Level Contour

                ax.contour(self.dataset.X1g, self.dataset.X2g, Vgt, [0.], linewidths=4, alpha=0.7, colors='k')

                # ## Plot the Linear Ground-Truth (ideal warm-start) Zero-Level Contour

                # Vgt = self.dataset.values_DP_linear_grid[ix:ix+n_grid_plane_pts].reshape(n_grid_len, n_grid_len).cpu()
                # ax.contour(self.dataset.X1g, self.dataset.X2g, Vgt, [0.], colors='gold', linestyles='dashed')

            ## Plot 3D Value Fn

            else:
                if plot_value:
                    # ax_val.grid(False)
                    ax.view_init(elev=15, azim=-60)
                    ax.set_facecolor((1, 1, 1, 1))
                    surf = ax.plot_surface(Xg, Yg, learned_value, cmap=RdWhBl_vscaled, alpha=0.8) #cmap='bwr_r')
                    # surf = ax.plot_surface(self.dataset.X1g, self.dataset.X2g, Vgt, cmap=RdWhBl_vscaled, alpha=0.8) #cmap='bwr_r')
                    
                    # divider = make_axes_locatable(ax_set)
                    # cax = divider.append_axes("right", size="5%", pad=0.05)
                    # fig_set.colorbar(s, cax=cax)
                    cbar = fig.colorbar(surf, ax=ax, fraction=0.02, pad=0.0)

                    # cbar.ax.yaxis.set_ticks_position('left')
                    # cbar.ax.yaxis.set_label_position('left')

                    # ax.set_zlim(-max(ax.get_zlim()[1]/5, 0.5))
                    # ax.set_zlim(-max(learned_value.max()/5, 0.5), max(learned_value.max(), 2.5))
                    ax.set_zlim(learned_value.min() - (learned_value.max() - learned_value.min())/5)
                    # ax.contour(Xg, Yg, learned_value, zdir='z', offset=ax.get_zlim()[0], cmap=RdWh, levels=[0.]) #cmap='bwr_r')

                    ax.contour(Xg, Yg, learned_value, zdir='z', offset=ax.get_zlim()[0], colors='k', levels=[0.]) #cmap='bwr_r')
                    # ax.contour(self.dataset.X1g, self.dataset.X2g, Vgt, zdir='z', offset=ax.get_zlim()[0], colors='k', levels=[0.]) #cmap='bwr_r')

                    ax.set_facecolor((1, 1, 1, 1))
                    # ax_val.grid(False)

        # fig_set.savefig(save_path)
        # if plot_value: fig_val.savefig(save_path.split('_epoch')[0] + '_Vfn' + save_path.split('_epoch')[1])
        # if self.use_wandb:
        #     log_dict_plot = {'step': epoch,
        #                 'val_plot': wandb.Image(fig_set),} # (silly) legacy name
        #     if plot_value: log_dict_plot['val_fn_plot'] = wandb.Image(fig_val)
        #     wandb.log(log_dict_plot)
        # plt.close()

        fig.savefig(save_path)
        if self.use_wandb:
            log_dict_plot = {'step': epoch,
                        'val_plot': wandb.Image(fig),} # (silly) legacy name
            wandb.log(log_dict_plot)
        plt.close()

        # if self.dataset.record_gt_metrics:
        #     self.plot_set_metrics_eachtime(epoch, times)
        #     if self.use_wandb:
        #         wandb.log({'Time vs. Epoch vs. Set Accuracy compared to DP': wandb.Image(self.t_ep_acc_fig),})
        #     plt.close()

        if was_training:
            self.model.train()
            self.model.requires_grad_(True)

    def validateNDlambda(self, epoch, save_path, x_resolution, y_resolution, z_resolution, time_resolution, plot_value=True):
        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)

        plot_config = self.dataset.dynamics.plot_config()

        state_test_range = self.dataset.dynamics.state_test_range()
        x_min, x_max = state_test_range[plot_config['x_axis_idx']]
        y_min, y_max = state_test_range[plot_config['y_axis_idx']]
        # z_min, z_max = state_test_range[plot_config['z_axis_idx']]

        times = torch.linspace(0, self.dataset.tMax, time_resolution)
        xs = torch.linspace(x_min, x_max, x_resolution)
        ys = torch.linspace(y_min, y_max, y_resolution)
        # zs = torch.linspace(z_min, z_max, z_resolution)
        xys = torch.cartesian_prod(xs, ys)
        Xg, Yg = torch.meshgrid(xs, ys)
        
        ## Plot Set and Value Fn
        
        # fig_set = plt.figure(figsize=(5*len(times), 2*5*1))
        # fig_val = plt.figure(figsize=(5*len(times), 2*5*1), facecolor='white')

        fig = plt.figure(figsize=(5*len(times), 4*5*1), facecolor='white')
        
        plt.rcParams['text.usetex'] = False

        for i in range(4*len(times)):
        # for i in range(2*len(times)):
            
            # ax_set = fig_set.add_subplot(2, len(times), 1+i)
            # ax_val = fig_val.add_subplot(2, len(times), 1+i, projection='3d')
            # ax_set.set_title('t = %0.2f' % (times[i % len(times)]))
            # ax_val.set_title('t = %0.2f' % (times[i % len(times)]))

            if i < len(times):
                lambda_val = 0 
            elif i < 2*len(times):
                lambda_val = self.dataset.lambda_int_1
            elif i < 3*len(times):
                lambda_val = self.dataset.lambda_int_3
            else:
                lambda_val = 1 

            ax = fig.add_subplot(4, len(times), 1+i)
            ax.set_title(f"t = {times[i % len(times)]:0.2f}, lambda = {lambda_val:1.1f}")

            ## Define Grid Slice to Plot

            coords = torch.zeros(x_resolution*y_resolution, self.dataset.dynamics.state_dim + 1)
            coords[:, 0] = times[i % len(times)]
            coords[:, 1:] = torch.tensor(plot_config['state_slices']) # initialized to zero (nothing else to set!)
            coords[:, -1] = lambda_val

            # if i < len(times): # xN - xi plane
            #     ax_set.set_xlabel("xN"); ax_set.set_ylabel("xi")
            #     ax_val.set_xlabel("xN"); ax_val.set_ylabel("xi")
            #     coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
            #     coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]

            # elif i < 2*len(times): # xi - xj plane
            #     ax_set.set_xlabel("xi"); ax_set.set_ylabel("xj")
            #     ax_val.set_xlabel("xi"); ax_val.set_ylabel("xj")
            #     coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 0]
            #     coords[:, 1 + plot_config['z_axis_idx']] = xys[:, 1]

            # xN - (xi = xj) plane
            pad_label = 0
            ax.set_xlabel(r"$x_N$", fontsize=12, labelpad=pad_label); ax.set_ylabel(r"$x_i = x_j$", fontsize=12, labelpad=pad_label)
            ax.set_xticks([-1, 1])
            ax.set_xticklabels([r'$-1$', r'$1$'])
            ax.set_yticks([-1, 1])
            ax.set_yticklabels([r'$-1$', r'$1$'])

            ax_pad = 0
            ax.xaxis.set_tick_params(pad=ax_pad)
            ax.yaxis.set_tick_params(pad=ax_pad)

            coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
            coords[:, 2:-1] = (xys[:, 1] * torch.ones(self.N-1, xys.size()[0])).t()

            with torch.no_grad():
                model_results = self.model({'coords': self.dataset.dynamics.coord_to_input(coords.cuda())})
                values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach())
            
            learned_value = values.detach().cpu().numpy().reshape(x_resolution, y_resolution)

            n_grid_plane_pts = int(self.dataset.n_grid_pts/3)
            n_grid_len = int(n_grid_plane_pts ** 0.5)
            pix_start = (i // len(times)) * n_grid_plane_pts
            tix_start = (i % len(times)) * self.dataset.n_grid_pts
            # ix = pix_start + tix_start # plane and grid no longer synced
            ix = tix_start

            if i < len(times):
                Vgt = self.dataset.values_DP_linear_grid[ix:ix+n_grid_plane_pts].reshape(n_grid_len, n_grid_len).cpu()
            elif i < 2*len(times):
                Vgt = self.dataset.values_DP_grid_inlam1[ix:ix+n_grid_plane_pts].reshape(n_grid_len, n_grid_len).cpu()
            elif i < 3*len(times):
                Vgt = self.dataset.values_DP_grid_inlam2[ix:ix+n_grid_plane_pts].reshape(n_grid_len, n_grid_len).cpu()
            else:
                Vgt = self.dataset.values_DP_grid[ix:ix+n_grid_plane_pts].reshape(n_grid_len, n_grid_len).cpu() 

            ## Make Value-Based Colormap
            # cmap_name = "coolwarm"
            cmap_name = "RdBu"

            if learned_value.min() > 0:
                # RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', [(1,1,1), (0.5,0.5,1), (0,0,1), (0,0,1)])
                scaled_colors = np.vstack((matplotlib.colormaps[cmap_name](np.linspace(0.6, 1., 256))))
                RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', scaled_colors)

            elif learned_value.max() < 0:
                # RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', [(1,0,0), (1,0,0), (1,0.5,0.5), (1,1,1)])
                scaled_colors = np.vstack((matplotlib.colormaps[cmap_name](np.linspace(0., 0.4, 256))))
                RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', scaled_colors)

            else:
                # n_bins_high = int(256 * (learned_value.max()/(learned_value.max() - learned_value.min())) // 1)
                n_bins_high = round(256 * learned_value.max()/(learned_value.max() - learned_value.min()))

                # RdWh = matplotlib.colors.LinearSegmentedColormap.from_list('RdWh', [(1,0,0), (1,0,0), (1,0.5,0.5), (1,1,1)])
                # WhBl = matplotlib.colors.LinearSegmentedColormap.from_list('WhBl', [(1,1,1), (0.5,0.5,1), (0,0,1), (0,0,1)])
                # RdWh = matplotlib.colors.LinearSegmentedColormap.from_list('RdWh', [(1,0,0), (1,0,0), (0.5, 0.,0.), (0,0,0)])
                # WhBl = matplotlib.colors.LinearSegmentedColormap.from_list('WhBl', [(0,0,0), (0.,0.,0.5), (0,0,1), (0,0,1)])
                # RdWh = matplotlib.colors.LinearSegmentedColormap.from_list('RdWh', [(153/255, 21/255, 39/255), (153/255, 21/255, 39/255), (153/255, 21/255, 39/255), (10/255,10/255,15/255)])
                # WhBl = matplotlib.colors.LinearSegmentedColormap.from_list('WhBl', [(10/255,10/255,15/255), (38/255, 69/255, 168/255), (38/255, 69/255, 168/255), (38/255, 69/255, 168/255)])
                                                                                    
                # colors = np.vstack((RdWh(np.linspace(0., 1, 256-n_bins_high)), WhBl(np.linspace(0., 1, n_bins_high))))
                # RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', colors)

                # gray_band_width = 4
                # Gray = matplotlib.colors.LinearSegmentedColormap.from_list('Gray', [(10/255,10/255,15/255),(10/255,10/255,15/255)])
                # colors = np.vstack((matplotlib.colormaps["RdBu"](np.linspace(0., 0.5, 256-n_bins_high)), Gray(np.linspace(0., 1., int(gray_band_width))), matplotlib.colormaps["RdBu"](np.linspace(0.5, 1., n_bins_high-gray_band_width))))
                # RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', colors)

                offset = 0
                scaled_colors = np.vstack((matplotlib.colormaps[cmap_name](np.linspace(0., 0.4, 256-n_bins_high+offset)), matplotlib.colormaps[cmap_name](np.linspace(0.6, 1., n_bins_high-offset))))
                RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', scaled_colors)

            ## Plot Zero-level Set of Learned Value
            
            # s = ax.imshow(1*(learned_value.T <= 0), cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
            s = ax.imshow(learned_value.T, cmap=RdWhBl_vscaled, origin='lower', extent=(-1., 1., -1., 1.))
            # s = ax.contourf(Xg, Yg, learned_value, cmap=RdWhBl_vscaled, levels=256)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(s, cax=cax)
            cbar.set_ticks([learned_value.min(), 0., learned_value.max()])  # Define custom tick locations
            cbar.set_ticklabels([f'{learned_value.min():1.1f}', '0', f'{learned_value.max():1.1f}'])  # Define custom tick labels

            ## Plot Ground-Truth Zero-Level Contour

            ax.contour(self.dataset.X1g, self.dataset.X2g, Vgt, [0.], linewidths=4, alpha=0.7, colors='k')

            # ## Plot the Linear Ground-Truth (ideal warm-start) Zero-Level Contour

            # Vgt = self.dataset.values_DP_linear_grid[ix:ix+n_grid_plane_pts].reshape(n_grid_len, n_grid_len).cpu()
            # ax.contour(self.dataset.X1g, self.dataset.X2g, Vgt, [0.], colors='gold', linestyles='dashed')

            ## Plot 3D Value Fn

            # else:
            #     if plot_value:
            #         # ax_val.grid(False)
            #         ax.view_init(elev=15, azim=-60)
            #         ax.set_facecolor((1, 1, 1, 1))
            #         surf = ax.plot_surface(Xg, Yg, learned_value, cmap=RdWhBl_vscaled, alpha=0.8) #cmap='bwr_r')
            #         # surf = ax.plot_surface(self.dataset.X1g, self.dataset.X2g, Vgt, cmap=RdWhBl_vscaled, alpha=0.8) #cmap='bwr_r')
                    
            #         # divider = make_axes_locatable(ax_set)
            #         # cax = divider.append_axes("right", size="5%", pad=0.05)
            #         # fig_set.colorbar(s, cax=cax)
            #         cbar = fig.colorbar(surf, ax=ax, fraction=0.02, pad=0.0)

            #         # cbar.ax.yaxis.set_ticks_position('left')
            #         # cbar.ax.yaxis.set_label_position('left')

            #         # ax.set_zlim(-max(ax.get_zlim()[1]/5, 0.5))
            #         # ax.set_zlim(-max(learned_value.max()/5, 0.5), max(learned_value.max(), 2.5))
            #         ax.set_zlim(learned_value.min() - (learned_value.max() - learned_value.min())/5)
            #         # ax.contour(Xg, Yg, learned_value, zdir='z', offset=ax.get_zlim()[0], cmap=RdWh, levels=[0.]) #cmap='bwr_r')

            #         ax.contour(Xg, Yg, learned_value, zdir='z', offset=ax.get_zlim()[0], colors='k', levels=[0.]) #cmap='bwr_r')
            #         # ax.contour(self.dataset.X1g, self.dataset.X2g, Vgt, zdir='z', offset=ax.get_zlim()[0], colors='k', levels=[0.]) #cmap='bwr_r')

            #         ax.set_facecolor((1, 1, 1, 1))
            #         # ax_val.grid(False)

        # fig_set.savefig(save_path)
        # if plot_value: fig_val.savefig(save_path.split('_epoch')[0] + '_Vfn' + save_path.split('_epoch')[1])
        # if self.use_wandb:
        #     log_dict_plot = {'step': epoch,
        #                 'val_plot': wandb.Image(fig_set),} # (silly) legacy name
        #     if plot_value: log_dict_plot['val_fn_plot'] = wandb.Image(fig_val)
        #     wandb.log(log_dict_plot)
        # plt.close()

        fig.savefig(save_path)
        if self.use_wandb:
            log_dict_plot = {'step': epoch,
                        'val_plot': wandb.Image(fig),} # (silly) legacy name
            wandb.log(log_dict_plot)
        plt.close()

        # if self.dataset.record_gt_metrics:
        #     self.plot_set_metrics_eachtime(epoch, times)
        #     if self.use_wandb:
        #         wandb.log({'Time vs. Epoch vs. Set Accuracy compared to DP': wandb.Image(self.t_ep_acc_fig),})
        #     plt.close()

        if was_training:
            self.model.train()
            self.model.requires_grad_(True)   

    ## TODO
    def validateNDlambdamulob(self, epoch, save_path, x_resolution, y_resolution, z_resolution, time_resolution, plot_value=True):
        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)

        plot_config = self.dataset.dynamics.plot_config()

        state_test_range = self.dataset.dynamics.state_test_range()
        x_min, x_max = state_test_range[plot_config['x_axis_idx']]
        y_min, y_max = state_test_range[plot_config['y_axis_idx']]
        # z_min, z_max = state_test_range[plot_config['z_axis_idx']]

        times = torch.linspace(0, self.dataset.tMax, time_resolution)
        xs = torch.linspace(x_min, x_max, x_resolution)
        ys = torch.linspace(y_min, y_max, y_resolution)
        # zs = torch.linspace(z_min, z_max, z_resolution)
        xys = torch.cartesian_prod(xs, ys)
        Xg, Yg = torch.meshgrid(xs, ys)
        
        ## Plot Set and Value Fn
        
        # fig_set = plt.figure(figsize=(5*len(times), 2*5*1))
        # fig_val = plt.figure(figsize=(5*len(times), 2*5*1), facecolor='white')

        fig = plt.figure(figsize=(5*len(times), 5*5*1), facecolor='white')
        
        plt.rcParams['text.usetex'] = False

        for i in range(5*len(times)):
        # for i in range(2*len(times)):
            
            # ax_set = fig_set.add_subplot(2, len(times), 1+i)
            # ax_val = fig_val.add_subplot(2, len(times), 1+i, projection='3d')
            # ax_set.set_title('t = %0.2f' % (times[i % len(times)]))
            # ax_val.set_title('t = %0.2f' % (times[i % len(times)]))

            if i < len(times):
                lambda_val = -1 
            elif i < 2*len(times):
                lambda_val = -0.2
            elif i < 3*len(times):
                lambda_val = 0.
            elif i < 4*len(times):
                lambda_val = 0.2
            else:
                lambda_val = 1 

            ax = fig.add_subplot(5, len(times), 1+i)
            ax.set_title(f"t = {times[i % len(times)]:0.2f}, lambda = {lambda_val:1.1f}")

            ## Define Grid Slice to Plot

            coords = torch.zeros(x_resolution*y_resolution, self.dataset.dynamics.state_dim + 1)
            coords[:, 0] = times[i % len(times)]
            coords[:, 1:] = torch.tensor(plot_config['state_slices']) # initialized to zero (nothing else to set!)
            coords[:, -1] = lambda_val

            # xN - (xi = xj) plane
            pad_label = 0
            ax.set_xlabel(r"$x_N$", fontsize=12, labelpad=pad_label); ax.set_ylabel(r"$x_i = x_j$", fontsize=12, labelpad=pad_label)
            ax.set_xticks([-1, 1])
            ax.set_xticklabels([r'$-1$', r'$1$'])
            ax.set_yticks([-1, 1])
            ax.set_yticklabels([r'$-1$', r'$1$'])

            ax_pad = 0
            ax.xaxis.set_tick_params(pad=ax_pad)
            ax.yaxis.set_tick_params(pad=ax_pad)

            # FIXME
            coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
            coords[:, 2:-1] = (xys[:, 1] * torch.ones(self.N-1, xys.size()[0])).t()

            with torch.no_grad():
                model_results = self.model({'coords': self.dataset.dynamics.coord_to_input(coords.cuda())})
                values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach())
            
            learned_value = values.detach().cpu().numpy().reshape(x_resolution, y_resolution)

            n_grid_plane_pts = int(self.dataset.n_grid_pts/3)
            n_grid_len = int(n_grid_plane_pts ** 0.5)
            pix_start = (i // len(times)) * n_grid_plane_pts
            tix_start = (i % len(times)) * self.dataset.n_grid_pts
            # ix = pix_start + tix_start # plane and grid no longer synced
            ix = tix_start

            Vgt = self.dataset.values_DP_grid[ix:ix+n_grid_plane_pts].reshape(n_grid_len, n_grid_len).cpu() 
            if i < len(times):
                # Vgt = self.dataset.values_DP_linear_grid[ix:ix+n_grid_plane_pts].reshape(n_grid_len, n_grid_len).cpu()
                pass # FIXME: add bc 2
            elif i < 2*len(times):
                # Vgt = self.dataset.values_DP_grid_inlam1[ix:ix+n_grid_plane_pts].reshape(n_grid_len, n_grid_len).cpu()
                pass
            elif i < 3*len(times):
                Vgt = self.dataset.values_DP_grid[ix:ix+n_grid_plane_pts].reshape(n_grid_len, n_grid_len).cpu() 
            elif i < 4*len(times):
                # Vgt = self.dataset.values_DP_grid_inlam2[ix:ix+n_grid_plane_pts].reshape(n_grid_len, n_grid_len).cpu()
                pass
            else:
                # Vgt = self.dataset.values_DP_grid[ix:ix+n_grid_plane_pts].reshape(n_grid_len, n_grid_len).cpu() 
                pass # FIXME: add bc 1

            ## Make Value-Based Colormap
            # cmap_name = "coolwarm"
            cmap_name = "RdBu"

            if learned_value.min() > 0:
                # RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', [(1,1,1), (0.5,0.5,1), (0,0,1), (0,0,1)])
                scaled_colors = np.vstack((matplotlib.colormaps[cmap_name](np.linspace(0.6, 1., 256))))
                RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', scaled_colors)

            elif learned_value.max() < 0:
                # RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', [(1,0,0), (1,0,0), (1,0.5,0.5), (1,1,1)])
                scaled_colors = np.vstack((matplotlib.colormaps[cmap_name](np.linspace(0., 0.4, 256))))
                RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', scaled_colors)

            else:
                # n_bins_high = int(256 * (learned_value.max()/(learned_value.max() - learned_value.min())) // 1)
                n_bins_high = round(256 * learned_value.max()/(learned_value.max() - learned_value.min()))

                offset = 0
                scaled_colors = np.vstack((matplotlib.colormaps[cmap_name](np.linspace(0., 0.4, 256-n_bins_high+offset)), matplotlib.colormaps[cmap_name](np.linspace(0.6, 1., n_bins_high-offset))))
                RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', scaled_colors)

            ## Plot Zero-level Set of Learned Value
            
            # s = ax.imshow(1*(learned_value.T <= 0), cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
            s = ax.imshow(learned_value.T, cmap=RdWhBl_vscaled, origin='lower', extent=(-1., 1., -1., 1.))
            # s = ax.contourf(Xg, Yg, learned_value, cmap=RdWhBl_vscaled, levels=256)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(s, cax=cax)
            cbar.set_ticks([learned_value.min(), 0., learned_value.max()])  # Define custom tick locations
            cbar.set_ticklabels([f'{learned_value.min():1.1f}', '0', f'{learned_value.max():1.1f}'])  # Define custom tick labels

            ## Plot Ground-Truth Zero-Level Contour
            ## TODO: remove gt plotting for plots 2/4
            ax.contour(self.dataset.X1g, self.dataset.X2g, Vgt, [0.], linewidths=4, alpha=0.7, colors='k')

            # ## Plot the Linear Ground-Truth (ideal warm-start) Zero-Level Contour

            # Vgt = self.dataset.values_DP_linear_grid[ix:ix+n_grid_plane_pts].reshape(n_grid_len, n_grid_len).cpu()
            # ax.contour(self.dataset.X1g, self.dataset.X2g, Vgt, [0.], colors='gold', linestyles='dashed')

            ## Plot 3D Value Fn

            # else:
            #     if plot_value:
            #         # ax_val.grid(False)
            #         ax.view_init(elev=15, azim=-60)
            #         ax.set_facecolor((1, 1, 1, 1))
            #         surf = ax.plot_surface(Xg, Yg, learned_value, cmap=RdWhBl_vscaled, alpha=0.8) #cmap='bwr_r')
            #         # surf = ax.plot_surface(self.dataset.X1g, self.dataset.X2g, Vgt, cmap=RdWhBl_vscaled, alpha=0.8) #cmap='bwr_r')
                    
            #         # divider = make_axes_locatable(ax_set)
            #         # cax = divider.append_axes("right", size="5%", pad=0.05)
            #         # fig_set.colorbar(s, cax=cax)
            #         cbar = fig.colorbar(surf, ax=ax, fraction=0.02, pad=0.0)

            #         # cbar.ax.yaxis.set_ticks_position('left')
            #         # cbar.ax.yaxis.set_label_position('left')

            #         # ax.set_zlim(-max(ax.get_zlim()[1]/5, 0.5))
            #         # ax.set_zlim(-max(learned_value.max()/5, 0.5), max(learned_value.max(), 2.5))
            #         ax.set_zlim(learned_value.min() - (learned_value.max() - learned_value.min())/5)
            #         # ax.contour(Xg, Yg, learned_value, zdir='z', offset=ax.get_zlim()[0], cmap=RdWh, levels=[0.]) #cmap='bwr_r')

            #         ax.contour(Xg, Yg, learned_value, zdir='z', offset=ax.get_zlim()[0], colors='k', levels=[0.]) #cmap='bwr_r')
            #         # ax.contour(self.dataset.X1g, self.dataset.X2g, Vgt, zdir='z', offset=ax.get_zlim()[0], colors='k', levels=[0.]) #cmap='bwr_r')

            #         ax.set_facecolor((1, 1, 1, 1))
            #         # ax_val.grid(False)

        # fig_set.savefig(save_path)
        # if plot_value: fig_val.savefig(save_path.split('_epoch')[0] + '_Vfn' + save_path.split('_epoch')[1])
        # if self.use_wandb:
        #     log_dict_plot = {'step': epoch,
        #                 'val_plot': wandb.Image(fig_set),} # (silly) legacy name
        #     if plot_value: log_dict_plot['val_fn_plot'] = wandb.Image(fig_val)
        #     wandb.log(log_dict_plot)
        # plt.close()

        fig.savefig(save_path)
        if self.use_wandb:
            log_dict_plot = {'step': epoch,
                        'val_plot': wandb.Image(fig),} # (silly) legacy name
            wandb.log(log_dict_plot)
        plt.close()

        # if self.dataset.record_gt_metrics:
        #     self.plot_set_metrics_eachtime(epoch, times)
        #     if self.use_wandb:
        #         wandb.log({'Time vs. Epoch vs. Set Accuracy compared to DP': wandb.Image(self.t_ep_acc_fig),})
        #     plt.close()

        if was_training:
            self.model.train()
            self.model.requires_grad_(True)            

    def compute_gt_metrics(self):

        JIp, FIp, FEp, Vmse, DVXmse = 0, 0, 0, 0, 0

        ## Compute Value Gradient MSE on Grid
        if self.dataset.solve_grad:
            model_results_grid = self.model({'coords': self.dataset.model_coords_grid_allt})
            DVX = self.dataset.dynamics.io_to_dv(model_results_grid['model_in'], model_results_grid['model_out'].squeeze(dim=-1))[..., 1:].detach()
            DVXmse = (self.dataset.value_grads_DP_grid - DVX).square().mean()

        with torch.no_grad():
            if not self.dataset.solve_grad:
                model_results_grid = self.model({'coords': self.dataset.model_coords_grid_allt})
            values_grid = self.dataset.dynamics.io_to_value(model_results_grid['model_in'].detach(), model_results_grid['model_out'].squeeze(dim=-1)).detach()

            del model_results_grid
            torch.cuda.empty_cache()

            values_grid_sub0_ixs = torch.argwhere(values_grid <= 0).flatten()

            ## Compute MSE on Grid
            Vmse = (self.dataset.values_DP_grid - values_grid).square().mean()

            ## Compute Set Metrics over Time
            n_intersect = values_grid_sub0_ixs[(values_grid_sub0_ixs.view(1, -1) == self.dataset.values_DP_grid_sub0_ixs.view(-1, 1)).any(dim=0)].size()[0] # ty Amin_Jun
            n_overlap = values_grid_sub0_ixs.size()[0] + self.dataset.values_DP_grid_sub0_ixs.size()[0] - n_intersect
            
            if values_grid_sub0_ixs.size()[0] > 0:
                FIp = (values_grid_sub0_ixs.size()[0] - n_intersect) / values_grid_sub0_ixs.size()[0] # <- wrt learned set, wrt grid
            else:
                FIp = 0.
            
            if self.dataset.values_DP_grid_sub0_ixs.size()[0] > 0:
                FEp = (self.dataset.values_DP_grid_sub0_ixs.size()[0] - n_intersect) / self.dataset.values_DP_grid_sub0_ixs.size()[0] # <- wrt true set, wrt grid
            else:
                FEp = 0.

            JIp = n_intersect / n_overlap
            ## NOTE: still wondering if there is a bug in FIp and JIp... they look slightly off sometimes... I've to think they're right but just very nonlinear metrics (maybe bad).
        
        del values_grid
        del values_grid_sub0_ixs
        torch.cuda.empty_cache()

        return JIp, FIp, FEp, Vmse, DVXmse
        
    def set_metrics_eachtime(self): 

        model_results_grid = self.model({'coords': self.dataset.model_coords_grid_allt_hi})
        values_grid = self.dataset.dynamics.io_to_value(model_results_grid['model_in'].detach(), model_results_grid['model_out'].squeeze(dim=-1))
        
        JIps, FIps, FEps = torch.zeros(self.dataset.n_grid_t_pts_hi), torch.zeros(self.dataset.n_grid_t_pts_hi), torch.zeros(self.dataset.n_grid_t_pts_hi)
        for i in range(self.dataset.n_grid_t_pts_hi):

            values_grid_sub0_ixs = torch.argwhere(values_grid[i*self.dataset.n_grid_pts:(i+1)*self.dataset.n_grid_pts] <= 0).flatten()
            values_DP_grid_sub0_ixs = torch.argwhere(self.dataset.values_DP_grid_hi[i*self.dataset.n_grid_pts:(i+1)*self.dataset.n_grid_pts] <= 0).flatten()

            # n_intersect = np.intersect1d(values_grid_sub0_ixs, values_DP_grid_sub0_ixs).size
            n_intersect = values_grid_sub0_ixs[(values_grid_sub0_ixs.view(1, -1) == values_DP_grid_sub0_ixs.view(-1, 1)).any(dim=0)].size()[0] # ty Amin_Jun
            n_overlap = values_grid_sub0_ixs.size()[0] + values_DP_grid_sub0_ixs.size()[0] - n_intersect

            if values_grid_sub0_ixs.size()[0] > 0:
                FIps[i] = (values_grid_sub0_ixs.size()[0] - n_intersect) / values_grid_sub0_ixs.size()[0] # <- wrt true set, wrt grid: self.dataset.n_grid_pts
            else:
                FIps[i]  = 1.
            FEps[i] = (values_DP_grid_sub0_ixs.size()[0] - n_intersect) / values_DP_grid_sub0_ixs.size()[0] # <- wrt true set, wrt grid: self.dataset.n_grid_pts
            JIps[i] = n_intersect / n_overlap

        return JIps, FIps, FEps

    def plot_set_metrics_eachtime(self, epoch, times):
        with torch.no_grad():
            JIps, FIps, FEps = self.set_metrics_eachtime()
        
        if not(hasattr(self, 't_ep_acc_fig')):
            self.t_ep_acc_fig, (self.t_ep_acc_fig_ax1, self.t_ep_acc_fig_ax2, self.t_ep_acc_fig_ax3) = plt.subplots(1, 3, figsize=(5*len(times), 5*1), subplot_kw={'projection': '3d'})

            for axi, ax in enumerate([self.t_ep_acc_fig_ax1, self.t_ep_acc_fig_ax2, self.t_ep_acc_fig_ax3]):
                ax.set_xlim(self.dataset.tMin, self.dataset.tMax)
                ax.set_ylim(1, self.dataset.counter_end/100)
                ax.set_zlim(0, 1)
                ax.set_xlabel('Time')
                ax.set_ylabel('Epoch (C)')
    
    def demo_lambdavar_plot(self, save_path, x_resolution, y_resolution, plot_value=True):
        # was_training = self.model.training
        # self.model.eval()
        # self.model.requires_grad_(False)

        plot_config = self.dataset.dynamics.plot_config()

        state_test_range = self.dataset.dynamics.state_test_range()
        x_min, x_max = state_test_range[plot_config['x_axis_idx']]
        y_min, y_max = state_test_range[plot_config['y_axis_idx']]
        # z_min, z_max = state_test_range[plot_config['z_axis_idx']]

        # times = torch.linspace(0, self.dataset.tMax, time_resolution)
        lambdas = torch.tensor([0, 0.1, 0.2, 0.5, 1])
        xs = torch.linspace(x_min, x_max, x_resolution)
        ys = torch.linspace(y_min, y_max, y_resolution)
        # zs = torch.linspace(z_min, z_max, z_resolution)
        xys = torch.cartesian_prod(xs, ys)
        Xg, Yg = torch.meshgrid(xs, ys)
        
        ## Plot Set and Value Fn
        
        fig = plt.figure(figsize=(5*len(lambdas), 2*5*1), facecolor='white')
        # fig = plt.figure(figsize=(5*len(lambdas), 2*5*1), facecolor='white')
        
        plt.rcParams['text.usetex'] = False

        # for i in range(3*len(times)):
        for i in range(2*len(lambdas)):

            lambda_val = lambdas[i % len(lambdas)]
            if i > len(lambdas):
                ax = fig.add_subplot(2, len(lambdas), 1+i)
                if i % len(lambdas) == 4:
                    ax.set_title(r"$|V_\ell - V_\lambda|$, $\lambda = 1$", fontsize=14)
                elif i % len(lambdas) == 1:
                    ax.set_title(r"$|V_\ell - V_\lambda|$, $\lambda = 1/10$", fontsize=16)
                elif i % len(lambdas) == 2:
                    ax.set_title(r"$|V_\ell - V_\lambda|$, $\lambda = 1/5$", fontsize=16)
                elif i % len(lambdas) == 3:
                    ax.set_title(r"$|V_\ell - V_\lambda|$, $\lambda = 1/2$", fontsize=16)
                # else:
                #     ax.set_title(r"$|V_\ell - V_\lambda|$, $\lambda = $ " + f"{lambda_val:1.1f}", fontsize=14)
            elif i == len(lambdas):
                continue
            else:
                ax = fig.add_subplot(2, len(lambdas), 1+i, projection='3d')
                if i == 0:
                    ax.set_title(r"$V_\ell = V_\lambda$, $\lambda = 0$", fontsize=16)
                elif i == 1:
                    ax.set_title(r"$V_\lambda$, $\lambda = 1/10$", fontsize=16)
                elif i == 2:
                    ax.set_title(r"$V_\lambda$, $\lambda = 1/5$", fontsize=16)
                elif i == 3:
                    ax.set_title(r"$V_\lambda$, $\lambda = 1/2$", fontsize=16)
                elif i == 4:
                    ax.set_title(r"$V_\lambda$, $\lambda = 1$", fontsize=16)
                # else:
                #     ax.set_title(r"$V_\lambda$, $\lambda =$ " + f"{lambda_val:1.1f}", fontsize=16)

            ## Define Grid Slice to Plot

            coords = torch.zeros(x_resolution*y_resolution, self.dataset.dynamics.state_dim + 1)
            # coords[:, 0] = times[i % len(times)]
            coords[:, 0] = 1.
            coords[:, 1:] = torch.tensor(plot_config['state_slices']) # initialized to zero (nothing else to set!)
            coords[:, -1] = lambda_val

            if i >= len(lambdas):
                pad_label = 0
                ax.set_xlabel(r"$x_N$", fontsize=12, labelpad=pad_label); ax.set_ylabel(r"$x_i = x_j$", fontsize=12, labelpad=pad_label)
                ax.set_xticks([-1, 1])
                ax.set_xticklabels([r'$-1$', r'$1$'])
                ax.set_yticks([-1, 1])
                ax.set_yticklabels([r'$-1$', r'$1$'])

                ax_pad = 0
                ax.xaxis.set_tick_params(pad=ax_pad)
                ax.yaxis.set_tick_params(pad=ax_pad)

            else:
                pad_label = 6
                ax.set_xlabel(r"$x_N$", fontsize=12, labelpad=pad_label); 
                ax.set_ylabel(r"$x_i = x_j$", fontsize=12, labelpad=pad_label); 
                # ax.set_zlabel(r"$V$", fontsize=12, labelpad=10) #, labelpad=pad_label)
                ax.set_xticks([-1, 0, 1])
                ax.set_xticklabels([r'$-1$', r'$0$', r'$1$'])
                ax.set_yticks([-1, 0, 1])
                ax.set_yticklabels([r'$-1$', r'$0$', r'$1$'])
                ax.set_zticks([])
                ax.zaxis.label.set_position((-0.1, 0.5))
                
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                
                ax_pad = 0
                ax.xaxis.set_tick_params(pad=ax_pad)
                ax.yaxis.set_tick_params(pad=ax_pad)
                ax.zaxis.set_tick_params(pad=ax_pad)

            # xN - (xi = xj) plane
            coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
            coords[:, 2:-1] = (xys[:, 1] * torch.ones(self.N-1, xys.size()[0])).t()

            # n_grid_plane_pts = int(self.dataset.n_grid_pts/3)
            # n_grid_len = int(n_grid_plane_pts ** 0.5)
            # # pix_start = (i // len(times)) * n_grid_plane_pts
            # # tix_start = (i % len(times)) * self.dataset.n_grid_pts
            # tix_start = 4 * self.dataset.n_grid_pts
            # # ix = pix_start + tix_start # plane and grid no longer synced
            # ix = tix_start

            # Vgt_linear = self.dataset.values_DP_linear_grid[ix:ix+n_grid_plane_pts].reshape(n_grid_len, n_grid_len).cpu()
            Vgt_linear = self.dataset.V_DP_linear(self.dataset.dynamics.input_to_coord(coords).t()).reshape(x_resolution, y_resolution).cpu()
            Vgt_full = self.dataset.V_DP(self.dataset.dynamics.input_to_coord(coords).t()).reshape(x_resolution, y_resolution).cpu()
            if i % len(lambdas) == 0:
                # Vgt = self.dataset.values_DP_grid_inlam1[ix:ix+n_grid_plane_pts].reshape(n_grid_len, n_grid_len).cpu()
                Vgt = self.dataset.V_DP_linear(self.dataset.dynamics.input_to_coord(coords).t()).reshape(x_resolution, y_resolution).cpu()
            elif i % len(lambdas) == 1:
                # Vgt = self.dataset.values_DP_grid_inlam1[ix:ix+n_grid_plane_pts].reshape(n_grid_len, n_grid_len).cpu()
                Vgt = self.dataset.V_DP_inlam1(self.dataset.dynamics.input_to_coord(coords).t()).reshape(x_resolution, y_resolution).cpu()
            elif i % len(lambdas) == 2:
                # Vgt = self.dataset.values_DP_grid_inlam2[ix:ix+n_grid_plane_pts].reshape(n_grid_len, n_grid_len).cpu()
                Vgt = self.dataset.V_DP_inlam2(self.dataset.dynamics.input_to_coord(coords).t()).reshape(x_resolution, y_resolution).cpu()
            elif i % len(lambdas) == 3:
                # Vgt = self.dataset.V_DP_inlam3(self.dynamics.input_to_coord(coords).t()).reshape(n_grid_len, n_grid_len).cpu() # need to make inlam3
                Vgt = self.dataset.V_DP_inlam3(self.dataset.dynamics.input_to_coord(coords).t()).reshape(x_resolution, y_resolution).cpu()
            else:
                Vgt = self.dataset.V_DP(self.dataset.dynamics.input_to_coord(coords).t()).reshape(x_resolution, y_resolution).cpu()

            ## Make Value-Based Colormap
            # cmap_name = "coolwarm"
            cmap_name = "RdBu"

            if Vgt.min() > 0: # FIXME delete
                # RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', [(1,1,1), (0.5,0.5,1), (0,0,1), (0,0,1)])
                scaled_colors = np.vstack((matplotlib.colormaps[cmap_name](np.linspace(0.6, 1., 256))))
                RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', scaled_colors)

            elif Vgt.max() < 0: # FIXME delete
                # RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', [(1,0,0), (1,0,0), (1,0.5,0.5), (1,1,1)])
                scaled_colors = np.vstack((matplotlib.colormaps[cmap_name](np.linspace(0., 0.4, 256))))
                RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', scaled_colors)

            else:
                # n_bins_high = int(256 * (Vgt.max()/(Vgt.max() - Vgt.min())) // 1)
                # n_bins_high = round(256 * Vgt.max().item()/(Vgt.max().item() - Vgt.min().item()))
                max_v_3d = 2
                n_bins_high = round(256 * max_v_3d/(max_v_3d - Vgt.min().item()))

                # RdWh = matplotlib.colors.LinearSegmentedColormap.from_list('RdWh', [(1,0,0), (1,0,0), (1,0.5,0.5), (1,1,1)])
                # WhBl = matplotlib.colors.LinearSegmentedColormap.from_list('WhBl', [(1,1,1), (0.5,0.5,1), (0,0,1), (0,0,1)])
                # RdWh = matplotlib.colors.LinearSegmentedColormap.from_list('RdWh', [(1,0,0), (1,0,0), (0.5, 0.,0.), (0,0,0)])
                # WhBl = matplotlib.colors.LinearSegmentedColormap.from_list('WhBl', [(0,0,0), (0.,0.,0.5), (0,0,1), (0,0,1)])
                # RdWh = matplotlib.colors.LinearSegmentedColormap.from_list('RdWh', [(153/255, 21/255, 39/255), (153/255, 21/255, 39/255), (153/255, 21/255, 39/255), (10/255,10/255,15/255)])
                # WhBl = matplotlib.colors.LinearSegmentedColormap.from_list('WhBl', [(10/255,10/255,15/255), (38/255, 69/255, 168/255), (38/255, 69/255, 168/255), (38/255, 69/255, 168/255)])
                                                                                    
                # colors = np.vstack((RdWh(np.linspace(0., 1, 256-n_bins_high)), WhBl(np.linspace(0., 1, n_bins_high))))
                # RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', colors)

                # gray_band_width = 4
                # Gray = matplotlib.colors.LinearSegmentedColormap.from_list('Gray', [(10/255,10/255,15/255),(10/255,10/255,15/255)])
                # colors = np.vstack((matplotlib.colormaps["RdBu"](np.linspace(0., 0.5, 256-n_bins_high)), Gray(np.linspace(0., 1., int(gray_band_width))), matplotlib.colormaps["RdBu"](np.linspace(0.5, 1., n_bins_high-gray_band_width))))
                # RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', colors)

                offset = 0
                scaled_colors = np.vstack((matplotlib.colormaps[cmap_name](np.linspace(0., 0.4, 256-n_bins_high+offset)), matplotlib.colormaps[cmap_name](np.linspace(0.6, 1., n_bins_high-offset))))
                RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', scaled_colors)

            if i >= len(lambdas):
                ## Plot Zero-level Set of Learned Value
                
                # s = ax.imshow(1*(Vgt.T <= 0), cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
                # s = ax.imshow(Vgt.T, cmap=matplotlib.colormaps["viridis"], origin='lower', extent=(-1., 1., -1., 1.))
                max_v = 1.5

                ## Log Colorbar
                offset = 1e-5
                log_norm = matplotlib.colors.LogNorm(vmin=offset, vmax=max_v)
                s = ax.imshow(torch.abs(Vgt - Vgt_linear).T + torch.tensor([offset]), cmap=matplotlib.colormaps["viridis_r"], origin='lower', extent=(-1., 1., -1., 1.), norm=log_norm) # viridis, terrain, nipy_spectral, rainbow
                # s = ax.contourf(Xg, Yg, Vgt, cmap=RdWhBl_vscaled, levels=256)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(s, cax=cax)
                cbar.set_ticks([offset, max_v])  # FIXME fixed max

                ## Linear Colorbar
                # offset = 0
                # s = ax.imshow(torch.abs(Vgt - Vgt_linear).T + torch.tensor([offset]), cmap=matplotlib.colormaps["viridis_r"], origin='lower', extent=(-1., 1., -1., 1.), vmin=offset, vmax=max_v) # viridis, terrain, nipy_spectral, rainbow
                # # s = ax.contourf(Xg, Yg, Vgt, cmap=RdWhBl_vscaled, levels=256)
                # divider = make_axes_locatable(ax)
                # cax = divider.append_axes("right", size="5%", pad=0.05)
                # cbar = fig.colorbar(s, cax=cax)
                # cbar.set_ticks([offset, max_v])  # FIXME fixed max

                # cbar.set_ticks([0., torch.abs(Vgt_full - Vgt_linear).max()])  # Define custom tick locations
                # cbar.set_ticks([0., torch.abs(Vgt - Vgt_linear).max()])  # FIXME fixed max
                # cbar.set_ticklabels([f'0', f'{torch.abs(Vgt_full - Vgt_linear).max():1.1f}'])  # Define custom tick labels
                # cbar.set_ticklabels([f'0', f'{max_v:1d}'])  # Define custom tick labels
                cbar.set_ticklabels([f'0', f'{max_v:1.1f}'])  # Define custom tick labels

                # ## Plot Ground-Truth Zero-Level Contour

                ax.contour(Xg, Yg, Vgt, [0.], linewidths=4, alpha=0.7, colors='k')

                # ## Plot the Linear Ground-Truth (ideal warm-start) Zero-Level Contour

                # ax.contour(self.dataset.X1g, self.dataset.X2g, Vgt_linear, [0.], linewidths=4, alpha=0.7, colors='k', linestyles='dashed')

            ## Plot 3D Value Fn

            else:
                if plot_value:
                    # ax_val.grid(False)
                    ax.view_init(elev=15, azim=-60)
                    ax.set_facecolor((1, 1, 1, 1))
                    surf = ax.plot_surface(Xg, Yg, Vgt, cmap=RdWhBl_vscaled, alpha=0.8, vmax=max_v_3d) #cmap='bwr_r')
                    # surf = ax.plot_surface(self.dataset.X1g, self.dataset.X2g, Vgt, cmap=RdWhBl_vscaled, alpha=0.8) #cmap='bwr_r')
                    
                    # divider = make_axes_locatable(ax_set)
                    # cax = divider.append_axes("right", size="5%", pad=0.05)
                    # fig_set.colorbar(s, cax=cax)
                    cbar = fig.colorbar(surf, ax=ax, fraction=0.02, pad=0.0)

                    # cbar.ax.yaxis.set_ticks_position('left')
                    # cbar.ax.yaxis.set_label_position('left')
                    
                    cbar.set_ticks([0, max_v_3d])  # FIXME fixed max
                    cbar.set_ticklabels([f'0', f'{max_v_3d:1d}'])  # Define custom tick labels

                    # ax.set_zlim(-max(ax.get_zlim()[1]/5, 0.5))
                    # ax.set_zlim(-max(Vgt.max().item()/5, 0.5), max(Vgt.max().item(), 2.5))
                    # ax.set_zlim(Vgt.min().item() - (Vgt.max().item() - Vgt.min().item())/5)
                    ax.set_zlim(Vgt.min().item() - (Vgt.max().item() - Vgt.min().item())/5, max_v_3d)
                    # ax.contour(Xg, Yg, Vgt, zdir='z', offset=ax.get_zlim()[0], cmap=RdWh, levels=[0.]) #cmap='bwr_r')

                    ax.contour(Xg, Yg, Vgt, zdir='z', offset=ax.get_zlim()[0], colors='k', levels=[0.]) #cmap='bwr_r')
                    # ax.contour(self.dataset.X1g, self.dataset.X2g, Vgt, zdir='z', offset=ax.get_zlim()[0], colors='k', levels=[0.]) #cmap='bwr_r')

                    ax.set_facecolor((1, 1, 1, 1))
                    # ax_val.grid(False)

        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        fig.savefig(save_path)
        plt.close()

    def plot_final_comparison(self, save_path, loaded_models_dict, x_resolution, y_resolution, time_resolution, plot_value=True):
            # was_training = self.model.training
            # self.model.eval()
            # self.model.requires_grad_(False)

            plot_config = self.dataset.dynamics.plot_config()

            state_test_range = self.dataset.dynamics.state_test_range()
            x_min, x_max = state_test_range[plot_config['x_axis_idx']]
            y_min, y_max = state_test_range[plot_config['y_axis_idx']]
            # z_min, z_max = state_test_range[plot_config['z_axis_idx']]

            times = torch.linspace(0, self.dataset.tMax, time_resolution)
            methods = ["Baseline", "LSS Augmentation", "LSS Decay"]

            xs = torch.linspace(x_min, x_max, x_resolution)
            ys = torch.linspace(y_min, y_max, y_resolution)
            # zs = torch.linspace(z_min, z_max, z_resolution)
            xys = torch.cartesian_prod(xs, ys)
            Xg, Yg = torch.meshgrid(xs, ys)
            
            ## Plot Set and Value Fn

            models_JIp = [] 
            models_FIp = [] 
            models_FEp = []  
            models_Vmse = [] 
            models_DVXmse = []  
            
            fig = plt.figure(figsize=(5*len(methods), 2*5*1), facecolor='white', dpi=300)
            # fig = plt.figure(figsize=(5*len(methods), 2*5*1), facecolor='white')
            
            plt.rcParams['text.usetex'] = False

            # for i in range(3*len(times)):
            for (i, model_name) in enumerate(loaded_models_dict.keys()):

                print(model_name)

                ax = fig.add_subplot(len(methods), 4, 1+i)

                pad_label = 0
                ax.set_xlabel(r"$x_0$", fontsize=12, labelpad=pad_label); ax.set_ylabel(r"$x_i = x_j$", fontsize=12, labelpad=pad_label)
                ax.set_xticks([-1, 1])
                ax.set_xticklabels([r'$-1$', r'$1$'])
                ax.set_yticks([-1, 1])
                ax.set_yticklabels([r'$-1$', r'$1$'])
                ax.yaxis.set_label_coords(-0.02, 0.5)  # (x, y) for the y-label
                ax.xaxis.set_label_coords(0.5, -0.02)  # (x, y) for the x-label

                ax_pad = 0
                ax.xaxis.set_tick_params(pad=ax_pad)
                ax.yaxis.set_tick_params(pad=ax_pad)

                ## Define Grid Slice to Plot
                
                if "b4" in model_name:
                    time_to_plot = 0.5
                else: 
                    time_to_plot = 1.

                if "ZLLS" in model_name:
                    coords = torch.zeros(x_resolution*y_resolution, self.dataset.dynamics.state_dim + 2)
                    coords[:, 0] = time_to_plot
                    # coords[:, 1:] = torch.tensor(plot_config['state_slices']) # initialized to zero (nothing else to set!)
                    coords[:, -1] = 1. # set lambda to 1

                    # xN - (xi = xj) plane
                    coords[:, 1] = xys[:, 0]
                    coords[:, 2:-1] = (xys[:, 1] * torch.ones(self.N-1, xys.size()[0])).t()

                else:
                    coords = torch.zeros(x_resolution*y_resolution, self.dataset.dynamics.state_dim + 1)
                    coords[:, 0] = time_to_plot
                    coords[:, 1:] = torch.tensor(plot_config['state_slices']) # initialized to zero (nothing else to set!)

                    # xN - (xi = xj) plane
                    coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
                    coords[:, 2:] = (xys[:, 1] * torch.ones(self.N-1, xys.size()[0])).t()

                if "ZLLS" not in model_name:
                    model_results = loaded_models_dict[model_name]({'coords': self.dataset.dynamics.coord_to_input(coords.cuda())})
                    values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach()).cpu()
                    learned_grads = self.dataset.dynamics.io_to_dv(model_results['model_in'], model_results['model_out'].squeeze(dim=-1))[..., 1:].detach().cpu()
                else:
                    model_results = loaded_models_dict[model_name]({'coords': self.dataset.loaded_dynamics.coord_to_input(coords.cuda())})
                    values = self.dataset.loaded_dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach()).cpu()
                    learned_grads = self.dataset.loaded_dynamics.io_to_dv(model_results['model_in'], model_results['model_out'].squeeze(dim=-1))[..., 1:-1].detach().cpu()
                    # print("learned_grads.shape", learned_grads.shape)
            
                learned_value = values.detach().cpu().numpy().reshape(x_resolution, y_resolution)
                
                n_grid_plane_pts = int(self.dataset.n_grid_pts/3)
                n_grid_len = int(n_grid_plane_pts ** 0.5)
                # tix_start = (i % len(times)) * self.dataset.n_grid_pts

                if "b4" in model_name:
                    tix_start = 2 * self.dataset.n_grid_pts
                else:
                    tix_start = 4 * self.dataset.n_grid_pts

                ix = tix_start

                if "ZLLS" not in model_name:
                    if i % 4 == 0:
                        Vgt_flat, Vgt_grad = self.dataset.V_DP_grad_b1(self.dataset.dynamics.input_to_coord(coords).t())
                    elif i % 4 == 1:
                        Vgt_flat, Vgt_grad = self.dataset.V_DP_grad_b2(self.dataset.dynamics.input_to_coord(coords).t())
                    elif i % 4 == 2:
                        Vgt_flat, Vgt_grad = self.dataset.V_DP_grad_b3(self.dataset.dynamics.input_to_coord(coords).t())
                    else:
                        Vgt_flat, Vgt_grad = self.dataset.V_DP_grad_b4(self.dataset.dynamics.input_to_coord(coords).t())
                else:
                    if i % 4 == 0:
                        Vgt_flat, Vgt_grad = self.dataset.V_DP_grad_b1(self.dataset.loaded_dynamics.input_to_coord(coords).t())
                    elif i % 4 == 1:
                        Vgt_flat, Vgt_grad = self.dataset.V_DP_grad_b2(self.dataset.loaded_dynamics.input_to_coord(coords).t())
                    elif i % 4 == 2:
                        Vgt_flat, Vgt_grad = self.dataset.V_DP_grad_b3(self.dataset.loaded_dynamics.input_to_coord(coords).t())
                    else:
                        Vgt_flat, Vgt_grad = self.dataset.V_DP_grad_b4(self.dataset.loaded_dynamics.input_to_coord(coords).t())
                Vgt = Vgt_flat.reshape(x_resolution, y_resolution).cpu()

                ## Make Value-Based Colormap
                # cmap_name = "coolwarm"
                cmap_name = "RdBu"
                num_bins = 1024

                # min_val = learned_value.min()
                # max_val = learned_value.max()
                
                min_vals = [-4.5, -3.8, -5.2, -7.8]
                max_vals = [33.1, 48.1, 70.3, 34.2]
                
                min_val = min_vals[i % 4]
                max_val = max_vals[i % 4]

                if min_val > 0:
                    # RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', [(1,1,1), (0.5,0.5,1), (0,0,1), (0,0,1)])
                    scaled_colors = np.vstack((matplotlib.colormaps[cmap_name](np.linspace(0.6, 1., num_bins))))
                    RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', scaled_colors)

                elif max_val < 0:
                    # RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', [(1,0,0), (1,0,0), (1,0.5,0.5), (1,1,1)])
                    scaled_colors = np.vstack((matplotlib.colormaps[cmap_name](np.linspace(0., 0.4, num_bins))))
                    RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', scaled_colors)

                else:
                    # n_bins_high = int(256 * (max_val/(max_val - min_val)) // 1)
                    n_bins_high = round(num_bins * max_val/(max_val - min_val))

                    offset = 0
                    scaled_colors = np.vstack((matplotlib.colormaps[cmap_name](np.linspace(0., 0.4, num_bins-n_bins_high+offset)), matplotlib.colormaps[cmap_name](np.linspace(0.6, 1., n_bins_high-offset))))
                    RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', scaled_colors)

                ## Plot Learned Value
                s = ax.imshow(learned_value.T, cmap=RdWhBl_vscaled, origin='lower', extent=(-1., 1., -1., 1.), interpolation='bilinear', vmax=max_val, vmin=min_val)
                # levels = np.linspace(min_val, max_val, num_bins)
                # s = ax.contourf(learned_value.T, cmap=RdWhBl_vscaled, levels=levels, origin='lower', extent=(-1., 1., -1., 1.))
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(s, cax=cax)
                cbar.set_ticks([min_val, 0., max_val])  # Define custom tick locations
                cbar.set_ticklabels([f'{min_val:1.1f}', '0', f'{max_val:1.1f}'])  # Define custom tick labels

                ## Plot Ground-Truth Zero-Level Contour

                ax.contour(Xg, Yg, Vgt, [0.], linewidths=4, alpha=0.7, colors='k')


                if i < 4:
                    titles = [r'$ \alpha = 20, \beta = 0$', r'$ \alpha = -20,, \beta = 0$', r'$ \alpha = -20, \beta = 20$', r'$ \alpha = 10, \beta = -10$']
                    ax.set_title(titles[i], fontsize=10)

                if i % 4 == 0:
                    labels = [r"BASELINE", r"LSS DECAY", r"$ V_\lambda $ LSS"]
                    ax.text(
                        x=-1.5,  # Position to the left of the y-axis
                        y=0.,   # Vertical position in data coordinates
                        s=labels[int(i/4)],  # Annotation text
                        rotation=90,              # Rotate text to vertical
                        va='center',              # Vertical alignment
                        ha='right',               # Horizontal alignment
                        fontsize=20,              # Font size
                        color='black'              # Text color
                    )

                ## Compute GT Metrics
                
                JIp, FIp, FEp, Vmse, DVXmse = 0, 0, 0, 0, 0

                ## Compute Value Gradient MSE on Grid
                with torch.no_grad():
                    
                    if "ZLLS" not in model_name:
                        DVXmse = (Vgt_grad - learned_grads).square().mean()
                    else:
                        DVXmse = (Vgt_grad[:,:-1] - learned_grads).square().mean() ## FIXME: Why would Vgt grad be of dim n+1? should those coords be one less dim? why no error then. maybe time?

                    values_grid_sub0_ixs = torch.argwhere(values <= 0).flatten()
                    values_DP_grid_sub0_ixs = torch.argwhere(Vgt_flat <= 0).flatten()

                    ## Compute MSE on Grid
                    Vmse = (Vgt_flat - values).square().mean()

                    ## Compute Set Metrics over Time
                    n_intersect = values_grid_sub0_ixs[(values_grid_sub0_ixs.view(1, -1) == values_DP_grid_sub0_ixs.view(-1, 1)).any(dim=0)].size()[0] # ty Amin_Jun
                    n_overlap = values_grid_sub0_ixs.size()[0] + values_DP_grid_sub0_ixs.size()[0] - n_intersect
                    
                    if values_grid_sub0_ixs.size()[0] > 0:
                        FIp = (values_grid_sub0_ixs.size()[0] - n_intersect) / values_grid_sub0_ixs.size()[0] # <- wrt true set, wrt grid: (self.dataset.n_grid_t_pts * self.dataset.n_grid_pts)
                    else:
                        FIp = 1.
                    FEp = (values_DP_grid_sub0_ixs.size()[0] - n_intersect) / values_DP_grid_sub0_ixs.size()[0] # <- wrt true set, wrt grid: (self.dataset.n_grid_t_pts * self.dataset.n_grid_pts)
                    JIp = n_intersect / n_overlap

                models_JIp.append(JIp) 
                models_FIp.append(FIp) 
                models_FEp.append(FEp) 
                models_Vmse.append(Vmse)
                models_DVXmse.append(DVXmse) 

            fig.subplots_adjust(left=0.075, right=0.95, top=0.95, bottom=0.05, wspace=0.25, hspace=0.3)

            fig.savefig(save_path)
            plt.close()

            fig_bar = plt.figure(figsize=(10, 10), facecolor='white', dpi=300)
            categories = [r'$(20, 0)$', r'$(-20, 0)$', r'$(-20, 20)$', r'$(10, -10)$']

            # colors = plt.cm.viridis(np.linspace(0, 1, 3))
            # colors = plt.cm.Pastel2(np.linspace(0, 1, 3))
            colors = plt.cm.Set1.colors[:3]

            model_times = [7.00, 6.85, 7.01, 6.98, 0.33, 0.45, 0.31, 0.38, 7.96, 7.85, 7.58, 7.91]
            # data_list = [models_JIp, models_FIp, models_FEp, models_Vmse, models_DVXmse, model_times]
            datas = {r"IOU":models_JIp, r"Run Time (Hours)":model_times, r"Mean Square Error of Value":models_Vmse, r"Mean Square Error of Gradient":models_DVXmse}

            for (i, data_name) in enumerate(datas.keys()):

                ax = fig_bar.add_subplot(2, 2, 1+i)

                data = datas[data_name]

                group1 = data[0:4]
                group2 = data[4:8]
                group3 = data[8:]

                x = np.arange(4)  # x locations for the groups
                width = 0.2  # Width of each bar

                # Create the plot 
                labels = [r"BASELINE", r"LSS DECAY", r"$ V_\lambda $ LSS"]
                ax.bar(x - width, group1, width, label=labels[0] + f", mean {sum(group1)/4:1.2f}", edgecolor='black', color=colors[0])
                ax.bar(x, group2, width, label=labels[1] + f", mean {sum(group2)/4:1.2f}", edgecolor='black', color=colors[1])
                ax.bar(x + width, group3, width, label=labels[2] + f", mean {sum(group3)/4:1.2f}", edgecolor='black', color=colors[2])
                ax.set_xlabel(r'$(\alpha, \beta)$')

                # ax.text(
                #     x=ax.get_xlim[1]/2,  # Position to the left of the y-axis
                #     y=ax.get_ylim[1] - 0.1,   # Vertical position in data coordinates
                #     s=f"{f:}",  # Annotation text
                #     va='center',              # Vertical alignment
                #     ha='center',               # Horizontal alignment
                #     fontsize=8,              # Font size
                #     color='black'              # Text color
                # )

                if i == 0:
                    ax.set_ylim(0., 1.0)

                if i == 1:
                    ax.set_ylim(0., 10.)

                if i == 2:
                    ax.set_ylim(1.)

                if i == 3:
                    ax.set_ylim(0., 25.)

                ax.grid(True, axis='y', alpha=0.7, zorder=0)
                ax.set_axisbelow(True)

                if i == 2:
                    ax.set_yscale('log')

                ax.set_title(data_name, fontsize=15)
                ax.set_xticks(x, categories)  # Replace x-ticks with category names
                ax.legend(loc='upper left')
            
            fig_bar.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.2)
            fig_bar.savefig("bar_" + save_path)
            plt.close()

            from PIL import Image

            # Open the saved images
            img1 = Image.open(save_path)
            img2 = Image.open("bar_" + save_path)

            # Combine the images side by side
            combined_width = img1.width + img2.width
            combined_height = max(img1.height, img2.height)
            combined_img = Image.new("RGB", (combined_width, combined_height))

            # Paste the two images
            combined_img.paste(img1, (0, 0))
            combined_img.paste(img2, (img1.width, 0))

            # Save the combined image
            combined_img.save("Final_Comparison_Plot_combined.png", dpi=(300, 300))
