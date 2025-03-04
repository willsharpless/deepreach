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
import scipy.io as spio

from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from collections import OrderedDict

from utils.error_evaluators import scenario_optimization, ValueThresholdValidator, MultiValidator, MLPConditionedValidator, target_fraction, MLP, MLPValidator, SliceSampleGenerator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns

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
            self.model.load_state_dict(torch.load(model_path)['model'])
        else:
            model_path = os.path.join(self.experiment_dir, 'training', 'checkpoints', 'model_epoch_%04d.pth' % epoch)
            self.model.load_state_dict(torch.load(model_path)['model'])

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
            record_temporal_loss = False, 
            deposit_blocking = True, deposit_blocking_period = 5000 # seg faults if nonblocking rn...
        ):
        was_eval = not self.model.training
        self.model.train()
        self.model.requires_grad_(True)

        train_dataloader = DataLoader(self.dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=0)

        ## Define Optimizers and Schedulers ## TODO, would SGD be better than Adam?
        if dual_lr and self.dataset.hopf_pretrain:
            optim_hopf = torch.optim.Adam(lr=lr_hopf, params=self.model.parameters())
            optim_std = torch.optim.Adam(lr=lr, params=self.model.parameters())
            lr_scheduler_hopf = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim_hopf, gamma=lr_hopf_decay_w)
            lr_scheduler_std = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim_std, gamma=lr_decay_w)
            optim = optim_hopf
            lr_scheduler = lr_scheduler_hopf
        else:
            optim = torch.optim.Adam(lr=lr, params=self.model.parameters())
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
        total_pretrain_iters = self.dataset.pretrain_iters + self.dataset.hopf_pretrain_iters
        self.total_pretrain_iters = total_pretrain_iters
        self.epochs = epochs
        nl_perc = 0.

        ## Dynamic Weighting
        rel_weight_hopf=1.0
        rel_weight_grad=1.0
        loss_weights = {'dirichlet': 1., 'hopf': 1., 'diff_constraint_hom': 1.}
        if diff_con_loss_incr:
            loss_weights['diff_constraint_hom'] = 0.
        if hopf_loss_decay_type == 'negative_exponential': 
            loss_weights['hopf'] = 1 - (hopf_loss_decay_w ** (epochs - 1 - total_pretrain_iters))
        if hopf_loss == 'lin_val_grad_diff':
            loss_weights['hopf_grad'] = loss_weights['hopf']
        og_loss_weights = loss_weights.copy()
        
        ## Train
        # with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA], 
        #               record_shapes=True, 
        #               profile_memory=True) as prof:
        with tqdm(total=len(train_dataloader) * epochs) as pbar:

            train_losses = []
            last_CSL_epoch = -1
            for epoch in range(0, epochs):

                if self.dataset.pretrain: # skip CSL
                    last_CSL_epoch = epoch
                time_interval_length = (self.dataset.counter/self.dataset.counter_end)*(self.dataset.tMax-self.dataset.tMin)
                CSL_tMax = self.dataset.tMin + int(time_interval_length/CSL_dt)*CSL_dt

                ## Reset Weights
                if reset_loss_w and epoch % reset_loss_period == 0 and not self.dataset.pretrain and not self.dataset.hopf_pretrain:
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
                not_pretraining = not(self.dataset.pretrain) and not(self.dataset.hopf_pretrain)
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
                    results_coord = self.dataset.dynamics.input_to_coord(model_results['model_in'].detach())
                    state_times, states = results_coord[..., 0], results_coord[..., 1:]
                    values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1))
                    dvs = self.dataset.dynamics.io_to_dv(model_results['model_in'], model_results['model_out'].squeeze(dim=-1))
                    boundary_values = gt['boundary_values']
                    if self.dataset.dynamics.loss_type == 'brat_hjivi':
                        reach_values = gt['reach_values']
                        avoid_values = gt['avoid_values']
                    dirichlet_masks = gt['dirichlet_masks']
                    if self.timing: print("Pre-loss Computation took:", time.time() - start_time_2)

                    if self.dataset.memory_tracking:
                        print(f"Epoch {epoch}-1, torch.cuda.memory_allocated: {torch.cuda.memory_allocated()/1000000:2.2f} MB")
                        print(f"Epoch {epoch}-1, torch.cuda.memory_reserved:  {torch.cuda.memory_reserved()/1000000:2.2f} MB")
                        print()  

                    ## Compute Loss
                    if self.timing: start_time_2 = time.time()

                    ## Standard BRT
                    if self.dataset.dynamics.loss_type == 'brt_hjivi':
                        losses = loss_fn(states, values, dvs[..., 0], dvs[..., 1:], boundary_values, dirichlet_masks, model_results['model_out'])
                    
                    ## Standard BRAT
                    elif self.dataset.dynamics.loss_type == 'brat_hjivi':
                        losses = loss_fn(states, values, dvs[..., 0], dvs[..., 1:], boundary_values, reach_values, avoid_values, dirichlet_masks, model_results['model_out'])
                    
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
                                    if not self.dataset.zerolambda_LS or self.dataset.pretrain or self.dataset.hopf_pretrain or self.dataset.hopf_pretrain_counter == self.dataset.hopf_pretrain_iters:
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
                                if not self.dataset.zerolambda_LS or self.dataset.pretrain or self.dataset.hopf_pretrain or self.dataset.hopf_pretrain_counter == self.dataset.hopf_pretrain_iters:
                                    loaded_model_results = self.dataset.loaded_model({'coords': model_input['model_coords'][..., :-1]}) # remove lambda
                                    # loaded_results_w_lambda = torch.cat((loaded_model_results['model_in'], torch.zeros(1, self.dataset.numpoints, 1).cuda()), dim=2) # put lambda back just for next line (removed b4 loss)
                                    # hopf_grads = self.dataset.dynamics.io_to_dv(loaded_model_results['model_in'], loaded_model_results['model_out'].squeeze(dim=-1)).detach()
                                    hopf_grads = self.dataset.loaded_dynamics.io_to_dv(loaded_model_results['model_in'], loaded_model_results['model_out'].squeeze(dim=-1))[..., 1:].detach()
                                else:
                                    loaded_model_results = self.dataset.loaded_model({'coords': gt['model_coords_hopf'][..., :-1]}) # remove lambda
                                    hopf_grads = self.dataset.loaded_dynamics.io_to_dv(loaded_model_results['model_in'], loaded_model_results['model_out'].squeeze(dim=-1))[..., 1:].detach()    
                            
                        if self.dataset.memory_tracking:
                            print(f"Epoch {epoch}-2, torch.cuda.memory_allocated: {torch.cuda.memory_allocated()/1000000:2.2f} MB")
                            print(f"Epoch {epoch}-2, torch.cuda.memory_reserved:  {torch.cuda.memory_reserved()/1000000:2.2f} MB")
                            print()

                        ## Seperate Hopf Coords (for the hopf loss to allow unrestricted sampling for PDE loss OR supervision on lambda=0 data only)
                        if (not self.dataset.use_bank or self.dataset.hopf_pretrain_counter == 0) and (not self.dataset.zerolambda_LS or (self.dataset.pretrain or self.dataset.hopf_pretrain or self.dataset.hopf_pretrain_counter == self.dataset.hopf_pretrain_iters)):
                            
                            learned_hopf_values = values
                            if hopf_loss == 'lin_val_grad_diff':
                                if not self.dataset.lambda_var:
                                    learned_hopf_grads = dvs[..., 1:]
                                else:
                                    learned_hopf_grads = dvs[..., 1:-1] # remove lambda grad
                        else:
                            model_results_hopf = self.model({'coords': gt['model_coords_hopf']})
                            learned_hopf_values = self.dataset.dynamics.io_to_value(model_results_hopf['model_in'].detach(), model_results_hopf['model_out'].squeeze(dim=-1))
                            
                            if hopf_loss == 'lin_val_grad_diff':
                                if not self.dataset.lambda_var:
                                    learned_hopf_grads = self.dataset.dynamics.io_to_dv(model_results_hopf['model_in'], model_results_hopf['model_out'].squeeze(dim=-1))[..., 1:]
                                else:
                                    learned_hopf_grads = self.dataset.dynamics.io_to_dv(model_results_hopf['model_in'], model_results_hopf['model_out'].squeeze(dim=-1))[..., 1:-1] # remove lambda grad
                        
                        if self.dataset.memory_tracking:
                            print(f"Epoch {epoch}-3, torch.cuda.memory_allocated: {torch.cuda.memory_allocated()/1000000:2.2f} MB")
                            print(f"Epoch {epoch}-3, torch.cuda.memory_reserved:  {torch.cuda.memory_reserved()/1000000:2.2f} MB")
                            print()

                        if hopf_loss == 'lin_val_grad_diff':
                            losses = loss_fn(states, values, dvs[..., 0], dvs[..., 1:], boundary_values, dirichlet_masks, model_results['model_out'], hopf_values, learned_hopf_values, hopf_grads, learned_hopf_grads, epoch, state_times)
                        else:
                            losses = loss_fn(states, values, dvs[..., 0], dvs[..., 1:], boundary_values, dirichlet_masks, model_results['model_out'], hopf_values, learned_hopf_values, epoch, state_times)
                            # losses = loss_fn_baseline(states, values, dvs[..., 0], dvs[..., 1:], boundary_values, dirichlet_masks, model_results['model_out'])
                            # print("\nUsing the loaded hopf values")

                    else:
                        raise NotImplementedError
                    
                    if self.dataset.memory_tracking:
                        print(f"Epoch {epoch}-4, torch.cuda.memory_allocated: {torch.cuda.memory_allocated()/1000000:2.2f} MB")
                        print(f"Epoch {epoch}-4, torch.cuda.memory_reserved:  {torch.cuda.memory_reserved()/1000000:2.2f} MB")
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
                    
                    if self.dataset.memory_tracking:
                        print(f"Epoch {epoch}-5, torch.cuda.memory_allocated: {torch.cuda.memory_allocated()/1000000:2.2f} MB")
                        print(f"Epoch {epoch}-5, torch.cuda.memory_reserved:  {torch.cuda.memory_reserved()/1000000:2.2f} MB")
                        print()

                    ## Switch Optimizers/Rates (after Hopf Pretraining)
                    if self.timing: start_time_2 = time.time()
                    if dual_lr and not(self.dataset.hopf_pretrain) and self.dataset.hopf_pretrained:
                        optim = optim_std
                        lr_scheduler = lr_scheduler_std
                    if self.timing: print("Loss Scheduler took:", time.time() - start_time_2)
                    
                    if self.dataset.memory_tracking:
                        print(f"Epoch {epoch}-6, torch.cuda.memory_allocated: {torch.cuda.memory_allocated()/1000000:2.2f} MB")
                        print(f"Epoch {epoch}-6, torch.cuda.memory_reserved:  {torch.cuda.memory_reserved()/1000000:2.2f} MB")
                        print()

                    ## Decay Hopf Loss(es)
                    if hopf_loss_decay and hopf_loss != 'none': #  
                        if adjust_relative_grads:
                            if epoch > total_pretrain_iters or hopf_loss_decay_early:
                                loss_weights['diff_constraint_hom']=1.0
                                params = OrderedDict(self.model.named_parameters())
                                # Gradients with respect to the PDE loss
                                optim.zero_grad()

                                losses['diff_constraint_hom'].backward(
                                    retain_graph=True)

                                grads_PDE = []
                                for key, param in params.items():
                                    grads_PDE.append(param.grad.view(-1))
                                grads_PDE = torch.cat(grads_PDE)

                                # Gradients with respect to the hopf loss
                                optim.zero_grad()
                                losses['hopf'].backward(retain_graph=True)
                                grads_hopf = []
                                for key, param in params.items():
                                    grads_hopf.append(param.grad.view(-1))
                                grads_hopf = torch.cat(grads_hopf)
                                # Set the new weight according to the paper
                                # num = torch.max(torch.abs(grads_PDE))
                                num = torch.mean(torch.abs(grads_hopf))
                                den = torch.mean(torch.abs(grads_PDE))
                                
                                hopf_importance_coef = hopf_loss_decay_w * math.e**(math.log(10/hopf_loss_decay_w)*(1-(epoch - total_pretrain_iters)/(epochs - 1 - total_pretrain_iters))) # decaying from 10 to decay_w
                                    
                                rel_weight_hopf = 0.9*rel_weight_hopf + 0.1*hopf_importance_coef*num/den
                                loss_weights['hopf'] = rel_weight_hopf 
                                
                                # Gradients with respect to the hopf loss
                                optim.zero_grad()
                                losses['hopf_grad'].backward(retain_graph=True)
                                grads_hopf_grad = []
                                for key, param in params.items():
                                    grads_hopf_grad.append(param.grad.view(-1))
                                grads_hopf_grad = torch.cat(grads_hopf_grad)     
                                num = torch.mean(torch.abs(grads_hopf_grad))
                                rel_weight_grad = 0.9*rel_weight_grad + 0.1*hopf_importance_coef*num/den
                                loss_weights['hopf_grad'] = rel_weight_grad 
                        else:                              
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
                        print(f"Epoch {epoch}-7, torch.cuda.memory_allocated: {torch.cuda.memory_allocated()/1000000:2.2f} MB")
                        print(f"Epoch {epoch}-7, torch.cuda.memory_reserved:  {torch.cuda.memory_reserved()/1000000:2.2f} MB")
                        print()

                    ## Combine Losses
                    if self.timing: start_time_2 = time.time()
                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean() ## TODO: this is not the right place for this
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
                        print(f"Epoch {epoch}-8, torch.cuda.memory_allocated: {torch.cuda.memory_allocated()/1000000:2.2f} MB")
                        print(f"Epoch {epoch}-8, torch.cuda.memory_reserved:  {torch.cuda.memory_reserved()/1000000:2.2f} MB")
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
                        print(f"Epoch {epoch}-9, torch.cuda.memory_allocated: {torch.cuda.memory_allocated()/1000000:2.2f} MB")
                        print(f"Epoch {epoch}-9, torch.cuda.memory_reserved:  {torch.cuda.memory_reserved()/1000000:2.2f} MB")
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
                                'train_loss': train_loss}
                            
                            for loss_name, loss in losses.items():
                                log_dict[loss_name + "_loss"] = loss

                            if nonlin_scale and not(self.dataset.pretrain) and not(self.dataset.hopf_pretrain):
                                log_dict["Nonlinearity Scale"] = nl_perc

                            if self.dataset.record_gt_metrics:

                                log_dict["Jaccard Index over Time"] = JIp
                                log_dict["Smooth Jaccard Index over Time"] = JIp_s
                                log_dict["Max Smooth Jaccard Index over Time"] = JIp_s_max
                                log_dict["Falsely Included percent over Time"] = FIp
                                log_dict["Falsely Excluded percent over Time"] = FEp
                                log_dict["Mean Absolute Spatial Gradient"] = torch.abs(dvs[..., 1:]).sum() / (self.dataset.numpoints * self.N)
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
                        print(f"Epoch {epoch}-end, torch.cuda.memory_allocated: {torch.cuda.memory_allocated()/1000000:2.2f} MB")
                        print(f"Epoch {epoch}-end, torch.cuda.memory_reserved:  {torch.cuda.memory_reserved()/1000000:2.2f} MB")
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
                    
                    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % (epoch+1)),
                        np.array(train_losses))
                    self.validate(
                        # epoch=epoch+1, save_path=os.path.join(checkpoints_dir, 'BRS_validation_plot_epoch_%04d.png' % (epoch+1)),
                        epoch=epoch+1, save_path=os.path.join(checkpoints_dir, 'BRS_validation_plot.png'), # overwriting to save data
                        x_resolution = val_x_resolution, y_resolution = val_y_resolution, z_resolution=val_z_resolution, time_resolution=val_time_resolution)
                if self.timing: print("Checkpointing took:", time.time() - start_time_2)
        checkpoint = { 
                        'epoch': epoch+1,
                        'model': self.model.state_dict(),
                        'optimizer': optim.state_dict()}
        torch.save(checkpoint,
                        os.path.join(checkpoints_dir, 'model_final.pth'))
        # print("\n PROFILER RESULTS \n")
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        # print()

        if was_eval:
            self.model.eval()
            self.model.requires_grad_(False)

    def test(self, current_time, last_checkpoint, checkpoint_dt, dt, num_scenarios, num_violations, set_type, control_type, data_step, checkpoint_toload=None, lambda_var=False):
        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)
        if data_step in ["plot_basic_recovery", 'run_basic_recovery', 'plot_hists', 'run_robust_recovery', 'plot_robust_recovery']:
            testing_dir = self.experiment_dir
        else:
            testing_dir = os.path.join(
                self.experiment_dir, 'testing_%s' % current_time.strftime('%m_%d_%Y_%H_%M'))
            if os.path.exists(testing_dir):
                overwrite = input(
                    "The testing directory %s already exists. Overwrite? (y/n)" % testing_dir)
                if not (overwrite == 'y'):
                    print('Exiting.')
                    quit()
                shutil.rmtree(testing_dir)
            os.makedirs(testing_dir)

        if checkpoint_toload is None:
            print('running cross-checkpoint testing')

            # checkpoint x simulation_time square matrices
            sidelen = 10
            assert (last_checkpoint /
                    checkpoint_dt) % sidelen == 0, 'checkpoints cannot be even divided by sidelen'
            BRT_volumes_matrix = np.zeros((sidelen, sidelen))
            BRT_errors_matrix = np.zeros((sidelen, sidelen))
            BRT_error_rates_matrix = np.zeros((sidelen, sidelen))
            BRT_error_region_fracs_matrix = np.zeros((sidelen, sidelen))

            exBRT_volumes_matrix = np.zeros((sidelen, sidelen))
            exBRT_errors_matrix = np.zeros((sidelen, sidelen))
            exBRT_error_rates_matrix = np.zeros((sidelen, sidelen))
            exBRT_error_region_fracs_matrix = np.zeros((sidelen, sidelen))

            checkpoints = np.linspace(0, last_checkpoint, num=sidelen+1)[1:]
            checkpoints[-1] = -1
            times = np.linspace(self.dataset.tMin,
                                self.dataset.tMax, num=sidelen+1)[1:]
            print('constructing matrices for')
            print('checkpoints:', checkpoints)
            print('times:', times)
            for i in tqdm(range(sidelen), desc='Checkpoint'):
                self._load_checkpoint(epoch=checkpoints[i])
                for j in tqdm(range(sidelen), desc='Simulation Time', leave=False):
                    # get BRT volume, error, error rate, error region fraction
                    results = scenario_optimization(
                        model=self.model, dynamics=self.dataset.dynamics, tMin=self.dataset.tMin, t=times[
                            j], dt=dt,
                        set_type=set_type, control_type=control_type,
                        scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 1000000),
                        sample_validator=ValueThresholdValidator(
                            v_min=float('-inf'), v_max=0.0),
                        violation_validator=ValueThresholdValidator(
                            v_min=0.0, v_max=float('inf')),
                        max_scenarios=num_scenarios, max_samples=1000*num_scenarios)
                    BRT_volumes_matrix[i, j] = results['valid_sample_fraction']
                    if results['maxed_scenarios']:
                        BRT_errors_matrix[i,
                                          j] = results['max_violation_error']
                        BRT_error_rates_matrix[i,
                                               j] = results['violation_rate']
                        BRT_error_region_fracs_matrix[i, j] = target_fraction(
                            model=self.model, dynamics=self.dataset.dynamics, t=times[j],
                            sample_validator=ValueThresholdValidator(
                                v_min=float('-inf'), v_max=0.0),
                            target_validator=ValueThresholdValidator(
                                v_min=-results['max_violation_error'], v_max=0.0),
                            num_samples=num_scenarios, batch_size=min(10*num_scenarios, 1000000))
                    else:
                        BRT_errors_matrix[i, j] = np.NaN
                        BRT_error_rates_matrix[i, j] = np.NaN
                        BRT_error_region_fracs_matrix[i, j] = np.NaN

                    # get exBRT error, error rate, error region fraction
                    results = scenario_optimization(
                        model=self.model, dynamics=self.dataset.dynamics, tMin=self.dataset.tMin, t=times[
                            j], dt=dt,
                        set_type=set_type, control_type=control_type,
                        scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 1000000),
                        sample_validator=ValueThresholdValidator(
                            v_min=0.0, v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(
                            v_min=float('-inf'), v_max=0.0),
                        max_scenarios=num_scenarios, max_samples=1000*num_scenarios)
                    exBRT_volumes_matrix[i,
                                         j] = results['valid_sample_fraction']
                    if results['maxed_scenarios']:
                        exBRT_errors_matrix[i,
                                            j] = results['max_violation_error']
                        exBRT_error_rates_matrix[i,
                                                 j] = results['violation_rate']
                        exBRT_error_region_fracs_matrix[i, j] = target_fraction(
                            model=self.model, dynamics=self.dataset.dynamics, t=times[j],
                            sample_validator=ValueThresholdValidator(
                                v_min=0.0, v_max=float('inf')),
                            target_validator=ValueThresholdValidator(
                                v_min=0.0, v_max=results['max_violation_error']),
                            num_samples=num_scenarios, batch_size=min(10*num_scenarios, 1000000))
                    else:
                        exBRT_errors_matrix[i, j] = np.NaN
                        exBRT_error_rates_matrix[i, j] = np.NaN
                        exBRT_error_region_fracs_matrix[i, j] = np.NaN

            # save the matrices
            matrices = {
                'BRT_volumes_matrix': BRT_volumes_matrix,
                'BRT_errors_matrix': BRT_errors_matrix,
                'BRT_error_rates_matrix': BRT_error_rates_matrix,
                'BRT_error_region_fracs_matrix': BRT_error_region_fracs_matrix,
                'exBRT_volumes_matrix': exBRT_volumes_matrix,
                'exBRT_errors_matrix': exBRT_errors_matrix,
                'exBRT_error_rates_matrix': exBRT_error_rates_matrix,
                'exBRT_error_region_fracs_matrix': exBRT_error_region_fracs_matrix,
            }
            for name, arr in matrices.items():
                with open(os.path.join(testing_dir, f'{name}.npy'), 'wb') as f:
                    np.save(f, arr)

            # plot the matrices
            matrices = {
                'BRT_volumes_matrix': [
                    BRT_volumes_matrix, 'BRT Fractions of Test State Space'
                ],
                'BRT_errors_matrix': [
                    BRT_errors_matrix, 'BRT Errors'
                ],
                'BRT_error_rates_matrix': [
                    BRT_error_rates_matrix, 'BRT Error Rates'
                ],
                'BRT_error_region_fracs_matrix': [
                    BRT_error_region_fracs_matrix, 'BRT Error Region Fractions'
                ],
                'exBRT_volumes_matrix': [
                    exBRT_volumes_matrix, 'exBRT Fractions of Test State Space'
                ],
                'exBRT_errors_matrix': [
                    exBRT_errors_matrix, 'exBRT Errors'
                ],
                'exBRT_error_rates_matrix': [
                    exBRT_error_rates_matrix, 'exBRT Error Rates'
                ],
                'exBRT_error_region_fracs_matrix': [
                    exBRT_error_region_fracs_matrix, 'exBRT Error Region Fractions'
                ],
            }
            for name, data in matrices.items():
                cmap = matplotlib.cm.get_cmap('Reds')
                cmap.set_bad(color='blue')
                fig, ax = plt.subplots(1, 1)
                ax.set_xticks(range(sidelen))
                ax.set_yticks(range(sidelen))
                ax.set_xticklabels(np.round_(times, decimals=2))
                ax.set_yticklabels(np.linspace(
                    0, last_checkpoint, num=sidelen+1)[1:])
                plt.xlabel('Simulation Time')
                plt.ylabel('Checkpoint')
                ax.imshow(data[0], cmap=cmap)
                plt.title(data[1])
                for (y, x), label in np.ndenumerate(data[0]):
                    plt.text(x, y, '%.7f' %
                             label, ha='center', va='center', fontsize=4)
                plt.savefig(os.path.join(testing_dir, name + '.png'), dpi=600)
                plt.clf()
                # log version
                cmap = matplotlib.cm.get_cmap('Reds')
                cmap.set_bad(color='blue')
                fig, ax = plt.subplots(1, 1)
                ax.set_xticks(range(sidelen))
                ax.set_yticks(range(sidelen))
                ax.set_xticklabels(np.round_(times, decimals=2))
                ax.set_yticklabels(np.linspace(
                    0, last_checkpoint, num=sidelen+1)[1:])
                plt.xlabel('Simulation Time')
                plt.ylabel('Checkpoint')
                new_matrix = np.log(data[0])
                ax.imshow(new_matrix, cmap=cmap)
                plt.title('(Log) ' + data[1])
                for (y, x), label in np.ndenumerate(new_matrix):
                    plt.text(x, y, '%.7f' %
                             label, ha='center', va='center', fontsize=4)
                plt.savefig(os.path.join(
                    testing_dir, name + '_log' + '.png'), dpi=600)
                plt.clf()

        else:
            print('running specific-checkpoint testing')
            self._load_checkpoint(checkpoint_toload)

            model = self.model
            dataset = self.dataset
            dynamics = dataset.dynamics

            if data_step == 'plot_violations':
                # plot violations on slice
                plot_config = dynamics.plot_config()
                slices = plot_config['state_slices']
                slices[plot_config['x_axis_idx']] = None
                slices[plot_config['y_axis_idx']] = None
                results = scenario_optimization(
                    model=model, dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=100000, sample_batch_size=100000,
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=slices),
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=100000, max_samples=1000000)
                plt.title('violations for slice = %s' %
                          plot_config['state_slices'], fontsize=8)
                plt.scatter(results['states'][..., plot_config['x_axis_idx']][~results['violations']], results['states']
                            [...,  plot_config['y_axis_idx']][~results['violations']], s=0.05, color=(0, 0, 1), marker='o')
                plt.scatter(results['states'][..., plot_config['x_axis_idx']][results['violations']], results['states']
                            [...,  plot_config['y_axis_idx']][results['violations']], s=0.05, color=(1, 0, 0), marker='o')
                x_min, x_max = dynamics.state_test_range()[
                    plot_config['x_axis_idx']]
                y_min, y_max = dynamics.state_test_range()[
                    plot_config['y_axis_idx']]
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
                plt.savefig(os.path.join(
                    testing_dir, f'violations.png'), dpi=800)
                plt.clf()

                # plot distribution of violations over state variables
                results = scenario_optimization(
                    model=model, dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=100000, sample_batch_size=100000,
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=100000, max_samples=1000000)
                for i in range(dynamics.state_dim):
                    plt.title('violations over %s' %
                              plot_config['state_labels'][i])
                    plt.scatter(results['states'][..., i][~results['violations']], results['values']
                                [~results['violations']], s=0.05, color=(0, 0, 1), marker='o')
                    plt.scatter(results['states'][..., i][results['violations']], results['values']
                                [results['violations']], s=0.05, color=(1, 0, 0), marker='o')
                    plt.savefig(os.path.join(
                        testing_dir, f'violations_over_state_dim_{i}.png'), dpi=800)
                    plt.clf()
            if data_step == 'plot_hists':
                logs = {}

                # rollout samples all over the state space
                beta = 1e-16
                epsilon = 1e-3
                N = int(math.ceil((2/epsilon)*(np.log(1/beta)+1)))
                M = 5

                logs['beta'] = beta
                logs['epsilon'] = epsilon
                logs['N'] = N
                logs['M'] = M

                delta_level = float(
                    'inf') if dynamics.set_mode == 'reach' else float('-inf')

                results = scenario_optimization(
                    model=model, dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=min(N, 100000), sample_batch_size=10*min(N, 10000),
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=ValueThresholdValidator(v_min=float(
                        '-inf'), v_max=delta_level) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=delta_level, v_max=float('inf')),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=N, max_samples=1000*min(N, 10000))

                sns.set_style('whitegrid')
                costs_ = results['costs'].cpu().numpy()
                values_ = results['values'].cpu().numpy()
                unsafe_cost_safe_value_indeces = np.argwhere(
                    np.logical_and(costs_ < 0, values_ >= 0))
                fig1 = plt.figure()
                positive_values_with_negative_cost = values_[
                    unsafe_cost_safe_value_indeces]
                unsafe_trajs = results['batch_state_trajs'].cpu().numpy()[
                    unsafe_cost_safe_value_indeces, ...]
                outlier_states = results['states'][unsafe_cost_safe_value_indeces, ...]
                torch.set_printoptions(
                    precision=2, threshold=10_000, sci_mode=False)
                
                # print(results['batch_state_trajs'].shape)
                print("False positive: ", unsafe_cost_safe_value_indeces.shape[0]/values_.shape[0], "False negative: ",
                        np.argwhere(np.logical_and(costs_ >= 0, values_ < 0)).shape[0]/values_.shape[0])
                # print(outlier_states[:200, ...])

                (vs, bins, patches) = plt.hist(
                    positive_values_with_negative_cost, bins=200)
                # plt.title("Learned values for actually unsafe states marked as safe\n%s" %
                #           dynamics.deepReach_model)
                fig1.savefig(os.path.join(
                    testing_dir, f'value distribution.png'), dpi=800)
                print("save path: ", os.path.join(
                    testing_dir, f'value distribution.png'))
                plt.close(fig1)

                # fig2=plt.figure()
                # unsafe_cost_value_indeces=np.argwhere(costs_<0)
                # plt.hist(values_[unsafe_cost_value_indeces],bins=200)
                # plt.title("Learned values for actually unsafe states marked as safe \n %s"%dynamics.deepReach_model)
                # fig2.savefig(os.path.join(testing_dir, f'all value distribution.png'), dpi=800)
                # plt.close(fig2)
                np.save(os.path.join(testing_dir, f'state_traj'),
                        unsafe_trajs)
                np.save(os.path.join(testing_dir, f'safe_traj'),
                        results['batch_state_trajs'].cpu().numpy()[costs_ < 0])

                np.save(os.path.join(testing_dir, f'value_data'),
                        positive_values_with_negative_cost)
                np.save(os.path.join(testing_dir, f'bins'), bins)
                np.save(os.path.join(testing_dir, f'vs'), vs)

            if data_step == 'plot_robust_recovery':
                epsilons=-np.load(os.path.join(testing_dir, f'epsilons.npy'))+1
                deltas=np.load(os.path.join(testing_dir, f'deltas.npy'))
                target_eps=0.01
                delta_level=deltas[np.argmin(np.abs(epsilons-target_eps))]
                fig,values_slices = self.plot_recovery_fig(
                    dataset, dynamics, model, delta_level)
                fig.savefig(os.path.join(
                    testing_dir, f'robust_BRTs_1e-2.png'), dpi=800)
                np.save(os.path.join(testing_dir, f'values_slices'),values_slices)

            if data_step == 'run_robust_recovery':
                logs = {}
                # rollout samples all over the state space
                beta_ = 1e-10
                N = 300000
                logs['beta_'] = beta_
                logs['N'] = N
                delta_level = float(
                    'inf') if dynamics.set_mode == 'reach' else float('-inf')

                results = scenario_optimization(
                    model=model, dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=min(N, 100000), sample_batch_size=10*min(N, 10000),
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=ValueThresholdValidator(v_min=float(
                        '-inf'), v_max=delta_level) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=delta_level, v_max=float('inf')),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=N, max_samples=1000*min(N, 10000))

                sns.set_style('whitegrid')
                costs_ = results['costs'].cpu().numpy()
                values_ = results['values'].cpu().numpy()
                unsafe_cost_safe_value_indeces = np.argwhere(
                    np.logical_and(costs_ < 0, values_ >= 0))

                print("k max: ", unsafe_cost_safe_value_indeces.shape[0])

                # determine delta_level_max
                delta_level_max = np.max(
                    values_[unsafe_cost_safe_value_indeces])
                print("delta_level_max: ", delta_level_max)

                # for each delta level, determine (1) the corresponding volume;
                # (2) k and and corresponding epsilon
                ks = []
                epsilons = []
                volumes = []

                for delta_level_ in np.arange(0, delta_level_max, delta_level_max/100):
                    k = int(np.argwhere(np.logical_and(
                        costs_ < 0, values_ >= delta_level_)).shape[0])
                    eps = beta__dist.ppf(beta_,  N-k, k+1)
                    volume = values_[values_ >= delta_level_].shape[0]/values_.shape[0]
                    
                    ks.append(k)
                    epsilons.append(eps)
                    volumes.append(volume)

                # plot epsilon volume graph
                fig1, ax1 = plt.subplots()
                color = 'tab:red'
                ax1.set_xlabel('volumes')
                ax1.set_ylabel('epsilons', color=color)
                ax1.plot(volumes, epsilons, color=color)
                ax1.tick_params(axis='y', labelcolor=color)

                ax2 = ax1.twinx()

                color = 'tab:blue'
                ax2.set_ylabel('number of outliers', color=color)
                ax2.plot(volumes, ks, color=color)
                ax2.tick_params(axis='y', labelcolor=color)

                plt.title("beta_=1e-10, N =3e6")
                fig1.savefig(os.path.join(
                    testing_dir, f'robust_verification_results.png'), dpi=800)
                plt.close(fig1)
                np.save(os.path.join(testing_dir, f'epsilons'),
                        epsilons)
                np.save(os.path.join(testing_dir, f'volumes'),
                        volumes)
                np.save(os.path.join(testing_dir, f'deltas'),
                        np.arange(0, delta_level_max, delta_level_max/100))
                np.save(os.path.join(testing_dir, f'ks'),
                        ks)
                
            if data_step == 'run_basic_recovery':
                logs = {}

                # 0. explicit statement of probabilistic guarantees, N, \beta, \epsilon
                beta = 1e-16
                epsilon = 1e-3
                N = int(math.ceil((2/epsilon)*(np.log(1/beta)+1)))
                M = 5

                logs['beta'] = beta
                logs['epsilon'] = epsilon
                logs['N'] = N
                logs['M'] = M

                # 1. execute algorithm for tMax
                # record state/learned_value/violation for each while loop iteration
                delta_level = float(
                    'inf') if dynamics.set_mode == 'reach' else float('-inf')
                algorithm_iters = []
                if lambda_var:
                    slices=[None]*(dynamics.state_dim-1)+[1]
                else:
                    slices=[None]*dynamics.state_dim
                for i in range(M):
                    print('algorithm iter', str(i))
                    results = scenario_optimization(
                        model=model, dynamics=dynamics,
                        tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                        set_type=set_type, control_type=control_type,
                        scenario_batch_size=min(N, 100000), sample_batch_size=10*min(N, 10000),
                        sample_generator=SliceSampleGenerator(
                            dynamics=dynamics, slices=slices),
                        sample_validator=ValueThresholdValidator(v_min=float(
                            '-inf'), v_max=delta_level) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=delta_level, v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                            'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                        max_scenarios=N, max_samples=1000*min(N, 10000))
                    if not results['maxed_scenarios']:
                        delta_level = float(
                            '-inf') if dynamics.set_mode == 'reach' else float('inf')
                        break
                    algorithm_iters.append(
                        {
                            'states': results['states'],
                            'values': results['values'],
                            'violations': results['violations']
                        }
                    )
                    if results['violation_rate'] == 0:
                        break
                    violation_levels = results['values'][results['violations']]
                    delta_level_arg = np.argmin(
                        violation_levels) if dynamics.set_mode == 'reach' else np.argmax(violation_levels)
                    delta_level = violation_levels[delta_level_arg].item()

                    print('violation_rate:', str(results['violation_rate']))
                    print('delta_level:', str(delta_level))
                    print('valid_sample_fraction:', str(
                        results['valid_sample_fraction'].item()))
                    sns.set_style('whitegrid')
                    # density_plot=sns.kdeplot(results['costs'].cpu().numpy(), bw=0.5)
                    # density_plot=sns.displot(results['costs'].cpu().numpy(), x="cost function")
                    # fig1 = density_plot.get_figure()
                    fig1 = plt.figure()
                    plt.hist(results['costs'].cpu().numpy(), bins=200)
                    fig1.savefig(os.path.join(
                        testing_dir, f'cost distribution.png'), dpi=800)
                    plt.close(fig1)
                    fig2 = plt.figure()
                    plt.hist(results['costs'].cpu().numpy() -
                             results['values'].cpu().numpy(), bins=200)
                    fig2.savefig(os.path.join(
                        testing_dir, f'diff distribution.png'), dpi=800)
                    
                    plt.close(fig1)

                logs['algorithm_iters'] = algorithm_iters
                logs['delta_level'] = delta_level

                # 2. record solution volume, recovered volume
                S = 1000000
                logs['S'] = S
                logs['learned_volume'] = target_fraction(
                    model=model, dynamics=dynamics, t=dataset.tMax,
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    target_validator=ValueThresholdValidator(v_min=float(
                        '-inf'), v_max=0.0) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=0.0, v_max=float('inf')),
                    num_samples=S,
                    batch_size=min(S, 1000000),
                    lambda_var=lambda_var,
                ).item()
                logs['recovered_volume'] = target_fraction(
                    model=model, dynamics=dynamics, t=dataset.tMax,
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    target_validator=ValueThresholdValidator(v_min=float(
                        '-inf'), v_max=delta_level) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=delta_level, v_max=float('inf')),
                    num_samples=S,
                    batch_size=min(S, 1000000),
                    lambda_var=lambda_var,
                ).item()

                results = scenario_optimization(
                    model=model, dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=min(S, 100000), sample_batch_size=10*min(S, 10000),
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=S, max_samples=1000*min(S, 10000))
                if results['maxed_scenarios']:
                    logs['theoretically_recoverable_volume'] = 1 - \
                        results['violation_rate']
                else:
                    logs['theoretically_recoverable_volume'] = 0

                print('learned_volume', str(logs['learned_volume']))
                print('recovered_volume', str(logs['recovered_volume']))
                print('theoretically_recoverable_volume', str(
                    logs['theoretically_recoverable_volume']))

                # 3. validate theoretical guarantees via mass sampling
                results = scenario_optimization(
                    model=model, dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=min(S, 100000), sample_batch_size=10*min(S, 10000),
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=ValueThresholdValidator(v_min=float(
                        '-inf'), v_max=delta_level) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=delta_level, v_max=float('inf')),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=S, max_samples=1000*min(S, 10000))
                if results['maxed_scenarios']:
                    logs['recovered_violation_rate'] = results['violation_rate']
                else:
                    logs['recovered_violation_rate'] = 0
                print('recovered_violation_rate', str(
                    logs['recovered_violation_rate']))

                with open(os.path.join(testing_dir, 'basic_logs.pickle'), 'wb') as f:
                    pickle.dump(logs, f)

            if data_step == 'plot_basic_recovery':
                with open(os.path.join(self.experiment_dir, 'basic_logs.pickle'), 'rb') as f:
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

                fig, values_slices = self.plot_recovery_fig(
                    dataset, dynamics, model, delta_level)

                plt.tight_layout()
                fig.savefig(os.path.join(
                    testing_dir, f'basic_BRTs.png'), dpi=800)

            if data_step == 'collect_samples':
                logs = {}

                # 1. record 10M state, learned value, violation
                P = int(1e7)
                logs['P'] = P
                print('collecting training samples')
                results = scenario_optimization(
                    model=model, dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=min(P, 100000), sample_batch_size=10*min(P, 10000),
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=P, max_samples=1000*min(P, 10000))
                logs['training_samples'] = {
                    'states': results['states'],
                    'values': results['values'],
                    'violations': results['violations'],
                }
                with open(os.path.join(testing_dir, 'sample_logs.pickle'), 'wb') as f:
                    pickle.dump(logs, f)

            if data_step == 'train_binner':
                with open(os.path.join(self.experiment_dir, 'sample_logs.pickle'), 'rb') as f:
                    logs = pickle.load(f)

                # 1. train MLP predictor
                # plot validation of MLP predictor
                def validate_predictor(predictor, epoch):
                    print('validating predictor at epoch', str(epoch))
                    predictor.eval()

                    results = scenario_optimization(
                        model=model, dynamics=dynamics,
                        tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                        set_type=set_type, control_type=control_type,
                        scenario_batch_size=100000, sample_batch_size=100000,
                        sample_generator=SliceSampleGenerator(
                            dynamics=dynamics, slices=[None]*dynamics.state_dim),
                        sample_validator=ValueThresholdValidator(
                            v_min=float('-inf'), v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                            'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                        max_scenarios=100000, max_samples=1000000)

                    inputs = torch.cat(
                        (results['states'], results['values'][..., None]), dim=-1)
                    preds = torch.sigmoid(
                        predictor(inputs.cuda())).detach().cpu().numpy()

                    plt.title(f'Predictor Validation at Epoch {epoch}')
                    plt.ylabel('Value')
                    plt.xlabel('Prediction')
                    plt.scatter(preds[~results['violations']], results['values']
                                [~results['violations']], color='blue', label='nonviolations', alpha=0.1)
                    plt.scatter(preds[results['violations']], results['values']
                                [results['violations']], color='red', label='violations', alpha=0.1)
                    plt.legend()
                    plt.savefig(os.path.join(
                        testing_dir, f'predictor_validation_at_epoch_{epoch}.png'), dpi=800)
                    plt.clf()

                    predictor.train()

                print('training predictor')
                violation_scale = 5
                violation_weight = 1.5
                states = logs['training_samples']['states']
                values = logs['training_samples']['values']
                violations = logs['training_samples']['violations']
                violation_strengths = torch.where(violations, (torch.max(
                    values[violations]) - values) if dynamics.set_mode == 'reach' else (values - torch.min(values[violations])), torch.tensor([0.0])).cuda()
                violation_scales = torch.exp(
                    violation_scale * violation_strengths / torch.max(violation_strengths))

                plt.title(f'Violation Scales')
                plt.ylabel('Frequency')
                plt.xlabel('Scale')
                plt.hist(violation_scales.cpu().numpy(), range=(0, 10))
                plt.savefig(os.path.join(
                    testing_dir, f'violation_scales.png'), dpi=800)
                plt.clf()

                inputs = torch.cat((states, values[..., None]), dim=-1).cuda()
                outputs = 1.0*violations.cuda()
                # outputs = violation_strengths / torch.max(violation_strengths)

                predictor = MLP(input_size=dynamics.state_dim+1)
                predictor.cuda()
                predictor.train()

                lr = 0.00005
                lr_decay = 0.2
                decay_patience = 20
                decay_threshold = 1e-12
                opt = torch.optim.Adam(predictor.parameters(), lr=lr)
                sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, factor=lr_decay, patience=decay_patience, threshold=decay_threshold)

                pos_weight = violation_weight * \
                    ((outputs <= 0).sum() / (outputs > 0).sum())

                n_epochs = 1000
                batch_size = 100000
                for epoch in range(n_epochs):
                    idxs = torch.randperm(len(outputs))
                    for batch in range(math.ceil(len(outputs) / batch_size)):
                        batch_idxs = idxs[batch *
                                          batch_size: (batch+1)*batch_size]

                        BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss(
                            weight=violation_scales[batch_idxs], pos_weight=pos_weight)
                        loss = BCEWithLogitsLoss(
                            predictor(inputs[batch_idxs]).squeeze(dim=-1), outputs[batch_idxs])
                        # MSELoss = torch.nn.MSELoss()
                        # loss = MSELoss(predictor(inputs[batch_idxs]).squeeze(dim=-1), outputs[batch_idxs])
                        loss.backward()
                        opt.step()
                    print(f'Epoch {epoch}: loss: {loss.item()}')
                    sched.step(loss.item())

                    if (epoch+1) % 100 == 0:
                        torch.save(predictor.state_dict(), os.path.join(
                            testing_dir, f'predictor_at_epoch_{epoch}.pth'))
                        validate_predictor(predictor, epoch)
                logs['violation_scale'] = violation_scale
                logs['violation_weight'] = violation_weight
                logs['n_epochs'] = n_epochs
                logs['lr'] = lr
                logs['lr_decay'] = lr_decay
                logs['decay_patience'] = decay_patience
                logs['decay_threshold'] = decay_threshold

                with open(os.path.join(testing_dir, 'train_logs.pickle'), 'wb') as f:
                    pickle.dump(logs, f)

            if data_step == 'run_binned_recovery':
                logs = {}

                # 0. explicit statement of probabilistic guarantees, N, \beta, \epsilon
                beta = 1e-16
                epsilon = 1e-3
                N = int(math.ceil((2/epsilon)*(np.log(1/beta)+1)))
                M = 5

                logs['beta'] = beta
                logs['epsilon'] = epsilon
                logs['N'] = N
                logs['M'] = M

                # 1. execute algorithm for each bin of MLP predictor
                epoch = 699
                logs['epoch'] = epoch
                predictor = MLP(input_size=dynamics.state_dim+1)
                predictor.load_state_dict(torch.load(os.path.join(
                    self.experiment_dir, f'predictor_at_epoch_{epoch}.pth')))
                predictor.cuda()
                predictor.train()

                bins = [0, 0.8, 0.85, 0.9, 0.95, 1]
                logs['bins'] = bins

                binned_delta_levels = []
                for i in range(len(bins)-1):
                    print('bin', str(i))
                    binned_delta_level = float(
                        'inf') if dynamics.set_mode == 'reach' else float('-inf')
                    for j in range(M):
                        print('algorithm iter', str(j))
                        results = scenario_optimization(
                            model=model, dynamics=dynamics,
                            tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                            set_type=set_type, control_type=control_type,
                            scenario_batch_size=min(N, 100000), sample_batch_size=10*min(N, 10000),
                            sample_generator=SliceSampleGenerator(
                                dynamics=dynamics, slices=[None]*dynamics.state_dim),
                            sample_validator=MultiValidator([
                                MLPValidator(
                                    mlp=predictor, o_min=bins[i], o_max=bins[i+1], model=model, dynamics=dynamics),
                                ValueThresholdValidator(v_min=float(
                                    '-inf'), v_max=binned_delta_level) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=binned_delta_level, v_max=float('inf')),
                            ]),
                            violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                                'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                            max_scenarios=N, max_samples=100000*min(N, 10000))
                        if not results['maxed_scenarios']:
                            binned_delta_level = float(
                                '-inf') if dynamics.set_mode == 'reach' else float('inf')
                            break
                        if results['violation_rate'] == 0:
                            break
                        violation_levels = results['values'][results['violations']]
                        binned_delta_level_arg = np.argmin(
                            violation_levels) if dynamics.set_mode == 'reach' else np.argmax(violation_levels)
                        binned_delta_level = violation_levels[binned_delta_level_arg].item(
                        )
                        print('violation_rate:', str(
                            results['violation_rate']))
                        print('binned_delta_level:', str(binned_delta_level))
                        print('valid_sample_fraction:', str(
                            results['valid_sample_fraction'].item()))
                    binned_delta_levels.append(binned_delta_level)
                logs['binned_delta_levels'] = binned_delta_levels

                # 2. record solution volume, auto-binned recovered volume
                S = 1000000
                logs['S'] = S
                logs['learned_volume'] = target_fraction(
                    model=model, dynamics=dynamics, t=dataset.tMax,
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    target_validator=ValueThresholdValidator(v_min=float(
                        '-inf'), v_max=0.0) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=0.0, v_max=float('inf')),
                    num_samples=S,
                    batch_size=min(S, 1000000),
                ).item()
                logs['binned_volume'] = target_fraction(
                    model=model, dynamics=dynamics, t=dataset.tMax,
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    target_validator=MLPConditionedValidator(
                        mlp=predictor,
                        o_levels=bins,
                        v_levels=[[float('-inf'), binned_delta_level] if dynamics.set_mode == 'reach' else [
                            binned_delta_level, float('inf')] for binned_delta_level in binned_delta_levels],
                        model=model,
                        dynamics=dynamics,
                    ),
                    num_samples=S,
                    batch_size=min(S, 1000000),
                ).item()
                results = scenario_optimization(
                    model=model, dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=min(S, 100000), sample_batch_size=10*min(S, 10000),
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=S, max_samples=1000*min(S, 10000))
                if results['maxed_scenarios']:
                    logs['theoretically_recoverable_volume'] = 1 - \
                        results['violation_rate']
                else:
                    logs['theoretically_recoverable_volume'] = 0
                print('learned_volume', str(logs['learned_volume']))
                print('binned_volume', str(logs['binned_volume']))
                print('theoretically_recoverable_volume', str(
                    logs['theoretically_recoverable_volume']))

                # 3. validate theoretical guarantees via mass sampling
                results = scenario_optimization(
                    model=model, dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=min(S, 100000), sample_batch_size=10*min(S, 10000),
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=MLPConditionedValidator(
                        mlp=predictor,
                        o_levels=bins,
                        v_levels=[[float('-inf'), binned_delta_level] if dynamics.set_mode == 'reach' else [
                            binned_delta_level, float('inf')] for binned_delta_level in binned_delta_levels],
                        model=model,
                        dynamics=dynamics,
                    ),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=S, max_samples=1000*min(S, 10000))
                if results['maxed_scenarios']:
                    logs['binned_violation_rate'] = results['violation_rate']
                else:
                    logs['binned_violation_rate'] = 0
                print('binned_violation_rate', str(
                    logs['binned_violation_rate']))

                with open(os.path.join(testing_dir, 'binned_logs.pickle'), 'wb') as f:
                    pickle.dump(logs, f)

            if data_step == 'plot_binned_recovery':
                with open(os.path.join(self.experiment_dir, 'binned_logs.pickle'), 'rb') as f:
                    logs = pickle.load(f)

                # 0.
                print('N:', str(logs['N']))
                print('M:', str(logs['M']))
                print('beta:', str(logs['beta']))
                print('epsilon:', str(logs['epsilon']))
                # print('P:', str(logs['P']))
                print('S:', str(logs['S']))
                print('bins', str(logs['bins']))
                bins = logs['bins']
                print('binned delta levels', str(logs['binned_delta_levels']))
                binned_delta_levels = logs['binned_delta_levels']
                print('learned volume', str(logs['learned_volume']))
                print('binned volume', str(logs['binned_volume']))
                print('theoretically recoverable volume', str(
                    logs['theoretically_recoverable_volume']))
                print('binned violation rate', str(
                    logs['binned_violation_rate']))

                epoch = logs['epoch']
                predictor = MLP(input_size=dynamics.state_dim+1)
                predictor.load_state_dict(torch.load(os.path.join(
                    self.experiment_dir, f'predictor_at_epoch_{epoch}.pth')))
                predictor.cuda()
                predictor.eval()

                # 1. for ground truth slices (if available), record (higher-res) grid of learned values and MLP predictions
                # plot (with ground truth) learned BRTs, auto-binned recovered BRTs
                # plot MLP predictor bins
                z_res = 5
                plot_config = dataset.dynamics.plot_config()
                if os.path.exists(os.path.join(self.experiment_dir, 'ground_truth.mat')):
                    ground_truth = spio.loadmat(os.path.join(
                        self.experiment_dir, 'ground_truth.mat'))
                    if 'gmat' in ground_truth:
                        ground_truth_xs = ground_truth['gmat'][..., 0][:, 0, 0]
                        ground_truth_ys = ground_truth['gmat'][..., 1][0, :, 0]
                        ground_truth_zs = ground_truth['gmat'][..., 2][0, 0, :]
                        ground_truth_values = ground_truth['data']
                        ground_truth_ts = np.linspace(
                            0, 1, ground_truth_values.shape[3])
                    elif 'g' in ground_truth:
                        ground_truth_xs = ground_truth['g']['vs'][0,
                                                                  0][0][0][:, 0]
                        ground_truth_ys = ground_truth['g']['vs'][0,
                                                                  0][1][0][:, 0]
                        ground_truth_zs = ground_truth['g']['vs'][0,
                                                                  0][2][0][:, 0]
                        ground_truth_ts = ground_truth['tau'][0]
                        ground_truth_values = ground_truth['data']

                    # idxs to plot
                    x_idxs = np.linspace(
                        0, len(ground_truth_xs)-1, len(ground_truth_xs)).astype(dtype=int)
                    y_idxs = np.linspace(
                        0, len(ground_truth_ys)-1, len(ground_truth_ys)).astype(dtype=int)
                    z_idxs = np.linspace(
                        0, len(ground_truth_zs)-1, z_res).astype(dtype=int)
                    t_idxs = np.array(
                        [len(ground_truth_ts)-1]).astype(dtype=int)

                    # indexed ground truth to plot
                    ground_truth_xs = ground_truth_xs[x_idxs]
                    ground_truth_ys = ground_truth_ys[y_idxs]
                    ground_truth_zs = ground_truth_zs[z_idxs]
                    ground_truth_ts = ground_truth_ts[t_idxs]
                    ground_truth_values = ground_truth_values[
                        x_idxs[:, None, None, None],
                        y_idxs[None, :, None, None],
                        z_idxs[None, None, :, None],
                        t_idxs[None, None, None, :]
                    ]
                    ground_truth_grids = ground_truth_values
                    xs = ground_truth_xs
                    ys = ground_truth_ys
                    zs = ground_truth_zs

                else:
                    ground_truth_grids = None
                    resolution = 512
                    xs = np.linspace(*dynamics.state_test_range()
                                     [plot_config['x_axis_idx']], resolution)
                    ys = np.linspace(*dynamics.state_test_range()
                                     [plot_config['y_axis_idx']], resolution)
                    zs = np.linspace(*dynamics.state_test_range()
                                     [plot_config['z_axis_idx']], z_res)

                xys = torch.cartesian_prod(torch.tensor(xs), torch.tensor(ys))
                value_grids = np.zeros((len(zs), len(xs), len(ys)))
                prediction_grids = np.zeros((len(zs), len(xs), len(ys)))
                for i in range(len(zs)):
                    coords = torch.zeros(
                        xys.shape[0], dataset.dynamics.state_dim + 1)
                    coords[:, 0] = dataset.tMax
                    coords[:, 1:] = torch.tensor(plot_config['state_slices'])
                    coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
                    coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
                    coords[:, 1 + plot_config['z_axis_idx']] = zs[i]

                    model_results = model(
                        {'coords': dataset.dynamics.coord_to_input(coords.cuda())})
                    values = dataset.dynamics.io_to_value(model_results['model_in'].detach(
                    ), model_results['model_out'].detach().squeeze(dim=-1)).detach().cpu()
                    value_grids[i] = values.reshape(len(xs), len(ys))

                    inputs = torch.cat(
                        (coords[..., 1:], values[:, None]), dim=-1)
                    outputs = torch.sigmoid(
                        predictor(inputs.cuda()).cpu().squeeze(dim=-1))
                    prediction_grids[i] = outputs.reshape(
                        len(xs), len(ys)).detach().cpu()

                def overlay_ground_truth(image, z_idx):
                    thickness = max(0, image.shape[0] // 120 - 1)
                    ground_truth_grid = ground_truth_grids[:, :, z_idx, 0]
                    ground_truth_BRTs = ground_truth_grid < 0
                    for x in range(ground_truth_BRTs.shape[0]):
                        for y in range(ground_truth_BRTs.shape[1]):
                            if not ground_truth_BRTs[x, y]:
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
                                if neighbor[0] >= 0 and neighbor[1] >= 0 and neighbor[0] < ground_truth_BRTs.shape[0] and neighbor[1] < ground_truth_BRTs.shape[1]:
                                    if not ground_truth_BRTs[neighbor]:
                                        image[x-thickness:x+1+thickness, y-thickness:y +
                                              1+thickness] = np.array([50, 50, 50])
                                        break

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

                fig = plt.figure()
                fig.suptitle(plot_config['state_slices'], fontsize=8)
                for i in range(len(zs)):
                    values = value_grids[i]
                    predictions = prediction_grids[i]

                    # learned BRT and bin-recovered BRT
                    ax = fig.add_subplot(1, len(zs), (i+1))
                    ax.set_title('%s = %0.2f' % (
                        plot_config['state_labels'][plot_config['z_axis_idx']], zs[i]), fontsize=8)

                    image = np.full((*values.shape, 3), 255, dtype=int)
                    BRT = values < 0
                    bin_recovered_BRT = np.full(values.shape, 0, dtype=bool)
                    # per bin, set accordingly
                    for j in range(len(bins)-1):
                        mask = (
                            (predictions >= bins[j])*(predictions < bins[j+1]))
                        binned_delta_level = binned_delta_levels[j]
                        bin_recovered_BRT[mask *
                                          (values < binned_delta_level)] = True

                    if dynamics.set_mode == 'reach':
                        image[BRT] = np.array([252, 227, 152])
                        overlay_border(image, BRT, np.array([249, 188, 6]))
                        image[bin_recovered_BRT] = np.array([155, 241, 249])
                        overlay_border(image, bin_recovered_BRT,
                                       np.array([15, 223, 240]))
                        if ground_truth_grids is not None:
                            overlay_ground_truth(image, i)
                    else:
                        image[bin_recovered_BRT] = np.array([155, 241, 249])
                        image[BRT] = np.array([252, 227, 152])
                        overlay_border(image, BRT, np.array([249, 188, 6]))
                        # overlay recovered border over learned BRT
                        overlay_border(image, bin_recovered_BRT,
                                       np.array([15, 223, 240]))
                        if ground_truth_grids is not None:
                            overlay_ground_truth(image, i)

                    ax.imshow(image.transpose(1, 0, 2),
                              origin='lower', extent=(-1., 1., -1., 1.))
                    ax.set_xlabel(
                        plot_config['state_labels'][plot_config['x_axis_idx']])
                    ax.set_ylabel(
                        plot_config['state_labels'][plot_config['y_axis_idx']])
                    ax.set_xticks([-1, 1])
                    ax.set_yticks([-1, 1])
                    ax.tick_params(labelsize=6)
                    if i != 0:
                        ax.set_yticks([])
                plt.tight_layout()
                fig.savefig(os.path.join(
                    testing_dir, f'binned_BRTs.png'), dpi=800)

            if data_step == 'plot_cost_function':
                if os.path.exists(os.path.join(self.experiment_dir, 'cost_logs.pickle')):
                    with open(os.path.join(self.experiment_dir, 'cost_logs.pickle'), 'rb') as f:
                        logs = pickle.load(f)

                else:
                    with open(os.path.join(self.experiment_dir, 'basic_logs.pickle'), 'rb') as f:
                        logs = pickle.load(f)

                    S = logs['S']
                    delta_level = logs['delta_level']
                    results = scenario_optimization(
                        model=model, dynamics=dynamics,
                        tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                        set_type=set_type, control_type=control_type,
                        scenario_batch_size=min(S, 100000), sample_batch_size=10*min(S, 10000),
                        sample_generator=SliceSampleGenerator(
                            dynamics=dynamics, slices=[None]*dynamics.state_dim),
                        sample_validator=ValueThresholdValidator(v_min=float(
                            '-inf'), v_max=0.0) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=0.0, v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                            'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                        max_scenarios=S, max_samples=1000*min(S, 10000))
                    if results['maxed_scenarios']:
                        logs['learned_costs'] = results['costs']
                    else:
                        logs['learned_costs'] = None

                    results = scenario_optimization(
                        model=model, dynamics=dynamics,
                        tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                        set_type=set_type, control_type=control_type,
                        scenario_batch_size=min(S, 100000), sample_batch_size=10*min(S, 10000),
                        sample_generator=SliceSampleGenerator(
                            dynamics=dynamics, slices=[None]*dynamics.state_dim),
                        sample_validator=ValueThresholdValidator(v_min=float(
                            '-inf'), v_max=delta_level) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=delta_level, v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                            'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                        max_scenarios=S, max_samples=1000*min(S, 10000))
                    if results['maxed_scenarios']:
                        logs['recovered_costs'] = results['costs']
                    else:
                        logs['recovered_costs'] = None

                if logs['learned_costs'] is not None and logs['recovered_costs'] is not None:
                    plt.title(f'Trajectory Costs')
                    plt.ylabel('Frequency')
                    plt.xlabel('Cost')
                    plt.hist(logs['learned_costs'], color=(
                        247/255, 187/255, 8/255), alpha=0.5)
                    plt.hist(logs['recovered_costs'], color=(
                        14/255, 222/255, 241/255), alpha=0.5)
                    plt.axvline(x=0, linestyle='--', color='black')
                    plt.savefig(os.path.join(
                        testing_dir, f'cost_function.png'), dpi=800)

                with open(os.path.join(testing_dir, 'cost_logs.pickle'), 'wb') as f:
                    pickle.dump(logs, f)

        if was_training:
            self.model.train()
            self.model.requires_grad_(True)

    def plot_recovery_fig(self, dataset, dynamics, model, delta_level):
        # 1. for ground truth slices (if available), record (higher-res) grid of learned values
        # plot (with ground truth) learned BRTs, recovered BRTs
        z_res = 5
        plot_config = dataset.dynamics.plot_config()
        if os.path.exists(os.path.join(self.experiment_dir, 'ground_truth.mat')):
            ground_truth = spio.loadmat(os.path.join(
                self.experiment_dir, 'ground_truth.mat'))
            if 'gmat' in ground_truth:
                ground_truth_xs = ground_truth['gmat'][..., 0][:, 0, 0]
                ground_truth_ys = ground_truth['gmat'][..., 1][0, :, 0]
                ground_truth_zs = ground_truth['gmat'][..., 2][0, 0, :]
                ground_truth_values = ground_truth['data']
                ground_truth_ts = np.linspace(
                    0, 1, ground_truth_values.shape[3])

            elif 'g' in ground_truth:
                ground_truth_xs = ground_truth['g']['vs'][0, 0][0][0][:, 0]
                ground_truth_ys = ground_truth['g']['vs'][0, 0][1][0][:, 0]
                ground_truth_zs = ground_truth['g']['vs'][0, 0][2][0][:, 0]
                ground_truth_ts = ground_truth['tau'][0]
                ground_truth_values = ground_truth['data']

            # idxs to plot
            x_idxs = np.linspace(0, len(ground_truth_xs)-1,
                                 len(ground_truth_xs)).astype(dtype=int)
            y_idxs = np.linspace(0, len(ground_truth_ys)-1,
                                 len(ground_truth_ys)).astype(dtype=int)
            z_idxs = np.linspace(0, len(ground_truth_zs) -
                                 1, z_res).astype(dtype=int)
            t_idxs = np.array([len(ground_truth_ts)-1]).astype(dtype=int)

            # indexed ground truth to plot
            ground_truth_xs = ground_truth_xs[x_idxs]
            ground_truth_ys = ground_truth_ys[y_idxs]
            ground_truth_zs = ground_truth_zs[z_idxs]
            ground_truth_ts = ground_truth_ts[t_idxs]
            ground_truth_values = ground_truth_values[
                x_idxs[:, None, None, None],
                y_idxs[None, :, None, None],
                z_idxs[None, None, :, None],
                t_idxs[None, None, None, :]
            ]
            ground_truth_grids = ground_truth_values

            xs = ground_truth_xs
            ys = ground_truth_ys
            zs = ground_truth_zs
        else:
            ground_truth_grids = None
            resolution = 512
            xs = np.linspace(*dynamics.state_test_range()
                             [plot_config['x_axis_idx']], resolution)
            ys = np.linspace(*dynamics.state_test_range()
                             [plot_config['y_axis_idx']], resolution)
            zs = np.linspace(*dynamics.state_test_range()
                             [plot_config['z_axis_idx']], z_res)

        xys = torch.cartesian_prod(torch.tensor(xs), torch.tensor(ys))
        value_grids = np.zeros((len(zs), len(xs), len(ys)))
        for i in range(len(zs)):
            coords = torch.zeros(xys.shape[0], dataset.dynamics.state_dim + 1)
            coords[:, 0] = dataset.tMax
            coords[:, 1:] = torch.tensor(plot_config['state_slices'])
            coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
            coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
            if dataset.dynamics.state_dim > 2:
                coords[:, 1 + plot_config['z_axis_idx']] = zs[i]

            model_results = model(
                {'coords': dataset.dynamics.coord_to_input(coords.cuda())})
            values = dataset.dynamics.io_to_value(model_results['model_in'].detach(
            ), model_results['model_out'].detach().squeeze(dim=-1)).detach().cpu()
            value_grids[i] = values.reshape(len(xs), len(ys))

        fig = plt.figure()
        fig.suptitle(plot_config['state_slices'], fontsize=8)
        x_min, x_max = dataset.dynamics.state_test_range()[
            plot_config['x_axis_idx']]
        y_min, y_max = dataset.dynamics.state_test_range()[
            plot_config['y_axis_idx']]

        for i in range(len(zs)):
            values = value_grids[i]

            # learned BRT and recovered BRT
            ax = fig.add_subplot(1, len(zs), (i+1))
            ax.set_title('%s = %0.2f' % (
                plot_config['state_labels'][plot_config['z_axis_idx']], zs[i]), fontsize=8)

            image = np.full((*values.shape, 3), 255, dtype=int)
            BRT = values < 0
            recovered_BRT = values < delta_level

            if dynamics.set_mode == 'reach':
                image[BRT] = np.array([252, 227, 152])
                self.overlay_border(image, BRT, np.array([249, 188, 6]))
                image[recovered_BRT] = np.array([155, 241, 249])
                self.overlay_border(image, recovered_BRT,
                                    np.array([15, 223, 240]))
                if ground_truth_grids is not None:
                    self.overlay_ground_truth(image, i, ground_truth_grids)
            else:
                image[recovered_BRT] = np.array([155, 241, 249])
                image[BRT] = np.array([252, 227, 152])
                self.overlay_border(image, BRT, np.array([249, 188, 6]))
                # overlay recovered border over learned BRT
                self.overlay_border(image, recovered_BRT,
                                    np.array([15, 223, 240]))
                if ground_truth_grids is not None:
                    self.overlay_ground_truth(image, i, ground_truth_grids)

            ax.imshow(image.transpose(1, 0, 2), origin='lower',
                      extent=(x_min, x_max, y_min, y_max))

            ax.set_xlabel(plot_config['state_labels']
                          [plot_config['x_axis_idx']])
            ax.set_ylabel(plot_config['state_labels']
                          [plot_config['y_axis_idx']])
            ax.set_xticks([x_min, x_max])
            ax.set_yticks([y_min, y_max])
            ax.tick_params(labelsize=6)
            if i != 0:
                ax.set_yticks([])
        return fig, value_grids

    def overlay_ground_truth(self, image, z_idx, ground_truth_grids):
        thickness = max(0, image.shape[0] // 120 - 1)
        ground_truth_grid = ground_truth_grids[:, :, z_idx, 0]
        ground_truth_brts = ground_truth_grid < 0
        for x in range(ground_truth_brts.shape[0]):
            for y in range(ground_truth_brts.shape[1]):
                if not ground_truth_brts[x, y]:
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
                    if neighbor[0] >= 0 and neighbor[1] >= 0 and neighbor[0] < ground_truth_brts.shape[0] and neighbor[1] < ground_truth_brts.shape[1]:
                        if not ground_truth_brts[neighbor]:
                            image[x-thickness:x+1, y-thickness:y +
                                  1+thickness] = np.array([50, 50, 50])
                            break

    def overlay_border(self, image, set, color):
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


class DeepReach(Experiment):
    def init_special(self):
        self.timing = False
        pass

class DeepReachHopf(Experiment):
    def init_special(self, N=2, timing=False):
        self.N = N
        self.timing = timing
        if N == 2:
            self.validate = self.validate2D
        elif N > 2:
            self.validate = self.validateND
            if self.dataset.lambda_var:
                self.validate = self.validateNDlambda
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
                model_results = self.model({'coords': self.dataset.dynamics.coord_to_input(coords.cuda())})
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
                FIp = (values_grid_sub0_ixs.size()[0] - n_intersect) / values_grid_sub0_ixs.size()[0] # <- wrt true set, wrt grid: (self.dataset.n_grid_t_pts * self.dataset.n_grid_pts)
            else:
                FIp = 1.
            FEp = (self.dataset.values_DP_grid_sub0_ixs.size()[0] - n_intersect) / self.dataset.values_DP_grid_sub0_ixs.size()[0] # <- wrt true set, wrt grid: (self.dataset.n_grid_t_pts * self.dataset.n_grid_pts)
            JIp = n_intersect / n_overlap
            ## FIXME: still wondering if there is a bug in FIp and JIp... they look slightly off sometimes 
        
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
