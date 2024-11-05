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
from scipy.stats import beta as beta__dist
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
            loss_fn, clip_grad, use_lbfgs, adjust_relative_grads, 
            val_x_resolution, val_y_resolution, val_z_resolution, val_time_resolution,
            use_CSL, CSL_lr, CSL_dt, epochs_til_CSL, num_CSL_samples, CSL_loss_frac_cutoff, max_CSL_epochs, CSL_loss_weight, CSL_batch_size,
            dual_lr=False, lr_decay_w=1., lr_hopf=2e-5, lr_hopf_decay_w=1., smoothing_factor=0.8, 
            hopf_loss='none', hopf_loss_decay_early = True, hopf_loss_decay=True, hopf_loss_decay_w=0.9998,
            reset_loss_w=False, reset_loss_period=0, 
            diff_con_loss_incr=False, hopf_loss_decay_type = 'exponential',
            nonlin_scale=False, nl_scale_epoch_step=10000, nl_scale_epoch_post=10000, 
            record_temporal_loss = False, 
            deposit_blocking = True, deposit_blocking_period = 1000 # seg faults if nonblocking rn...
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
        loss_weights = {'dirichlet': 1., 'hopf': 1., 'diff_constraint_hom': 1.}
        if diff_con_loss_incr:
            loss_weights['diff_constraint_hom'] = 0.
        if hopf_loss_decay_type == 'negative_exponential': 
            loss_weights['hopf'] = 1 - (hopf_loss_decay_w ** (epochs - 1 - total_pretrain_iters))
        if hopf_loss == 'lin_val_grad_diff':
            loss_weights['hopf_grad'] = loss_weights['hopf']
        og_loss_weights = loss_weights.copy()
        
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

                    ## Compute Loss
                    if self.timing: start_time_2 = time.time()

                    ## Standard BRT
                    if self.dataset.dynamics.loss_type == 'brt_hjivi':
                        losses = loss_fn(states, values, dvs[..., 0], dvs[..., 1:], boundary_values, dirichlet_masks, model_results['model_out'])
                    
                    ## Standard BRAT
                    elif self.dataset.dynamics.loss_type == 'brat_hjivi':
                        losses = loss_fn(states, values, dvs[..., 0], dvs[..., 1:], boundary_values, reach_values, avoid_values, dirichlet_masks, model_results['model_out'])
                    
                    ## Linear Supervision BRT (Hopf-based)
                    elif hopf_loss != 'none':
                        hopf_values = gt['hopf_values']

                        if hopf_loss == 'lin_val_grad_diff':
                            hopf_grads = gt['hopf_grads']

                        if self.dataset.load_hopf_model:
                            loaded_model_results = self.dataset.loaded_model({'coords': model_input['model_coords']})
                            hopf_values = self.dataset.dynamics.io_to_value(loaded_model_results['model_in'].detach(), loaded_model_results['model_out'].squeeze(dim=-1))
                            if self.dataset.solve_grad:
                                hopf_grads = self.dataset.dynamics.io_to_dv(loaded_model_results['model_in'], loaded_model_results['model_out'].squeeze(dim=-1))[..., 1:]
                        
                        # the following allows separate coordinates for the hopf loss (to allow unrestricted sampling for PDE loss)
                        if not(self.dataset.use_bank) or self.dataset.hopf_pretrain_counter == 0:
                            
                            learned_hopf_values = values
                            if hopf_loss == 'lin_val_grad_diff':
                                learned_hopf_grads = dvs[..., 1:]
                                
                        else:
                            model_results_hopf = self.model({'coords': gt['model_coords_hopf']})
                            learned_hopf_values = self.dataset.dynamics.io_to_value(model_results_hopf['model_in'].detach(), model_results_hopf['model_out'].squeeze(dim=-1))   
                            
                            if hopf_loss == 'lin_val_grad_diff':
                                learned_hopf_grads = self.dataset.dynamics.io_to_dv(model_results_hopf['model_in'], model_results_hopf['model_out'].squeeze(dim=-1))[..., 1:]   
                        
                        if hopf_loss == 'lin_val_grad_diff':
                            losses = loss_fn(states, values, dvs[..., 0], dvs[..., 1:], boundary_values, dirichlet_masks, model_results['model_out'], hopf_values, learned_hopf_values, hopf_grads, learned_hopf_grads, epoch, state_times)
                        else:
                            losses = loss_fn(states, values, dvs[..., 0], dvs[..., 1:], boundary_values, dirichlet_masks, model_results['model_out'], hopf_values, learned_hopf_values, epoch, state_times)

                    else:
                        raise NotImplementedError
                    
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
                    if dual_lr and not(self.dataset.hopf_pretrain) and self.dataset.hopf_pretrained:
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

                    ## Record Data Summary
                    if not total_steps % steps_til_summary:
                        iter_time = time.time() - start_time

                        if self.dataset.record_gt_metrics:   
                            JIp, FIp, FEp, Vmse, DVXmse = self.compute_gt_metrics()
                            JIp_s = smoothing_factor * JIp + (1 - smoothing_factor) * JIp_s
                            JIp_s_max = max(JIp_s, JIp_s_max)
                        
                            tqdm.write("Epoch %d, Total loss %0.6f, MSE %3.5e, IOU %0.6f, iter time %0.6f, pde loss %0.2f, hopf loss %0.2f" % (epoch, train_loss, Vmse, JIp, iter_time, losses["diff_constraint_hom"], losses["hopf"]))
                        else:
                            tqdm.write("Epoch %d, Total loss %0.6f, iter time %0.6f" % (epoch, train_loss, iter_time))
                        
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
                        epoch=epoch+1, save_path=os.path.join(checkpoints_dir, 'BRS_validation_plot_epoch_%04d.png' % (epoch+1)),
                        x_resolution = val_x_resolution, y_resolution = val_y_resolution, z_resolution=val_z_resolution, time_resolution=val_time_resolution)
                if self.timing: print("Checkpointing took:", time.time() - start_time_2)
        torch.save(checkpoint,
            os.path.join(checkpoints_dir, 'model_final.pth'))
        if was_eval:
            self.model.eval()
            self.model.requires_grad_(False)
    def test(self, current_time, last_checkpoint, checkpoint_dt, dt, num_scenarios, num_violations, set_type, control_type, data_step, checkpoint_toload=None):
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
                print(results['batch_state_trajs'].shape)
                print(outlier_states.shape)
                print(outlier_states[:200, ...])

                (vs, bins, patches) = plt.hist(
                    positive_values_with_negative_cost, bins=200)
                plt.title("Learned values for actually unsafe states marked as safe\n%s" %
                          dynamics.deepReach_model)
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
                for i in range(M):
                    print('algorithm iter', str(i))
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
                ).item()
                logs['recovered_volume'] = target_fraction(
                    model=model, dynamics=dynamics, t=dataset.tMax,
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    target_validator=ValueThresholdValidator(v_min=float(
                        '-inf'), v_max=delta_level) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=delta_level, v_max=float('inf')),
                    num_samples=S,
                    batch_size=min(S, 1000000)
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

    # def post_training_CSL(self,loss_fn, clip_grad, use_lbfgs,
    #     val_x_resolution, val_y_resolution, val_z_resolution, val_time_resolution,
    #     use_CSL, CSL_dt, epochs_til_CSL, num_CSL_samples, CSL_loss_frac_cutoff, max_CSL_epochs, CSL_loss_weight, CSL_batch_size):

    #             last_CSL_epoch = epoch

    #             # generate CSL datasets
    #             self.model.eval()

    #             CSL_dataset = scenario_optimization(
    #                 model=self.model, dynamics=self.dataset.dynamics,
    #                 tMin=self.dataset.tMin, tMax=CSL_tMax, dt=CSL_dt,
    #                 set_type="BRT", control_type="value",  # TODO: implement option for BRS too
    #                 scenario_batch_size=min(num_CSL_samples, 100000), sample_batch_size=min(10*num_CSL_samples, 1000000),
    #                 sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[
    #                                                         None]*self.dataset.dynamics.state_dim),
    #                 sample_validator=ValueThresholdValidator(
    #                     v_min=float('-inf'), v_max=float('inf')),
    #                 violation_validator=ValueThresholdValidator(
    #                     v_min=0.0, v_max=0.0),
    #                 max_scenarios=num_CSL_samples, tStart_generator=lambda n: torch.zeros(
    #                     n).uniform_(self.dataset.tMin, CSL_tMax)
    #             )
    #             CSL_coords = torch.cat(
    #                 (CSL_dataset['times'].unsqueeze(-1), CSL_dataset['states']), dim=-1)
    #             CSL_costs = CSL_dataset['costs']

    #             num_CSL_val_samples = int(0.1*num_CSL_samples)
    #             CSL_val_dataset = scenario_optimization(
    #                 model=self.model, dynamics=self.dataset.dynamics,
    #                 tMin=self.dataset.tMin, tMax=CSL_tMax, dt=CSL_dt,
    #                 set_type="BRT", control_type="value",  # TODO: implement option for BRS too
    #                 scenario_batch_size=min(num_CSL_val_samples, 100000), sample_batch_size=min(10*num_CSL_val_samples, 1000000),
    #                 sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[
    #                                                         None]*self.dataset.dynamics.state_dim),
    #                 sample_validator=ValueThresholdValidator(
    #                     v_min=float('-inf'), v_max=float('inf')),
    #                 violation_validator=ValueThresholdValidator(
    #                     v_min=0.0, v_max=0.0),
    #                 max_scenarios=num_CSL_val_samples, tStart_generator=lambda n: torch.zeros(
    #                     n).uniform_(self.dataset.tMin, CSL_tMax)
    #             )
    #             CSL_val_coords = torch.cat(
    #                 (CSL_val_dataset['times'].unsqueeze(-1), CSL_val_dataset['states']), dim=-1)
    #             CSL_val_costs = CSL_val_dataset['costs']

    #             CSL_val_tMax_dataset = scenario_optimization(
    #                 model=self.model, dynamics=self.dataset.dynamics,
    #                 tMin=self.dataset.tMin, tMax=self.dataset.tMax, dt=CSL_dt,
    #                 set_type="BRT", control_type="value",  # TODO: implement option for BRS too
    #                 scenario_batch_size=min(num_CSL_val_samples, 100000), sample_batch_size=min(10*num_CSL_val_samples, 1000000),
    #                 sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[
    #                                                         None]*self.dataset.dynamics.state_dim),
    #                 sample_validator=ValueThresholdValidator(
    #                     v_min=float('-inf'), v_max=float('inf')),
    #                 violation_validator=ValueThresholdValidator(
    #                     v_min=0.0, v_max=0.0),
    #                 # no tStart_generator, since I want all tMax times
    #                 max_scenarios=num_CSL_val_samples
    #             )
    #             CSL_val_tMax_coords = torch.cat(
    #                 (CSL_val_tMax_dataset['times'].unsqueeze(-1), CSL_val_tMax_dataset['states']), dim=-1)
    #             CSL_val_tMax_costs = CSL_val_tMax_dataset['costs']

    #             self.model.train()

    #             # CSL optimizer
    #             CSL_optim = torch.optim.Adam(
    #                 lr=csl_lr, params=self.model.parameters())

    #             # initial CSL val loss
    #             CSL_val_results = self.model(
    #                 {'coords': self.dataset.dynamics.coord_to_input(CSL_val_coords.cuda())})
    #             CSL_val_preds = self.dataset.dynamics.io_to_value(
    #                 CSL_val_results['model_in'], CSL_val_results['model_out'].squeeze(dim=-1))
    #             CSL_val_errors = CSL_val_preds - CSL_val_costs.cuda()
    #             CSL_val_loss = torch.mean(torch.pow(CSL_val_errors, 2))
    #             CSL_initial_val_loss = CSL_val_loss
    #             wandb.log({
    #                 "step": epoch,
    #                 "CSL_val_loss": CSL_val_loss.item()
    #             })

    #             # initial self-supervised learning (SSL) val loss
    #             # right now, just took code from dataio.py and the SSL training loop above; TODO: refactor all this for cleaner modular code
    #             CSL_val_states = CSL_val_coords[..., 1:].cuda()
    #             CSL_val_dvs = self.dataset.dynamics.io_to_dv(
    #                 CSL_val_results['model_in'], CSL_val_results['model_out'].squeeze(dim=-1))
    #             CSL_val_boundary_values = self.dataset.dynamics.boundary_fn(
    #                 CSL_val_states)
    #             if self.dataset.dynamics.loss_type == 'brat_hjivi':
    #                 CSL_val_reach_values = self.dataset.dynamics.reach_fn(
    #                     CSL_val_states)
    #                 CSL_val_avoid_values = self.dataset.dynamics.avoid_fn(
    #                     CSL_val_states)
    #             # assumes time unit in dataset (model) is same as real time units
    #             CSL_val_dirichlet_masks = CSL_val_coords[:, 0].cuda(
    #             ) == self.dataset.tMin
    #             if self.dataset.dynamics.loss_type == 'brt_hjivi':
    #                 SSL_val_losses = loss_fn(
    #                     CSL_val_states, CSL_val_preds, CSL_val_dvs[..., 0], CSL_val_dvs[..., 1:], CSL_val_boundary_values, CSL_val_dirichlet_masks)
    #             elif self.dataset.dynamics.loss_type == 'brat_hjivi':
    #                 SSL_val_losses = loss_fn(CSL_val_states, CSL_val_preds, CSL_val_dvs[..., 0], CSL_val_dvs[..., 1:],
    #                                             CSL_val_boundary_values, CSL_val_reach_values, CSL_val_avoid_values, CSL_val_dirichlet_masks)
    #             else:
    #                 NotImplementedError
    #             # I assume there is no dirichlet (boundary) loss here, because I do not ever explicitly generate source samples at tMin (i.e. torch.all(CSL_val_dirichlet_masks == False))
    #             SSL_val_loss = SSL_val_losses['diff_constraint_hom'].mean()
    #             wandb.log({
    #                 "step": epoch,
    #                 "SSL_val_loss": SSL_val_loss.item()
    #             })

    #             # CSL training loop
    #             for CSL_epoch in tqdm(range(max_CSL_epochs)):
    #                 CSL_idxs = torch.randperm(num_CSL_samples)
    #                 for CSL_batch in range(math.ceil(num_CSL_samples/CSL_batch_size)):
    #                     CSL_batch_idxs = CSL_idxs[CSL_batch *
    #                                                 CSL_batch_size:(CSL_batch+1)*CSL_batch_size]
    #                     CSL_batch_coords = CSL_coords[CSL_batch_idxs]

    #                     CSL_batch_results = self.model(
    #                         {'coords': self.dataset.dynamics.coord_to_input(CSL_batch_coords.cuda())})
    #                     CSL_batch_preds = self.dataset.dynamics.io_to_value(
    #                         CSL_batch_results['model_in'], CSL_batch_results['model_out'].squeeze(dim=-1))
    #                     CSL_batch_costs = CSL_costs[CSL_batch_idxs].cuda()
    #                     CSL_batch_errors = CSL_batch_preds - CSL_batch_costs
    #                     CSL_batch_loss = CSL_loss_weight * \
    #                         torch.mean(torch.pow(CSL_batch_errors, 2))

    #                     CSL_batch_states = CSL_batch_coords[..., 1:].cuda()
    #                     CSL_batch_dvs = self.dataset.dynamics.io_to_dv(
    #                         CSL_batch_results['model_in'], CSL_batch_results['model_out'].squeeze(dim=-1))
    #                     CSL_batch_boundary_values = self.dataset.dynamics.boundary_fn(
    #                         CSL_batch_states)
    #                     if self.dataset.dynamics.loss_type == 'brat_hjivi':
    #                         CSL_batch_reach_values = self.dataset.dynamics.reach_fn(
    #                             CSL_batch_states)
    #                         CSL_batch_avoid_values = self.dataset.dynamics.avoid_fn(
    #                             CSL_batch_states)
    #                     # assumes time unit in dataset (model) is same as real time units
    #                     CSL_batch_dirichlet_masks = CSL_batch_coords[:, 0].cuda(
    #                     ) == self.dataset.tMin
    #                     if self.dataset.dynamics.loss_type == 'brt_hjivi':
    #                         SSL_batch_losses = loss_fn(
    #                             CSL_batch_states, CSL_batch_preds, CSL_batch_dvs[..., 0], CSL_batch_dvs[..., 1:], CSL_batch_boundary_values, CSL_batch_dirichlet_masks)
    #                     elif self.dataset.dynamics.loss_type == 'brat_hjivi':
    #                         SSL_batch_losses = loss_fn(
    #                             CSL_batch_states, CSL_batch_preds, CSL_batch_dvs[..., 0], CSL_batch_dvs[..., 1:], CSL_batch_boundary_values, CSL_batch_reach_values, CSL_batch_avoid_values, CSL_batch_dirichlet_masks)
    #                     else:
    #                         NotImplementedError
    #                     # I assume there is no dirichlet (boundary) loss here, because I do not ever explicitly generate source samples at tMin (i.e. torch.all(CSL_batch_dirichlet_masks == False))
    #                     SSL_batch_loss = SSL_batch_losses['diff_constraint_hom'].mean(
    #                     )

    #                     CSL_optim.zero_grad()
    #                     SSL_batch_loss.backward(retain_graph=True)
    #                     # no adjust_relative_grads, because I assume even with adjustment, the diff_constraint_hom remains unaffected and the only other loss (dirichlet) is zero
    #                     if (not use_lbfgs) and clip_grad:
    #                         if isinstance(clip_grad, bool):
    #                             torch.nn.utils.clip_grad_norm_(
    #                                 self.model.parameters(), max_norm=1.)
    #                         else:
    #                             torch.nn.utils.clip_grad_norm_(
    #                                 self.model.parameters(), max_norm=clip_grad)
    #                     CSL_batch_loss.backward()
    #                     CSL_optim.step()

    #                 # evaluate on CSL_val_dataset
    #                 CSL_val_results = self.model(
    #                     {'coords': self.dataset.dynamics.coord_to_input(CSL_val_coords.cuda())})
    #                 CSL_val_preds = self.dataset.dynamics.io_to_value(
    #                     CSL_val_results['model_in'], CSL_val_results['model_out'].squeeze(dim=-1))
    #                 CSL_val_errors = CSL_val_preds - CSL_val_costs.cuda()
    #                 CSL_val_loss = torch.mean(torch.pow(CSL_val_errors, 2))

    #                 CSL_val_states = CSL_val_coords[..., 1:].cuda()
    #                 CSL_val_dvs = self.dataset.dynamics.io_to_dv(
    #                     CSL_val_results['model_in'], CSL_val_results['model_out'].squeeze(dim=-1))
    #                 CSL_val_boundary_values = self.dataset.dynamics.boundary_fn(
    #                     CSL_val_states)
    #                 if self.dataset.dynamics.loss_type == 'brat_hjivi':
    #                     CSL_val_reach_values = self.dataset.dynamics.reach_fn(
    #                         CSL_val_states)
    #                     CSL_val_avoid_values = self.dataset.dynamics.avoid_fn(
    #                         CSL_val_states)
    #                 # assumes time unit in dataset (model) is same as real time units
    #                 CSL_val_dirichlet_masks = CSL_val_coords[:, 0].cuda(
    #                 ) == self.dataset.tMin
    #                 if self.dataset.dynamics.loss_type == 'brt_hjivi':
    #                     SSL_val_losses = loss_fn(
    #                         CSL_val_states, CSL_val_preds, CSL_val_dvs[..., 0], CSL_val_dvs[..., 1:], CSL_val_boundary_values, CSL_val_dirichlet_masks)
    #                 elif self.dataset.dynamics.loss_type == 'brat_hjivi':
    #                     SSL_val_losses = loss_fn(
    #                         CSL_val_states, CSL_val_preds, CSL_val_dvs[..., 0], CSL_val_dvs[..., 1:], CSL_val_boundary_values, CSL_val_reach_values, CSL_val_avoid_values, CSL_val_dirichlet_masks)
    #                 else:
    #                     raise NotImplementedError
    #                 # I assume there is no dirichlet (boundary) loss here, because I do not ever explicitly generate source samples at tMin (i.e. torch.all(CSL_val_dirichlet_masks == False))
    #                 SSL_val_loss = SSL_val_losses['diff_constraint_hom'].mean(
    #                 )

    #                 CSL_val_tMax_results = self.model(
    #                     {'coords': self.dataset.dynamics.coord_to_input(CSL_val_tMax_coords.cuda())})
    #                 CSL_val_tMax_preds = self.dataset.dynamics.io_to_value(
    #                     CSL_val_tMax_results['model_in'], CSL_val_tMax_results['model_out'].squeeze(dim=-1))
    #                 CSL_val_tMax_errors = CSL_val_tMax_preds - CSL_val_tMax_costs.cuda()
    #                 CSL_val_tMax_loss = torch.mean(
    #                     torch.pow(CSL_val_tMax_errors, 2))

    #                 # log CSL losses, recovered_safe_set_fracs
    #                 if self.dataset.dynamics.set_mode == 'reach':
    #                     CSL_train_batch_theoretically_recoverable_safe_set_frac = torch.sum(
    #                         CSL_batch_costs.cuda() < 0) / len(CSL_batch_preds)
    #                     if len(CSL_batch_preds[CSL_batch_costs.cuda() > 0]) > 0:
    #                         CSL_train_batch_recovered_safe_set_frac = torch.sum(CSL_batch_preds < torch.min(
    #                             CSL_batch_preds[CSL_batch_costs.cuda() > 0])) / len(CSL_batch_preds)
    #                     else:
    #                         CSL_train_batch_recovered_safe_set_frac = torch.FloatTensor(
    #                             1)
    #                     CSL_val_theoretically_recoverable_safe_set_frac = torch.sum(
    #                         CSL_val_costs.cuda() < 0) / len(CSL_val_preds)
    #                     CSL_val_recovered_safe_set_frac = torch.sum(CSL_val_preds < torch.min(
    #                         CSL_val_preds[CSL_val_costs.cuda() > 0])) / len(CSL_val_preds)
    #                     CSL_val_tMax_theoretically_recoverable_safe_set_frac = torch.sum(
    #                         CSL_val_tMax_costs.cuda() < 0) / len(CSL_val_tMax_preds)
    #                     CSL_val_tMax_recovered_safe_set_frac = torch.sum(CSL_val_tMax_preds < torch.min(
    #                         CSL_val_tMax_preds[CSL_val_tMax_costs.cuda() > 0])) / len(CSL_val_tMax_preds)
    #                 elif self.dataset.dynamics.set_mode == 'avoid':
    #                     CSL_train_batch_theoretically_recoverable_safe_set_frac = torch.sum(
    #                         CSL_batch_costs.cuda() > 0) / len(CSL_batch_preds)
    #                     # if len(CSL_batch_preds[CSL_batch_costs.cuda() < 0])>0:
    #                     #     CSL_train_batch_recovered_safe_set_frac = torch.sum(CSL_batch_preds > torch.max(CSL_batch_preds[CSL_batch_costs.cuda() < 0])) / len(CSL_batch_preds)
    #                     # else:
    #                     #     CSL_train_batch_recovered_safe_set_frac=torch.FloatTensor(1)
    #                     CSL_train_batch_recovered_safe_set_frac = torch.sum(CSL_batch_preds > torch.max(
    #                         CSL_batch_preds[CSL_batch_costs.cuda() < 0])) / len(CSL_batch_preds)

    #                     CSL_val_theoretically_recoverable_safe_set_frac = torch.sum(
    #                         CSL_val_costs.cuda() > 0) / len(CSL_val_preds)
    #                     CSL_val_recovered_safe_set_frac = torch.sum(CSL_val_preds > torch.max(
    #                         CSL_val_preds[CSL_val_costs.cuda() < 0])) / len(CSL_val_preds)
    #                     CSL_val_tMax_theoretically_recoverable_safe_set_frac = torch.sum(
    #                         CSL_val_tMax_costs.cuda() > 0) / len(CSL_val_tMax_preds)
    #                     CSL_val_tMax_recovered_safe_set_frac = torch.sum(CSL_val_tMax_preds > torch.max(
    #                         CSL_val_tMax_preds[CSL_val_tMax_costs.cuda() < 0])) / len(CSL_val_tMax_preds)

    #                     wandb.log({
    #                         "step": epoch+(CSL_epoch+1)*int(0.5*epochs_til_CSL/max_CSL_epochs),
    #                         "CSL_train_batch_loss": CSL_batch_loss.item(),
    #                         "SSL_train_batch_loss": SSL_batch_loss.item(),
    #                         "CSL_val_loss": CSL_val_loss.item(),
    #                         "SSL_val_loss": SSL_val_loss.item(),
    #                         "CSL_val_tMax_loss": CSL_val_tMax_loss.item(),
    #                         "CSL_train_batch_theoretically_recoverable_safe_set_frac": CSL_train_batch_theoretically_recoverable_safe_set_frac.item(),
    #                         "CSL_val_theoretically_recoverable_safe_set_frac": CSL_val_theoretically_recoverable_safe_set_frac.item(),
    #                         "CSL_val_tMax_theoretically_recoverable_safe_set_frac": CSL_val_tMax_theoretically_recoverable_safe_set_frac.item(),
    #                         "CSL_train_batch_recovered_safe_set_frac": CSL_train_batch_recovered_safe_set_frac.item(),
    #                         "CSL_val_recovered_safe_set_frac": CSL_val_recovered_safe_set_frac.item(),
    #                         "CSL_val_tMax_recovered_safe_set_frac": CSL_val_tMax_recovered_safe_set_frac.item(),
    #                     })

    #             if not (epoch+1) % epochs_til_checkpoint:
    #                 # Saving the optimizer state is important to produce consistent results
    #                 checkpoint = {
    #                     'epoch': epoch+1,
    #                     'model': self.model.state_dict(),
    #                     'optimizer': optim.state_dict()}
    #                 torch.save(checkpoint,
    #                            os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % (epoch+1)))
    #                 np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % (epoch+1)),
    #                            np.array(train_losses))
    #                 self.validate(
    #                     epoch=epoch+1, save_path=os.path.join(checkpoints_dir, 'BRS_validation_plot_epoch_%04d.png' % (epoch+1)),
    #                     x_resolution=val_x_resolution, y_resolution=val_y_resolution, z_resolution=val_z_resolution, time_resolution=val_time_resolution)

    #     torch.save(checkpoint, os.path.join(
    #         checkpoints_dir, 'model_CSL.pth'))
    
class DeepReach(Experiment):
    def init_special(self):
        pass

class DeepReachHopf(Experiment):
    def init_special(self, N=2, timing=False):
        self.N = N
        self.timing = timing
        if N == 2:
            self.validate = self.validate2D
        # elif N > 2:
        #     self.validate = self.validateND
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

        fig_set = plt.figure(figsize=(5*len(times), 3*5*1))
        fig_val = plt.figure(figsize=(5*len(times), 3*5*1))

        for i in range(3*len(times)):
            
            ax_set = fig_set.add_subplot(3, len(times), 1+i)
            ax_val = fig_val.add_subplot(3, len(times), 1+i, projection='3d')
            ax_set.set_title('t = %0.2f' % (times[i % len(times)]))
            ax_val.set_title('t = %0.2f' % (times[i % len(times)]))

            ## Define Grid Slice to Plot

            coords = torch.zeros(x_resolution*y_resolution, self.dataset.dynamics.state_dim + 1)
            coords[:, 0] = times[i % len(times)]
            coords[:, 1:] = torch.tensor(plot_config['state_slices']) # initialized to zero (nothing else to set!)

            if i < len(times): # xN - xi plane
                ax_set.set_xlabel("xN"); ax_set.set_ylabel("xi")
                ax_val.set_xlabel("xN"); ax_val.set_ylabel("xi")
                coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
                coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]

            elif i < 2*len(times): # xi - xj plane
                ax_set.set_xlabel("xi"); ax_set.set_ylabel("xj")
                ax_val.set_xlabel("xi"); ax_val.set_ylabel("xj")
                coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 0]
                coords[:, 1 + plot_config['z_axis_idx']] = xys[:, 1]

            else: # xN - (xi = xj) plane
                ax_set.set_xlabel("xN"); ax_set.set_ylabel("xi=xj")
                ax_val.set_xlabel("xN"); ax_val.set_ylabel("xi=xj")
                coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
                coords[:, 2:] = (xys[:, 1] * torch.ones(self.N-1, xys.size()[0])).t()

            with torch.no_grad():
                model_results = self.model({'coords': self.dataset.dynamics.coord_to_input(coords.cuda())})
                values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach())
            
            learned_value = values.detach().cpu().numpy().reshape(x_resolution, y_resolution)

            ## Plot Zero-level Set of Learned Value

            s = ax_set.imshow(1*(learned_value.T <= 0), cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
            divider = make_axes_locatable(ax_set)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig_set.colorbar(s, cax=cax)

            ## Plot Ground-Truth Zero-Level Contour
            if self.dataset.record_gt_metrics:
                n_grid_plane_pts = int(self.dataset.n_grid_pts/3)
                n_grid_len = int(n_grid_plane_pts ** 0.5)
                pix_start = (i // len(times)) * n_grid_plane_pts
                tix_start = (i % len(times)) * self.dataset.n_grid_pts
                ix = pix_start + tix_start
                Vg = self.dataset.values_DP_grid[ix:ix+n_grid_plane_pts].reshape(n_grid_len, n_grid_len)
                ax_set.contour(self.dataset.X1g, self.dataset.X2g, Vg.cpu(), [0.])

                ## Plot the Linear Ground-Truth (ideal warm-start) Zero-Level Contour

                Vg = self.dataset.values_DP_linear_grid[ix:ix+n_grid_plane_pts].reshape(n_grid_len, n_grid_len)
                ax_set.contour(self.dataset.X1g, self.dataset.X2g, Vg.cpu(), [0.], colors='gold', linestyles='dashed')

            ## Plot 3D Value Fn

            if plot_value:
                if learned_value.min() > 0:
                    RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', [(1,1,1), (0.5,0.5,1), (0,0,1), (0,0,1)])
                elif learned_value.max() < 0:
                    RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', [(1,0,0), (1,0,0), (1,0.5,0.5), (1,1,1)])
                else:
                    n_bins_high = int(256 * (learned_value.max()/(learned_value.max() - learned_value.min())) // 1)
                    RdWh = matplotlib.colors.LinearSegmentedColormap.from_list('RdWh', [(1,0,0), (1,0,0), (1,0.5,0.5), (1,1,1)])
                    WhBl = matplotlib.colors.LinearSegmentedColormap.from_list('WhBl', [(1,1,1), (0.5,0.5,1), (0,0,1), (0,0,1)])
                    colors = np.vstack((RdWh(np.linspace(0., 1, 256-n_bins_high)), WhBl(np.linspace(0., 1, n_bins_high))))
                    RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', colors)
                
                ax_val.view_init(elev=15, azim=-60)
                surf = ax_val.plot_surface(Xg, Yg, learned_value, cmap=RdWhBl_vscaled) #cmap='bwr_r')
                fig_val.colorbar(surf, ax=ax_val, fraction=0.02, pad=0.1)
                ax_val.set_zlim(-max(ax_val.get_zlim()[1]/5, 0.5))
                ax_val.contour(Xg, Yg, learned_value, zdir='z', offset=ax_val.get_zlim()[0], cmap=RdWh, levels=[0.]) #cmap='bwr_r')

        fig_set.savefig(save_path)
        if plot_value: fig_val.savefig(save_path.split('_epoch')[0] + '_Vfn' + save_path.split('_epoch')[1])
        if self.use_wandb:
            log_dict_plot = {'step': epoch,
                        'val_plot': wandb.Image(fig_set),} # (silly) legacy name
            if plot_value: log_dict_plot['val_fn_plot'] = wandb.Image(fig_val)
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

        # if self.N == 2: do whats here, else: use grid only on slices / actually jk, just fill .dataset loads with proper grids (full for 2D, slices for ND)
        model_results_grid = self.model({'coords': self.dataset.model_coords_grid_allt})
        DVXmse = 0

        ## Compute Value Gradient MSE on Grid
        if self.dataset.solve_grad:
            DVX = self.dataset.dynamics.io_to_dv(model_results_grid['model_in'], model_results_grid['model_out'].squeeze(dim=-1))[..., 1:].detach()
            DVXmse = (self.dataset.value_grads_DP_grid - DVX).square().mean()

        with torch.no_grad():
            values_grid = self.dataset.dynamics.io_to_value(model_results_grid['model_in'].detach(), model_results_grid['model_out'].squeeze(dim=-1))
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
                