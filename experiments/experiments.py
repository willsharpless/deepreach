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
            self.model.load_state_dict(torch.load(model_path))
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
                        
                            tqdm.write("Epoch %d, Total loss %0.6f, MSE %3.5e, IOU %0.6f, iter time %0.6f" % (epoch, train_loss, Vmse, JIp, iter_time))
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
                    torch.save(checkpoint,
                        os.path.join(checkpoints_dir, 'model_final.pth'))
                    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % (epoch+1)),
                        np.array(train_losses))
                    self.validate(
                        # epoch=epoch+1, save_path=os.path.join(checkpoints_dir, 'BRS_validation_plot_epoch_%04d.png' % (epoch+1)),
                        epoch=epoch+1, save_path=os.path.join(checkpoints_dir, 'BRS_validation_plot.png'), # overwriting to save data
                        x_resolution = val_x_resolution, y_resolution = val_y_resolution, z_resolution=val_z_resolution, time_resolution=val_time_resolution)
                if self.timing: print("Checkpointing took:", time.time() - start_time_2)

        if was_eval:
            self.model.eval()
            self.model.requires_grad_(False)

    def test(self, current_time, last_checkpoint, checkpoint_dt, dt, num_scenarios, num_violations, set_type, control_type, data_step, checkpoint_toload=None):
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
            print('running cross-checkpoint testing')

            for i in tqdm(range(sidelen), desc='Checkpoint'):
                self._load_checkpoint(epoch=checkpoints[i])
                raise NotImplementedError

        else:
            print('running specific-checkpoint testing')
            self._load_checkpoint(checkpoint_toload)

            model = self.model
            dataset = self.dataset
            dynamics = dataset.dynamics
            raise NotImplementedError

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
                lambda_val = self.dataset.lambda_int_2
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
                