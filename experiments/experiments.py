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
            diff_con_loss_incr=False, hopf_loss_decay_rate = 'exponential',
            nonlin_scale=False, nl_scale_epoch_step=10000, nl_scale_epoch_post=10000, 
            record_temporal_loss = False,
        ):
        was_eval = not self.model.training
        self.model.train()
        self.model.requires_grad_(True)

        train_dataloader = DataLoader(self.dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=0)

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

        total_steps = 0
        JIp_s_max, JIp_s = 0., 0.
        gamma_orig, mu_orig, alpha_orig = self.dataset.dynamics.gamma, self.dataset.dynamics.mu, self.dataset.dynamics.alpha
        total_pretrain_iters = self.dataset.pretrain_iters + self.dataset.hopf_pretrain_iters
        self.total_pretrain_iters = total_pretrain_iters
        self.epochs = epochs
        nl_perc = 0.

        ## Dynamic Weighting
        loss_weights = {'dirichlet': 1., 'hopf': 1., 'diff_constraint_hom': 1.}
        if diff_con_loss_incr:
            loss_weights['diff_constraint_hom'] = 0.
        if hopf_loss_decay_rate == 'negative_exponential': 
            loss_weights['hopf'] = 1 - (hopf_loss_decay_w ** (epochs - 1 - total_pretrain_iters))
        if hopf_loss == 'lin_val_grad_diff':
            loss_weights['hopf_grad'] = loss_weights['hopf']

        with tqdm(total=len(train_dataloader) * epochs) as pbar:
            train_losses = []
            last_CSL_epoch = -1
            for epoch in range(0, epochs):
                if self.dataset.pretrain: # skip CSL
                    last_CSL_epoch = epoch
                time_interval_length = (self.dataset.counter/self.dataset.counter_end)*(self.dataset.tMax-self.dataset.tMin)
                CSL_tMax = self.dataset.tMin + int(time_interval_length/CSL_dt)*CSL_dt

                ## if hopf-solving, check bank deposit orders
                if self.dataset.solve_hopf:
                    if self.dataset.hjpool.jobs:
                        for job in self.dataset.hjpool.jobs:
                            if job.ready():
                                job.get() ## in case of error
                                self.dataset.hjpool.jobs.remove(job)
                    
                    ## Deposit jobs complete, recall with 
                    else:
                        self.dataset.solved_hopf_pts += self.dataset.bank_params["n_deposit"]

                        if self.dataset.hjpool.hopf_warm_start:
                            self.dataset.hjpool.solve_bank_deposit(model=self.model, n_splits=1, blocking=False)
                        else:
                            self.dataset.hjpool.solve_bank_deposit(model=None, n_splits=1, blocking=False)
                        
                        print("\nNew deposits ordered\n")

                        ## FIXME, reset self.dataset.bank_index and block_counter

                        # if reset_after_deposit:
                        #     reset grad steps

                ## Parameter Scaling for Nonlinearity Curriculum #TODO: generalize to other dynamics
                not_pretraining = not(self.dataset.pretrain) and not(self.dataset.hopf_pretrain)
                if nonlin_scale:
                    if not_pretraining and ((epoch-total_pretrain_iters) % nl_scale_epoch_step) == 0 and epoch <= (epochs-nl_scale_epoch_post):
                        nl_perc = ((epoch - total_pretrain_iters)/((epochs - nl_scale_epoch_post) - total_pretrain_iters)) # instead of epoch/(epochs-post), this gives 1 step to switch from hopf to pde loss w/o changin dynamcis
                    elif epoch == (epochs - nl_scale_epoch_post) + 1: # jic epoch / nl_scale_epoch_step is not an integer
                        nl_perc = 1
                    self.dataset.dynamics.gamma = nl_perc * gamma_orig
                    self.dataset.dynamics.mu = nl_perc * mu_orig
                    self.dataset.dynamics.alpha = nl_perc * alpha_orig

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

                    ## Linearly-Guided BRT (Hopf-based)
                    elif hopf_loss != 'none':
                        
                        hopf_values = gt['hopf_values']

                        if hopf_loss == 'lin_val_grad_diff':
                            hopf_grads = gt['hopf_grads']
                        
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

                    ## Standard BRAT
                    elif self.dataset.dynamics.loss_type == 'brat_hjivi':
                        losses = loss_fn(states, values, dvs[..., 0], dvs[..., 1:], boundary_values, reach_values, avoid_values, dirichlet_masks, model_results['model_out'])
                    
                    else:
                        raise NotImplementedError
                    
                    if self.timing: print("Loss Computation took:", time.time() - start_time_2)

                    # print("B/C  loss:", losses['dirichlet'])
                    # print("Hopf loss:", losses['hopf'])
                    # print("Hopf Grad loss:", losses['hopf_grad'])
                    # print("PDE  loss:", losses['diff_constraint_hom'])

                    ## Compute & Record Temporal Loss Distribution
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

                            # print("PDE  loss", tp, ':', losses_t[str(tp)]['diff_constraint_hom'].item())
                    
                    ## Switch Optimizers/Rates (after Hopf Pretraining)
                    if self.timing: start_time_2 = time.time()
                    if dual_lr and not(self.dataset.hopf_pretrain) and self.dataset.hopf_pretrained:
                        optim = optim_std
                        lr_scheduler = lr_scheduler_std
                    if self.timing: print("Loss Scheduler took:", time.time() - start_time_2)
                    
                    ## Decay Hopf Loss(es)
                    if hopf_loss_decay and hopf_loss != 'none': #                         
                        if epoch >= total_pretrain_iters or hopf_loss_decay_early:
                            if hopf_loss_decay_rate == 'exponential' and epoch > total_pretrain_iters or hopf_loss_decay_early:
                                loss_weights['hopf'] = hopf_loss_decay_w * loss_weights['hopf']
                            elif hopf_loss_decay_rate == 'linear':
                                loss_weights['hopf'] = 1 - hopf_loss_decay_w * (epoch - total_pretrain_iters)/(epochs - 1 - total_pretrain_iters)
                            elif hopf_loss_decay_rate == 'negative_exponential' and epoch > total_pretrain_iters:
                                loss_weights['hopf'] = 1 - ((1 - loss_weights['hopf']) / hopf_loss_decay_w)
                            # else:
                            #     raise NotImplementedError
                            # print("epoch", epoch, ", loss_weights['hopf']=", loss_weights['hopf'])
                        loss_weights['hopf'] = min(max(loss_weights['hopf'], 0.), 1.)
                        
                        if hopf_loss == 'lin_val_grad_diff':
                            loss_weights['hopf_grad'] = loss_weights['hopf'] 

                        ## Incrementally Introduce Differential Constraint Loss (After All Pretraining)
                        if diff_con_loss_incr and epoch >= total_pretrain_iters:
                            # new_weight_diff_con = 1. - new_weight_hopf
                            loss_weights['diff_constraint_hom'] = 1 - loss_weights['hopf']
                            # print("epoch", epoch, ", loss_weights['diff_constraint_hom']=", loss_weights['diff_constraint_hom'])
                        
                    # import ipdb; ipdb.set_trace()

                    ## Combine Losses
                    if self.timing: start_time_2 = time.time()
                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean() ## TODO: this is not the right place for this
                        writer.add_scalar(loss_name, single_loss, total_steps)
                        train_loss += loss_weights[loss_name] * single_loss
                        # print(loss_name, ":", loss_weights[loss_name] * single_loss.detach().item())

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
                        tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))
                        
                        if self.use_wandb:
                            log_dict = {
                                'step': epoch,
                                'train_loss': train_loss}
                            
                            for loss_name, loss in losses.items():
                                log_dict[loss_name + "_loss"] = loss

                            if nonlin_scale and not(self.dataset.pretrain) and not(self.dataset.hopf_pretrain):
                                log_dict["Nonlinearity Scale"] = nl_perc

                            if self.dataset.record_gt_metrics:

                                JIp, FIp, FEp, Vmse, DVXmse = self.compute_gt_metrics()
                                JIp_s = smoothing_factor * JIp + (1 - smoothing_factor) * JIp_s
                                JIp_s_max = max(JIp_s, JIp_s_max)

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
                