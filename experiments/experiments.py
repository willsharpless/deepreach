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
            nonlin_scale=False, nonlin_scale_e_step=10000,
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
        new_weight = 1
        new_weight_hopf = 1
        new_weight_diff_con = 1
        JIp_s_max, JIp_s = 0., 0.
        gamma_orig, mu_orig, alpha_orig = self.dataset.dynamics.gamma, self.dataset.dynamics.mu, self.dataset.dynamics.alpha

        with tqdm(total=len(train_dataloader) * epochs) as pbar:
            train_losses = []
            last_CSL_epoch = -1
            for epoch in range(0, epochs):
                if self.dataset.pretrain: # skip CSL
                    last_CSL_epoch = epoch
                time_interval_length = (self.dataset.counter/self.dataset.counter_end)*(self.dataset.tMax-self.dataset.tMin)
                CSL_tMax = self.dataset.tMin + int(time_interval_length/CSL_dt)*CSL_dt

                ## FIXME: make it better
                if nonlin_scale and not(self.dataset.pretrain) and not(self.dataset.hopf_pretrain) and epoch % nonlin_scale_e_step:
                    e_perc = ((epoch - nonlin_scale_e_step)/(epochs - nonlin_scale_e_step)) # instead of epoch/epochs, this gives 1 step to switch from hopf to pde loss
                    self.dataset.dynamics.gamma = e_perc * gamma_orig
                    self.dataset.dynamics.mu = e_perc * mu_orig
                    self.dataset.dynamics.alpha = e_perc * alpha_orig
                
                # self-supervised learning
                for step, (model_input, gt) in enumerate(train_dataloader):
                    start_time = time.time()
                
                    if self.timing: start_time_2 = time.time()
                    model_input = {key: value.cuda() for key, value in model_input.items()}
                    gt = {key: value.cuda() for key, value in gt.items()}

                    model_results = self.model({'coords': model_input['model_coords']})
                    if self.timing: print("Sample Evaluation took:", time.time() - start_time_2)

                    if self.timing: start_time_2 = time.time()
                    states = self.dataset.dynamics.input_to_coord(model_results['model_in'].detach())[..., 1:]
                    values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1))
                    dvs = self.dataset.dynamics.io_to_dv(model_results['model_in'], model_results['model_out'].squeeze(dim=-1))
                    boundary_values = gt['boundary_values']
                    if self.dataset.dynamics.loss_type == 'brat_hjivi':
                        reach_values = gt['reach_values']
                        avoid_values = gt['avoid_values']
                    dirichlet_masks = gt['dirichlet_masks']
                    if self.timing: print("Pre-loss Computation took:", time.time() - start_time_2)

                    if self.timing: start_time_2 = time.time()
                    if self.dataset.dynamics.loss_type == 'brt_hjivi':
                        losses = loss_fn(states, values, dvs[..., 0], dvs[..., 1:], boundary_values, dirichlet_masks, model_results['model_out'])
                    elif self.dataset.dynamics.loss_type == 'brt_hjivi_hopf':
                        hopf_values = gt['hopf_values']
                        if not(self.dataset.use_bank) or self.dataset.hopf_pretrain_counter == 0:
                            learned_hopf_values = values
                        else:
                            model_results_hopf = self.model({'coords': gt['model_coords_hopf']})
                            learned_hopf_values = self.dataset.dynamics.io_to_value(model_results_hopf['model_in'].detach(), model_results_hopf['model_out'].squeeze(dim=-1))   
                        losses = loss_fn(states, values, dvs[..., 0], dvs[..., 1:], boundary_values, dirichlet_masks, model_results['model_out'], hopf_values, learned_hopf_values, self.dataset.hopf_pretrain)
                    elif self.dataset.dynamics.loss_type == 'brat_hjivi':
                        losses = loss_fn(states, values, dvs[..., 0], dvs[..., 1:], boundary_values, reach_values, avoid_values, dirichlet_masks, model_results['model_out'])
                    else:
                        raise NotImplementedError
                    if self.timing: print("Loss Computation took:", time.time() - start_time_2)
                    
                    if self.timing: start_time_2 = time.time()
                    ## Switch Optimizers/Rates (after Hopf Pretraining)
                    if dual_lr and not(self.dataset.hopf_pretrain) and self.dataset.hopf_pretrained:
                        optim = optim_std
                        lr_scheduler = lr_scheduler_std
                    if self.timing: print("Loss Scheduler took:", time.time() - start_time_2)
                    
                    if use_lbfgs:
                        def closure():
                            optim.zero_grad()
                            train_loss = 0.
                            for loss_name, loss in losses.items():
                                train_loss += loss.mean() 
                            train_loss.backward()
                            return train_loss
                        optim.step(closure)

                    # Adjust the relative magnitude of the losses if required
                    if self.dataset.dynamics.deepreach_model in ['vanilla', 'diff'] and adjust_relative_grads:
                        if losses['diff_constraint_hom'] > 0.01:
                            params = OrderedDict(self.model.named_parameters())
                            # Gradients with respect to the PDE loss
                            optim.zero_grad()
                            losses['diff_constraint_hom'].backward(retain_graph=True)
                            grads_PDE = []
                            for key, param in params.items():
                                grads_PDE.append(param.grad.view(-1))
                            grads_PDE = torch.cat(grads_PDE)

                            # Gradients with respect to the boundary loss
                            optim.zero_grad()
                            losses['dirichlet'].backward(retain_graph=True)
                            grads_dirichlet = []
                            for key, param in params.items():
                                grads_dirichlet.append(param.grad.view(-1))
                            grads_dirichlet = torch.cat(grads_dirichlet)

                            # Gradients with respect to the hopf loss
                            if self.dataset.dynamics.loss_type == 'brt_hjivi_hopf':
                                optim.zero_grad()
                                losses['hopf'].backward(retain_graph=True)
                                grads_hopf = []
                                for key, param in params.items():
                                    grads_hopf.append(param.grad.view(-1))
                                grads_hopf = torch.cat(grads_hopf)

                            # # Plot the gradients
                            # import seaborn as sns
                            # import matplotlib.pyplot as plt
                            # fig = plt.figure(figsize=(5, 5))
                            # ax = fig.add_subplot(1, 1, 1)
                            # ax.set_yscale('symlog')
                            # sns.distplot(grads_PDE.cpu().numpy(), hist=False, kde_kws={"shade": False}, norm_hist=True)
                            # sns.distplot(grads_dirichlet.cpu().numpy(), hist=False, kde_kws={"shade": False}, norm_hist=True)
                            # fig.savefig('gradient_visualization.png')

                            # fig = plt.figure(figsize=(5, 5))
                            # ax = fig.add_subplot(1, 1, 1)
                            # ax.set_yscale('symlog')
                            # grads_dirichlet_normalized = grads_dirichlet * torch.mean(torch.abs(grads_PDE))/torch.mean(torch.abs(grads_dirichlet))
                            # sns.distplot(grads_PDE.cpu().numpy(), hist=False, kde_kws={"shade": False}, norm_hist=True)
                            # sns.distplot(grads_dirichlet_normalized.cpu().numpy(), hist=False, kde_kws={"shade": False}, norm_hist=True)
                            # ax.set_xlim([-1000.0, 1000.0])
                            # fig.savefig('gradient_visualization_normalized.png')

                            # Set the new weight according to the paper
                            # num = torch.max(torch.abs(grads_PDE))
                            num = torch.mean(torch.abs(grads_PDE))
                            den = torch.mean(torch.abs(grads_dirichlet))
                            new_weight = 0.9*new_weight + 0.1*num/den
                            losses['dirichlet'] = new_weight*losses['dirichlet']

                            if self.dataset.dynamics.loss_type == 'brt_hjivi_hopf':
                                den = torch.mean(torch.abs(grads_hopf))
                                new_weight_hopf = 0.9*new_weight_hopf + 0.1*num/den
                                losses['hopf'] = new_weight_hopf*losses['hopf']

                        writer.add_scalar('weight_scaling', new_weight, total_steps)
                        if self.dataset.dynamics.loss_type == 'brt_hjivi_hopf':
                            writer.add_scalar('weight_scaling_hopf', new_weight_hopf, total_steps)

                    ## Decay Hopf Loss (Works Better if Started in Pretraining)
                    if self.dataset.hopf_loss_decay and self.dataset.dynamics.loss_type == 'brt_hjivi_hopf': # and not(self.dataset.pretrain): # and not(self.dataset.hopf_pretrain):
                        losses['hopf'] = new_weight_hopf * losses['hopf']
                        new_weight_hopf = self.dataset.hopf_loss_decay_w * new_weight_hopf

                    ## Incrementally Introduce Differential Constraint Loss (After All Pretraining)
                    if self.dataset.diff_con_loss_incr and not(self.dataset.pretrain) and not(self.dataset.hopf_pretrain):
                        # losses['diff_constraint_hom'] = (1-new_weight_hopf) * losses['diff_constraint_hom']
                        losses['diff_constraint_hom'] = (1-new_weight_diff_con) * losses['diff_constraint_hom']
                        new_weight_diff_con = self.dataset.hopf_loss_decay_w * new_weight_diff_con

                    # import ipdb; ipdb.set_trace()

                    if self.timing: start_time_2 = time.time()
                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean()

                        if loss_name == 'dirichlet':
                            writer.add_scalar(loss_name, single_loss/new_weight, total_steps)
                        elif loss_name == 'hopf':
                            writer.add_scalar(loss_name, single_loss/new_weight_hopf, total_steps)
                        else:
                            writer.add_scalar(loss_name, single_loss/(1-new_weight_diff_con), total_steps)
                        train_loss += single_loss

                    train_losses.append(train_loss.item())
                    writer.add_scalar("total_train_loss", train_loss, total_steps)
                    if self.timing: print("Loss Combination took:", time.time() - start_time_2)

                    if not total_steps % steps_til_summary:
                        torch.save(self.model.state_dict(),
                                os.path.join(checkpoints_dir, 'model_current.pth'))
                        # summary_fn(model, model_input, gt, model_output, writer, total_steps)

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

                    pbar.update(1)

                    if not total_steps % steps_til_summary:
                        tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))
                        if self.use_wandb:
                            log_dict = {
                                'step': epoch,
                                'train_loss': train_loss}
                            for loss_name, loss in losses.items():
                                log_dict[loss_name + "_loss"] = loss
                            if self.dataset.record_set_metrics:
                                with torch.no_grad():
                                    JIp, FIp, FEp = self.set_metrics_overtime()
                                    JIp_s = smoothing_factor * JIp + (1 - smoothing_factor) * JIp_s
                                    JIp_s_max = max(JIp_s, JIp_s_max)
                                log_dict["Jaccard Index over Time"] = JIp
                                log_dict["Smooth Jaccard Index over Time"] = JIp_s
                                log_dict["Max Smooth Jaccard Index over Time"] = JIp_s_max
                                log_dict["Falsely Included percent over Time"] = FIp
                                log_dict["Falsely Excluded percent over Time"] = FEp
                                log_dict["Mean Absolute Spatial Gradient"] = torch.abs(dvs[..., 1:]).sum() / (self.dataset.numpoints * self.N)
                            wandb.log(log_dict)

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

        if self.dataset.record_set_metrics:
            self.plot_set_metrics_eachtime(epoch, times)
            if self.use_wandb:
                wandb.log({'Time vs. Epoch vs. Set Accuracy compared to DP': wandb.Image(self.t_ep_acc_fig),})
            plt.close()

        if was_training:
            self.model.train()
            self.model.requires_grad_(True)
    
    def validateND(self, epoch, save_path, x_resolution, y_resolution, z_resolution, time_resolution):
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
        
        fig = plt.figure(figsize=(5*len(times), 3*5*1))
        for i in range(3*len(times)):
            # for j in range(len(zs)):
            j = 0
            coords = torch.zeros(x_resolution*y_resolution, self.dataset.dynamics.state_dim + 1)
            coords[:, 0] = times[i % len(times)]
            coords[:, 1:] = torch.tensor(plot_config['state_slices']) # initialized to zero (nothing else to set!)
            if i < len(times): # xN - xi plane
                coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
                coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
            elif i < 2*len(times): # xi - xj plane
                coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 0]
                coords[:, 1 + plot_config['z_axis_idx']] = xys[:, 1]
            else: # xN - (xi = xj) plane
                coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
                coords[:, 2:] = (xys[:, 1] * torch.ones(self.N-1, xys.size()[0])).t()

            with torch.no_grad():
                model_results = self.model({'coords': self.dataset.dynamics.coord_to_input(coords.cuda())})
                values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach())
            
            ax = fig.add_subplot(3, len(times), (j+1) + i)
            ax.set_title('t = %0.2f' % (times[i % len(times)])) #, plot_config['state_labels'][plot_config['z_axis_idx']], zs[j]))
            if i < len(times): # xN - xi plane
                ax.set_xlabel("xN")
                ax.set_ylabel("xi")
            elif i < 2*len(times): # xi - xj plane
                ax.set_xlabel("xi")
                ax.set_ylabel("xj")
            else: # xN - (xi = xj) plane
                ax.set_xlabel("xN")
                ax.set_ylabel("xi=xj")

            ## Plot Inside vs. Outside zero-level set of NN
            s = ax.imshow(1*(values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T <= 0), cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(s, cax=cax)

            ## Plot ground-truth zero-level contour
            n_grid_plane_pts = int(self.dataset.n_grid_pts/3)
            n_grid_len = int(n_grid_plane_pts ** 0.5)
            pix_start = (i // len(times)) * n_grid_plane_pts
            tix_start = (i % len(times)) * self.dataset.n_grid_pts
            # if (i % len(times)) > 0: # FIXME this breaks if the std grid times qty changes (=5 atm)
            #     tix_start += (i % len(times)) * self.dataset.n_grid_pts
            ix = pix_start + tix_start
            Vg = self.dataset.values_DP_grid[ix:ix+n_grid_plane_pts].reshape(n_grid_len, n_grid_len)
            ax.contour(self.dataset.X1g, self.dataset.X2g, Vg.cpu(), [0.])

        fig.savefig(save_path)
        if self.use_wandb:
            wandb.log({
                'step': epoch,
                'val_plot': wandb.Image(fig),
            })
        plt.close()

        if self.dataset.record_set_metrics:
            self.plot_set_metrics_eachtime(epoch, times)
            if self.use_wandb:
                wandb.log({'Time vs. Epoch vs. Set Accuracy compared to DP': wandb.Image(self.t_ep_acc_fig),})
            plt.close()

        if was_training:
            self.model.train()
            self.model.requires_grad_(True)

    def set_metrics_overtime(self):

        # if self.N == 2: do whats here, else: use grid only on slices / actually jk, just fill .dataset loads with proper grids (full for 2D, slices for ND)
        model_results_grid = self.model({'coords': self.dataset.model_coords_grid_allt})
        values_grid = self.dataset.dynamics.io_to_value(model_results_grid['model_in'].detach(), model_results_grid['model_out'].squeeze(dim=-1))
        values_grid_sub0_ixs = torch.argwhere(values_grid <= 0).flatten()

        # n_intersect = np.intersect1d(values_grid_sub0_ixs, self.dataset.values_DP_grid_sub0_ixs).size
        n_intersect = values_grid_sub0_ixs[(values_grid_sub0_ixs.view(1, -1) == self.dataset.values_DP_grid_sub0_ixs.view(-1, 1)).any(dim=0)].size()[0] # ty Amin_Jun
        n_overlap = values_grid_sub0_ixs.size()[0] + self.dataset.values_DP_grid_sub0_ixs.size()[0] - n_intersect
        
        if values_grid_sub0_ixs.size()[0] > 0:
            FIp = (values_grid_sub0_ixs.size()[0] - n_intersect) / values_grid_sub0_ixs.size()[0] # <- wrt true set, wrt grid: (self.dataset.n_grid_t_pts * self.dataset.n_grid_pts)
        else:
            FIp = 1.
        FEp = (self.dataset.values_DP_grid_sub0_ixs.size()[0] - n_intersect) / self.dataset.values_DP_grid_sub0_ixs.size()[0] # <- wrt true set, wrt grid: (self.dataset.n_grid_t_pts * self.dataset.n_grid_pts)
        JIp = n_intersect / n_overlap

        return JIp, FIp, FEp
        
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
                ax.set_zlabel(['JI', 'FI%', 'FE%'][axi])
                ax.view_init(elev=20., azim=-35, roll=0)

        time_data = torch.linspace(self.dataset.tMin, self.dataset.tMax, self.dataset.n_grid_t_pts_hi)
        epoch_data = epoch*torch.ones(self.dataset.n_grid_t_pts_hi)/100

        lw = 1
        self.t_ep_acc_fig_ax1.plot(time_data, epoch_data, JIps, lw=lw, color="blue")
        self.t_ep_acc_fig_ax2.plot(time_data, epoch_data, FIps, lw=lw, color="green")
        self.t_ep_acc_fig_ax3.plot(time_data, epoch_data, FEps, lw=lw, color="red")