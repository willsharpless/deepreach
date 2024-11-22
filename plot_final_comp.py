import wandb
import configargparse
import inspect
import os
import torch
import shutil
import random
import numpy as np
import pickle

from datetime import datetime
from dynamics import dynamics 
from experiments import experiments
from utils import modules, dataio, losses

import multiprocessing as mp

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
        
    p = configargparse.ArgumentParser()
    p.add_argument('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')
    p.add_argument('--mode', type=str, default="train", choices=['all', 'train', 'test'], help="Experiment mode to run (new experiments must choose 'all' or 'train').") #FIXME: required=True instead of default

    # save/load directory options
    p.add_argument('--experiments_dir', type=str, default='./plots', help='Where to save the experiment subdirectory.')
    p.add_argument('--experiment_name', type=str, default='lambda_var', help='Name of the experient subdirectory.') #FIXME: required=True instead of default
    p.add_argument('--use_wandb', default=False, action='store_true', help='use wandb for logging')

    ## Hopf options
    p.add_argument('--hopf_loss', type=str, default='none', choices=['none', 'lin_val_diff', 'lin_val_grad_diff'], help='Method for using Hopf data')
    p.add_argument('--hopf_loss_divisor', default=5, required=False, type=float, help='What to divide the hopf loss by for loss reweighting')
    p.add_argument('--solve_grad', action='store_true', default=False, required=False, help='Compute gradient of linear guide (forced true if grad loss), but slower')
    p.add_argument('--hopf_grad_loss_divisor', default=25, required=False, type=float, help='What to divide the hopf grad loss by for loss reweighting')
    p.add_argument('--hopf_pretrain', action='store_true', default=True, required=False, help='Pretrain hopf conditions')
    p.add_argument('--hopf_pretrain_iters', type=int, default=10000, required=False, help='Number of pretrain iterations with Hopf loss')
    
    p.add_argument('--hopf_loss_decay', action='store_true', default=False, required=False, help='Hopf loss weight decay')
    p.add_argument('--hopf_loss_decay_type', type=str, default='linear', choices=['exponential', 'linear', 'negative_exponential'], help='Type of decay for hopf loss term')
    p.add_argument('--hopf_loss_decay_w', default=1., required=False, type=float, help='Hopf loss decay rate weight')
    p.add_argument('--hopf_loss_decay_early', action='store_true', default=False, required=False, help='Starts hopf loss decay in pretraining')
    p.add_argument('--diff_con_loss_incr', action='store_true', default=False, required=False, help='Increments PDE loss introduction of (1 - hopf decay)')
    p.add_argument('--dual_lr', action='store_true', default=True, required=False, help='Use separate lr for Hopf Pretraining and Training')
    p.add_argument('--lr_hopf', default=2e-5, required=False, type=float, help='Learning rate in hopf pretraining')
    p.add_argument('--lr_hopf_decay_w', default=1, required=False, type=float, help='LR exponential decay rate in hopf pretraining')
    p.add_argument('--nl_scale', action='store_true', default=False, required=False, help='Scales the "amount" of nonlinearity over training') # TODO: add correct contour?
    p.add_argument('--nl_scale_epoch_step', type=int, default=10000, required=False, help='Interval (after pt) to step the nonlinearity scale')
    p.add_argument('--nl_scale_epoch_post', type=int, default=50000, required=False, help='Number of epochs to add after nonlinearity scaling')
    p.add_argument('--temporal_weighting', action='store_true', default=False, required=False, help='Inversely weights the samples in the loss w.r.t. time')
    p.add_argument('--reset_loss_w', action='store_true', default=False, required=False, help='Resets the loss weights to their values at the beginning of training (pre-decay)')
    p.add_argument('--reset_loss_period', type=int, default=500, required=False, help='The loss weight reset period')

    p.add_argument('--gt_metrics', action='store_true', default=True, required=False, help='Compute and score the learned value and set (needs ground truth)')
    p.add_argument('--temporal_loss', action='store_true', default=False, required=False, help='Compute the loss over time chunks (slower)')
    p.add_argument('--capacity_test', action='store_true', default=False, required=False, help='Will use supervised-learning to train with the true solution (needs ground truth)')
    p.add_argument('--debug_params', action='store_true', default=False, required=False, help='Quick params for debugging')
    p.add_argument('--timing', action='store_true', default=False, required=False, help='Gives detailed breakdown of computation times per iteration')
    p.add_argument('--plot_comp_final', action='store_true', default=False, required=False, help='Computes ground-truths for all LLND benchamrks')
    
    p.add_argument('--solve_hopf', action='store_true', default=False, required=False, help='Dynamically makes a state & value bank by iteratively solving the Hopf formula')
    p.add_argument('--use_bank', action='store_true', default=False, required=False, help='Makes/loads a state & value bank to reduce compute')
    p.add_argument('--bank_name', type=str, default='none', required=False, help='Name of the state & value bank file (if none and using bank, will make)')
    p.add_argument('--just_make_hopf_bank', action='store_true', default=False, required=False, help='Just solve a bank of Hopf points (and not train)')
    p.add_argument('--refine_bank', action='store_true', default=False, required=False, help='Will iteratively throw out non-spatially unique points when hopf solving (slow)')
    p.add_argument('--hopf_warm_start', action='store_true', default=False, required=False, help='Passes estimated gradients from DeepReach to the Hopf solvers to warm-start them')
    p.add_argument('--load_hopf_model', action='store_true', default=False, required=False, help='Model to load for the supervision')
    p.add_argument('--load_hopf_model_name', type=str, default='capacity_linear', help='Supervision model name')
    p.add_argument('--load_model_type', type=str, default='learned', choices=['learned', 'DP'], help='Type of loaded model')

    use_wandb = p.parse_known_args()[0].use_wandb
    if use_wandb:
        p.add_argument('--wandb_project', type=str, default='deepreach_hopf_oct20', required=False, help='wandb project')
        p.add_argument('--wandb_entity', type=str, default='sas-lab', required=False, help='wandb entity')
        p.add_argument('--wandb_group', type=str, default='LessLinearND', required=False, help='wandb group')
        p.add_argument('--wandb_name', type=str, default='test_run', required=False, help='name of wandb run')

    mode = p.parse_known_args()[0].mode

    if (mode == 'all') or (mode == 'train'):
        p.add_argument('--seed', type=int, default=0, required=False, help='Seed for the experiment.')

        # load experiment_class choices dynamically from experiments module
        experiment_classes_dict = {name: clss for name, clss in inspect.getmembers(experiments, inspect.isclass) if clss.__bases__[0] == experiments.Experiment}
        p.add_argument('--experiment_class', type=str, default='DeepReachHopf', choices=experiment_classes_dict.keys(), help='Experiment class to use.')
        # load special experiment_class arguments dynamically from chosen experiment class
        experiment_class = experiment_classes_dict[p.parse_known_args()[0].experiment_class]
        experiment_params = {name: param for name, param in inspect.signature(experiment_class.init_special).parameters.items() if name != 'self'}
        for param in experiment_params.keys():
            if param == "N" or param == "timing": continue
            p.add_argument('--' + param, type=experiment_params[param].annotation, required=True, help='special experiment_class argument')

        # simulation data source options
        p.add_argument('--numpoints', type=int, default=65000, help='Number of points in simulation data source __getitem__.') # weird way to say batch size
        p.add_argument('--pretrain', action='store_true', default=True, required=False, help='Pretrain dirichlet conditions')
        p.add_argument('--pretrain_iters', type=int, default=2000, required=False, help='Number of pretrain iterations')
        p.add_argument('--tMin', type=float, default=0.0, required=False, help='Start time of the simulation')
        p.add_argument('--tMax', type=float, default=1.0, required=False, help='End time of the simulation')
        p.add_argument('--counter_start', type=int, default=0, required=False, help='Defines the initial time for the curriculum training')
        p.add_argument('--counter_end', type=int, default=-1, required=False, help='Defines the linear step for curriculum training starting from the initial time')
        p.add_argument('--num_src_samples', type=int, default=1000, required=False, help='Number of source samples (initial-time samples) at each time step')
        p.add_argument('--num_target_samples', type=int, default=0, required=False, help='Number of samples inside the target set')
        p.add_argument('--baseline', action='store_true', default=False, required=False, help='Baseline DeepReach method (no Hopf)')

        # model options
        p.add_argument('--model', type=str, default='sine', required=False, choices=['sine', 'tanh', 'sigmoid', 'relu'], help='Type of model to evaluate, default is sine.')
        p.add_argument('--model_mode', type=str, default='mlp', required=False, choices=['mlp', 'rbf', 'pinn'], help='Whether to use uniform velocity parameter')
        p.add_argument('--num_hl', type=int, default=3, required=False, help='The number of hidden layers')
        p.add_argument('--num_nl', type=int, default=512, required=False, help='Number of neurons per hidden layer.')
        p.add_argument('--deepreach_model', type=str, default='exact', required=False, choices=['exact', 'diff', 'vanilla'], help='deepreach model')

        # training options
        p.add_argument('--epochs_til_ckpt', type=int, default=1000, help='Time interval in seconds until checkpoint is saved.')
        p.add_argument('--steps_til_summary', type=int, default=100, help='Time interval in seconds until tensorboard summary is saved.')
        p.add_argument('--batch_size', type=int, default=1, help='Batch size used during training (irrelevant, since len(dataset) == 1).')
        p.add_argument('--lr', type=float, default=1e-5, help='learning rate. default=2e-6')
        p.add_argument('--lr_decay_w', default=1., required=False, type=float, help='LR Exponential Decay Rate') # 1 or 0.9999
        p.add_argument('--num_epochs', type=int, default=30000, help='Number of epochs to train for.')
        p.add_argument('--clip_grad', default=0.0, type=float, help='Clip gradient.')
        p.add_argument('--use_lbfgs', default=False, type=bool, help='use L-BFGS.')
        p.add_argument('--adj_rel_grads', default=False, type=bool, help='adjust the relative magnitude of the losses') # adds 0.05s/it FYI
        p.add_argument('--dirichlet_loss_divisor', default=1.0, required=False, type=float, help='What to divide the dirichlet loss by for loss reweighting')
        p.add_argument('--no_curr', default=False, action='store_true', help='Flag to turn off curriculum sampling')

        # cost-supervised learning (CSL) options
        p.add_argument('--use_CSL', default=False, action='store_true', help='use cost-supervised learning (CSL)')
        p.add_argument('--CSL_lr', type=float, default=2e-5, help='The learning rate used for CSL')
        p.add_argument('--CSL_dt', type=float, default=0.0025, help='The dt used in rolling out trajectories to get cost labels')
        p.add_argument('--epochs_til_CSL', type=int, default=10000, help='Number of epochs between CSL phases')
        p.add_argument('--num_CSL_samples', type=int, default=1000000, help='Number of cost samples in training dataset for CSL phases')
        p.add_argument('--CSL_loss_frac_cutoff', type=float, default=0.1, help='Fraction of initial cost loss on validation dataset to cutoff CSL phases')
        p.add_argument('--max_CSL_epochs', type=int, default=100, help='Max number of CSL epochs per phase')
        p.add_argument('--CSL_loss_weight', type=float, default=1.0, help='weight of cost loss (relative to PDE loss)')
        p.add_argument('--CSL_batch_size', type=int, default=1000, help='Batch size for training in CSL phases')

        # validation (during training) options
        p.add_argument('--val_x_resolution', type=int, default=200, help='x-axis resolution of validation plot during training')
        p.add_argument('--val_y_resolution', type=int, default=200, help='y-axis resolution of validation plot during training')
        p.add_argument('--val_z_resolution', type=int, default=5, help='z-axis resolution of validation plot during training')
        p.add_argument('--val_time_resolution', type=int, default=5, help='time-axis resolution of validation plot during training')

        ## loss options
        p.add_argument('--minWith', type=str, default='target', choices=['none', 'zero', 'target'], help='BRS vs BRT computation (typically should be using target for BRT)') #FIXME: required=True instead of default
        
        # load dynamics_class choices dynamically from dynamics module
        dynamics_classes_dict = {name: clss for name, clss in inspect.getmembers(dynamics, inspect.isclass) if clss.__bases__[0] == dynamics.Dynamics}
        p.add_argument('--dynamics_class', type=str, default="LessLinearND", choices=dynamics_classes_dict.keys(), help='Dynamics class to use.') #FIXME: required=True instead of default
        # load special dynamics_class arguments dynamically from chosen dynamics class
        dynamics_class = dynamics_classes_dict[p.parse_known_args()[0].dynamics_class]
        dynamics_params = {name: param for name, param in inspect.signature(dynamics_class).parameters.items() if name != 'self'}
        for param in dynamics_params.keys():
            # if param == 'N': continue
            if dynamics_params[param].annotation is bool:
                p.add_argument('--' + param, type=dynamics_params[param].annotation, default=False, help='special dynamics_class argument')
            else:
                p.add_argument('--' + param, type=dynamics_params[param].annotation, required=True, help='special dynamics_class argument')

    if (mode == 'all') or (mode == 'test'):
        p.add_argument('--dt', type=float, default=0.0025, help='The dt used in testing simulations')
        p.add_argument('--checkpoint_toload', type=int, default=None, help="The checkpoint to load for testing (-1 for final training checkpoint, None for cross-checkpoint testing")
        p.add_argument('--num_scenarios', type=int, default=100000, help='The number of scenarios sampled in scenario optimization for testing')
        p.add_argument('--num_violations', type=int, default=1000, help='The number of violations to sample for in scenario optimization for testing')
        p.add_argument('--control_type', type=str, default='value', choices=['value', 'ttr', 'init_ttr'], help='The controller to use in scenario optimization for testing')
        p.add_argument('--data_step', type=str, default='run_basic_recovery', choices=['plot_violations', 'run_basic_recovery', 'plot_basic_recovery', 'collect_samples', 'train_binner', 'run_binned_recovery', 'plot_binned_recovery', 'plot_cost_function'], help='The data processing step to run')

    opt = p.parse_args()

    if opt.debug_params:
        opt.pretrain_iters = 2
        opt.hopf_pretrain_iters = 2
        opt.num_epochs = 20
        opt.epochs_til_ckpt = 10
        opt.use_bank = False

    if opt.capacity_test:
        opt.pretrain_iters = 1
        opt.hopf_pretrain_iters = opt.num_epochs
        opt.hopf_loss = 'lin_val_grad_diff'
        opt.solve_grad = True
        opt.solve_hopf = False
        opt.use_bank = False
        opt.hopf_loss_decay = False

    if opt.solve_hopf:
        opt.use_bank = True
        opt.bank_name = 'none' # force dynamic bank construction #FIXME what if I want to load static hopf bank
        # opt.solve_grad = True # force this for warm-starting?

    if opt.hopf_loss == 'lin_val_grad_diff':
        opt.solve_grad = True # need to solve grad to be able to score it

    if opt.hopf_loss == 'none':
        opt.diff_con_loss_incr = False

    if opt.baseline:
        opt.hopf_loss = 'none'
        opt.temporal_loss = False ## bug
        # opt.numpoints, opt.lr, opt.lr_decay_w = 60000, 1e-5, 1.

    print("\n\nGenerating a Comparison Plot for the N-Dimensional Test Model\n\n")

    comp_model_paths = [

        # Baselines
        "./runs/baseline_50D_300k_b1",
        "./runs/baseline_50D_300k_b2",
        "./runs/baseline_50D_300k_b3",
        "./runs/baseline_50D_300k_b4",

        # LSS NL-Aug
        "./runs/ZLLS_hld100_hgld250_50D_300k_b1",
        "./runs/ZLLS_hld1k_hgld4k_50D_300k_b2",
        "./runs/ZLLS_hld500_hgld4k_50D_300k_b3",
        "./runs/ZLLS_hld100_hgld250_50D_300k_b4",

        # LSS Decays
        "./runs/LSD_50D_10k_dwp4_hld10_b1", 
        "./runs/LSD_50D_10k_dwp2_hld5_lr1e-6_b2", 
        "./runs/LSS_50D_10k_hld5_hgld50_b3", 
        "./runs/LSD_50D_10k_dwp4_hld50_lr5e-6_b4", 

    ]
        
    experiment_dir = os.path.join(opt.experiments_dir, opt.experiment_name)

    orig_opt = opt

    # set the experiment seed
    torch.manual_seed(orig_opt.seed)
    random.seed(orig_opt.seed)
    np.random.seed(orig_opt.seed)

    dynamics_class = getattr(dynamics, orig_opt.dynamics_class)
    dynamics_inst = dynamics_class(**{argname: getattr(orig_opt, argname) for argname in inspect.signature(dynamics_class).parameters.keys() if argname != 'self'})
    dynamics_inst.deepreach_model=orig_opt.deepreach_model
    if orig_opt.hopf_loss != 'none':
        dynamics_inst.loss_type = 'brt_hjivi_hopf' ## TODO: why is loss type in dynamics?

    lamb_dynamics_class = getattr(dynamics, orig_opt.dynamics_class + "lambda") # non-lambda version of dynamics
    lamb_dynamics_inst = lamb_dynamics_class(**{argname: getattr(orig_opt, argname) for argname in inspect.signature(lamb_dynamics_class).parameters.keys() if argname != 'self'})
    lamb_dynamics_inst.deepreach_model=orig_opt.deepreach_model
    lamb_dynamics_inst.loss_type = 'brt_hjivi_hopf'

    loaded_models_dict = {}
    for load_model_path in comp_model_paths:

        load_dir = load_model_path
        model_name = load_model_path.split('/')[-1]

        if "ZLLS" in model_name:
            load_dim = dynamics_inst.input_dim + 1
        else:
            load_dim = dynamics_inst.input_dim

        with open(os.path.join(load_dir, 'orig_opt.pickle'), 'rb') as opt_file:
            loaded_opt = pickle.load(opt_file)

        loaded_model = modules.SingleBVPNet(in_features=load_dim, out_features=1, type=loaded_opt.model, mode=loaded_opt.model_mode,
                                    final_layer_factor=1., hidden_features=loaded_opt.num_nl, num_hidden_layers=loaded_opt.num_hl)
        loaded_model.cuda()

        loaded_models_dict[model_name] = loaded_model
        
        model_path = os.path.join(load_dir, 'training', 'checkpoints', 'model_final.pth')
        loaded_model.load_state_dict(torch.load(model_path)['model']) # FIXME, key only needed for chkpts
    loaded_model=None

    dataset = dataio.ReachabilityDataset(
        dynamics=dynamics_inst, numpoints=orig_opt.numpoints, 
        pretrain=orig_opt.pretrain, pretrain_iters=orig_opt.pretrain_iters, 
        tMin=orig_opt.tMin, tMax=orig_opt.tMax, 
        counter_start=orig_opt.counter_start, counter_end=orig_opt.counter_end, 
        num_src_samples=orig_opt.num_src_samples, num_target_samples=orig_opt.num_target_samples,
        use_hopf=(orig_opt.hopf_loss != 'none'),
        hopf_pretrain=orig_opt.hopf_pretrain, hopf_pretrain_iters=orig_opt.hopf_pretrain_iters,
        no_curriculum=orig_opt.no_curr, record_gt_metrics=orig_opt.gt_metrics,
        use_bank=orig_opt.use_bank, bank_name=orig_opt.bank_name, capacity_test=orig_opt.capacity_test,
        solve_hopf=orig_opt.solve_hopf, solve_grad=orig_opt.solve_grad, hopf_warm_start=orig_opt.hopf_warm_start,
        just_make_hopf_bank=orig_opt.just_make_hopf_bank, refine_bank=orig_opt.refine_bank,
        loaded_model=loaded_model, lambda_var=orig_opt.dynamics_class.endswith("lambda"),
        make_benchmark_gts=orig_opt.plot_comp_final, loaded_dynamics=lamb_dynamics_inst)

    model = modules.SingleBVPNet(in_features=dynamics_inst.input_dim, out_features=1, type=orig_opt.model, mode=orig_opt.model_mode,
                                final_layer_factor=1., hidden_features=orig_opt.num_nl, num_hidden_layers=orig_opt.num_hl)
    # model.cuda()

    experiment_class = getattr(experiments, orig_opt.experiment_class)
    experiment = experiment_class(model=model, dataset=dataset, experiment_dir=experiment_dir, use_wandb=use_wandb)
    experiment.init_special(**{argname: getattr(orig_opt, argname) for argname in inspect.signature(experiment_class.init_special).parameters.keys() if argname != 'self'})

    save_path = os.path.join('./plots/Final_Comparison_Plot_50D.png')
    experiment.plot_final_comparison(save_path, loaded_models_dict, orig_opt.val_x_resolution, orig_opt.val_y_resolution, time_resolution=orig_opt.val_time_resolution, plot_value=True)