import torch
import numpy as np
from torch.utils.data import Dataset
import time
import os, sys
import pickle as pkl
import gc
from tqdm.autonotebook import tqdm
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import julia_multiproc
import warnings
warnings.filterwarnings("ignore",message="torch was imported before juliacall. This may cause a segfault.*",category=UserWarning,module="juliacall")
from juliacall import Main as jl, convert as jlconvert

import matplotlib.pyplot as plt
import matplotlib as mpl
import hj_reachability as hj

# uses model input and real boundary fn
class ReachabilityDataset(Dataset):
    def __init__(self, dynamics, numpoints, pretrain, pretrain_iters, tMin, tMax, counter_start, counter_end, num_src_samples, num_target_samples, 
                 use_hopf=False, super_pretrain=False, super_pretrain_iters=0, record_gt_metrics=False, solve_grad=False,
                 dp_manual_load=False, load_packet=None, no_curriculum=False, use_bank=False, bank_name=None, capacity_test=False,
                 solve_hopf=False, hopf_warm_start=False, hopf_time_step=5e-2, num_hopf_workers=5, hopf_starter_numsplits=1000, hopf_deposit_numsplits=100,
                 hopf_opt_p = {"vh":0.01, "stepsz":1, "tol":1e-3, "decay_stepsz":100, "conv_runs_rqd":1, "max_runs":1, "max_its":100},
                 hopf_bank_params = {"n_total":int(2e6), "n_starter":int(2e6), "n_deposit":int(2e5)}, # dynamic refresh
                 just_make_hopf_bank=False, refine_bank=False,
                 loaded_model=None, loaded_dynamics=None,
                 lambda_var=False, zerolambda_LS=False, LS_w_time_curr=False,
                 memory_tracking=False, make_benchmark_gts=False,

                 loaded_model_1=None, loaded_model_2=None,
                 loaded_dynamics_1=None, loaded_dynamics_2=None,
                 lam_slice_super=False,
                 ):

        self.dynamics = dynamics
        self.loaded_dynamics = loaded_dynamics
        self.numpoints = numpoints
        self.pretrain = pretrain
        self.pretrained = False
        self.pretrain_counter = 0
        self.pretrain_iters = pretrain_iters
        self.tMin = tMin 
        self.tMax = tMax 
        self.counter = counter_start 
        self.counter_end = counter_end 
        self.num_src_samples = num_src_samples
        self.num_target_samples = num_target_samples

        self.lam_slice_super = lam_slice_super

        # self.use_hopf = use_hopf # FIXME: old name, means use linear data (not necessarily solve hopf formula)
        self.use_hopf = False
        self.solve_grad = solve_grad
        self.super_pretrain = super_pretrain
        self.super_pretrained = False
        self.super_pretrain_counter = 0
        self.super_pretrain_iters = super_pretrain_iters
        self.record_gt_metrics = record_gt_metrics
        self.no_curriculum = no_curriculum
        self.N = dynamics.N
        self.capacity_test = capacity_test
        self.llnd_path = "value_fns/LessLinear/"
        self.memory_tracking = memory_tracking

        self.solve_hopf = solve_hopf # triggers online hopf formula solving via HopfReachability.jl
        self.num_hopf_workers = num_hopf_workers
        self.hopf_warm_start = hopf_warm_start
        self.hopf_time_step = hopf_time_step
        self.hopf_opt_p = hopf_opt_p
        self.hopf_bank_params = hopf_bank_params
        self.hopf_starter_numsplits = hopf_starter_numsplits
        self.hopf_deposit_numsplits = hopf_deposit_numsplits
        
        self.use_bank = use_bank
        self.make_bank = use_bank and bank_name == 'none'
        self.refine_bank = refine_bank

        if self.solve_hopf: 
            self.make_bank = True # redundant safety

        if self.solve_hopf:
            self.bank_total = hopf_bank_params["n_total"]
            self.numblocks = int(self.bank_total / numpoints) 
        else:
            self.numblocks = 40 # batch-sizes per bank, #FIXME: should be automatic
            self.bank_total = numpoints * self.numblocks

        if self.solve_hopf and (bank_name is None or bank_name == 'none'): 
            bank_name = self.make_bank_name()
        self.bank_name = bank_name
        self.mp_bank = "mp" in bank_name
        self.just_make_hopf_bank = just_make_hopf_bank

        if just_make_hopf_bank:
            self.hopf_starter_numsplits, self.hopf_deposit_numsplits = 10000, 10000
            if refine_bank:
                self.hopf_bank_params = {"n_total":int(4e6), "n_starter":int(4e6), "n_deposit":int(4e6)} # to make static bank
            else:
                self.hopf_bank_params = {"n_total":int(4e6), "n_starter":int(4e6), "n_deposit":int(2e6)} # to make static bank

        # Dynamic Programming Manual Load (added this to skirt WandB sweep + PyCall imcompatibility but still not working)
        self.dp_manual_load = dp_manual_load
        if self.dp_manual_load: 
            self.V_hopf_itp, self.fast_interp, self.V_hopf, self.V_DP_itp, self.V_DP = load_packet

        # Load a pretrained model for hopf supervision
        self.loaded_model = loaded_model
        if loaded_model: self.loaded_model = loaded_model.cuda()
        self.load_hopf_model = loaded_model is not None

        # Load pretrained models for decomposed supervision
        self.loaded_model_1 = loaded_model_1
        self.loaded_model_2 = loaded_model_2
        self.loaded_dynamics_1 = loaded_dynamics_1
        self.loaded_dynamics_2 = loaded_dynamics_2
        if loaded_model_1: self.loaded_model_1 = loaded_model_1.cuda()
        if loaded_model_2: self.loaded_model_2 = loaded_model_2.cuda()
        self.load_decomposed_models = loaded_model_1 is not None and loaded_model_2 is not None

        # Lambda Variation Options
        self.lambda_var = lambda_var
        self.lambda_int_1 = 0.1
        self.lambda_int_2 = 0.2
        self.lambda_int_3 = 0.5
        self.zerolambda_LS = lambda_var and zerolambda_LS
        self.LS_w_time_curr = LS_w_time_curr

        # Benchmark test only
        self.make_benchmark_gts = make_benchmark_gts
        
        ## Compute Linear Value from Model (if hopf loss)
        if use_hopf and not(self.dp_manual_load):

            if self.load_hopf_model:
                pass # must be loaded in mainscript
            
            ## Initialize Julia CPU Workers for Pool Solving Hopf Formula
            elif self.solve_hopf:
                mp.set_start_method('spawn', force=True)
                if self.solve_hopf and not self.record_gt_metrics: AssertionError("Must also interpolate DP solution if solving hopf. Ask Will to fix me.") #BUG: x/0 bug when only this on?
                self.hjpool = julia_multiproc.HopfJuliaPool(self.dynamics, self.hopf_time_step, self.hopf_opt_p,  
                                                    use_hopf=True, # if (manually) False, then using DP itp
                                                    solve_grad=self.solve_grad,
                                                    hopf_warm_start=self.hopf_warm_start,
                                                    gt_metrics=self.record_gt_metrics,
                                                    num_hopf_workers=self.num_hopf_workers) 

            ## Load Linear DP Solution for Proof-of-Concept
            else:
                self.load_julia_DP_interpolation()     
                
        ## Get Ground Truth for Special N-Dimensional Decomposable LessLinear System
        if record_gt_metrics:
            self.init_groundtruth_tests(load_lambda_var=self.lambda_var, make_benchmark_gts=self.make_benchmark_gts, manual_load=self.dp_manual_load)

        ## Make a bank of evaluated points, instead of evaluating online
        if self.make_bank:

            ## Make a Bank of Points from Linear DP Solution
            if not self.solve_hopf:
                self.make_DP_bank()
            
            ## Make Dynamic Bank by Solving Hopf Formula Online
            else:
                self.make_hopf_bank()

        ## Load Memory Map of the Bank
        if self.use_bank:
            self.load_bank()
                    
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        
        if self.memory_tracking:
            print(f"getitem - start, torch.cuda.memory_allocated: {torch.cuda.memory_allocated()/1000000:2.2f} MB")
            print(f"getitem - start, torch.cuda.memory_reserved:  {torch.cuda.memory_reserved()/1000000:2.2f} MB")
            print()
    
        ## Sample Points and Evaluate
        if False: #self.super_pretrain and self.lambda_var: #skipping for now
            model_states_nolam = torch.zeros(self.numpoints, self.dynamics.state_dim-1).uniform_(-1, 1)
            model_states = torch.cat((model_states_nolam, torch.zeros(self.numpoints, 1)), dim=1) # force lambda=0 for linear pretraining
            # TODO could also skip this and train the linear solution everywhere (after fixing loaded lam)
        else:
            model_states = torch.zeros(self.numpoints, self.dynamics.state_dim).uniform_(-1, 1)

        if self.num_target_samples > 0:
            target_state_samples = self.dynamics.sample_target_state(self.num_target_samples)
            model_states[-self.num_target_samples:] = self.dynamics.coord_to_input(torch.cat((torch.zeros(self.num_target_samples, 1), target_state_samples), dim=-1))[:, 1:self.dynamics.state_dim+1]

        if self.pretrain:
            times = torch.full((self.numpoints, 1), self.tMin)

        else:
            if self.super_pretrain or self.no_curriculum or (self.super_pretrained and not self.LS_w_time_curr):
                times = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin)) # during super pt, sample across all time?
            else:
                times = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * (self.counter/self.counter_end))
            times[-self.num_src_samples:, 0] = self.tMin # force include initial time samples

        model_coords = torch.cat((times, model_states), dim=1)        
        if self.dynamics.input_dim > self.dynamics.state_dim + 1: # temporary workaround for having to deal with dynamics classes for parametrized models with extra inputs
            model_coords = torch.cat((model_coords, torch.zeros(self.numpoints, self.dynamics.input_dim - self.dynamics.state_dim - 1)), dim=1)

        if self.memory_tracking:
            print(f"getitem - after model_coords, torch.cuda.memory_allocated: {torch.cuda.memory_allocated()/1000000:2.2f} MB")
            print(f"getitem - after model_coords, torch.cuda.memory_reserved:  {torch.cuda.memory_reserved()/1000000:2.2f} MB")
            print() 

        ## Get Hopf value
        if self.use_hopf:   

            if self.pretrain or self.load_hopf_model:
                hopf_values = torch.zeros(self.numpoints)
                if self.solve_grad:
                    hopf_grads = torch.zeros(self.numpoints, self.dynamics.state_dim)

                if self.zerolambda_LS and not self.pretrain and not self.super_pretrain: # only when model loading and out of pretraining
                    model_states_nolam = torch.zeros(self.numpoints, self.dynamics.state_dim-1).uniform_(-1, 1)
                    model_states_lam0 = torch.cat((model_states_nolam, torch.zeros(self.numpoints, 1)), dim=1) # force lambda=0 for linear pretraining
                    model_coords_hopf = torch.cat((times, model_states_lam0), dim=1)

            ## Sample Bank of Hopf-Evaluated Points
            elif self.use_bank:

                if not self.solve_hopf or self.hopf_bank_params["n_starter"] == self.hopf_bank_params["n_total"]:
                    sample_index = self.bank_index[self.block_counter * self.numpoints : (self.block_counter+1) * self.numpoints]
                
                    self.block_counter += 1
                    if self.block_counter == self.numblocks:
                        self.bank_index = torch.from_numpy(np.random.permutation(self.bank_total)) # reshuffle
                        self.block_counter = 0
                else:                    
                    sample_index = torch.from_numpy(np.random.permutation(self.max_solved_ix))[:self.numpoints]

                if not self.solve_hopf or self.hopf_bank_params["n_starter"] == self.hopf_bank_params["n_total"]:
                    bank_sample = torch.from_numpy(self.bank[sample_index, :])
                else:
                    with self.hjpool.lock:
                        bank_sample = torch.from_numpy(self.bank[sample_index, :])                

                # separate states so pde loss is not restricted to small bank sample
                model_coords_hopf = bank_sample[:, 0:self.dynamics.state_dim+1]
                hopf_values = bank_sample[:, self.dynamics.state_dim+2]

                if self.solve_grad:
                    if self.solve_hopf or self.mp_bank: ## TODO: fix static bank format to match hopf
                        hopf_grads = bank_sample[:, self.dynamics.state_dim+4:2*self.dynamics.state_dim+4]
                    else:
                        hopf_grads = bank_sample[:, self.dynamics.state_dim+3:]

            ## Compute Dynamic Programming Interpolation
            else:
                try: 
                    if self.solve_grad:
                        hopf_values, hopf_grads = self.V_hopf_grad(self.dynamics.input_to_coord(model_coords).t())
                    else:
                        hopf_values = self.V_hopf(self.dynamics.input_to_coord(model_coords).t()) # is slow in high d (yields)
                except: 
                    if self.solve_grad:
                        hopf_values, hopf_grads = self.V_hopf_grad(0.999 * self.dynamics.input_to_coord(model_coords).t())
                    else:
                        hopf_values = self.V_hopf(0.999 * self.dynamics.input_to_coord(model_coords).t()) # rare itp-lim fp issue
        
        coords_io = self.dynamics.input_to_coord(model_coords)
        states_io, times_io = coords_io[..., 1:], coords_io[..., 0:1]
        
        boundary_values = self.dynamics.boundary_fn(states_io, times_io)

        ## Reach-Avoid Data
        if self.dynamics.loss_type == 'brat_hjivi':
            reach_values = self.dynamics.reach_fn(states_io, times_io)
            avoid_values = self.dynamics.avoid_fn(states_io, times_io)

        ## Multi-Objective Data
        if self.dynamics.loss_type == 'mulob_hjivi':

            if hasattr(self.dynamics, 'reach_fn') and hasattr(self.dynamics, 'avoid_fn'):
                bc_values_1 = self.dynamics.reach_fn(states_io, times_io)
                bc_values_2 = self.dynamics.avoid_fn(states_io, times_io)
                
            elif hasattr(self.dynamics, 'reach_fn_1') and hasattr(self.dynamics, 'reach_fn_2'):
                bc_values_1 = self.dynamics.reach_fn_1(states_io, times_io)
                bc_values_2 = self.dynamics.reach_fn_2(states_io, times_io)

            if not self.load_decomposed_models:
                if not self.lambda_var:
                    if self.solve_grad:
                        gt_decomposed_values_1, gt_decomposed_grads_1 = self.V_DP_1(model_coords.t())
                        gt_decomposed_values_2, gt_decomposed_grads_2 = self.V_DP_2(model_coords.t())
                    else:
                        gt_decomposed_values_1 = self.V_DP_1(model_coords.t())
                        gt_decomposed_values_2 = self.V_DP_2(model_coords.t())
                        gt_decomposed_grads_1, gt_decomposed_grads_2 = torch.empty(0), torch.empty(0)

                else: # remove lambda (same coords)
                    if self.solve_grad:
                        gt_decomposed_values_1, gt_decomposed_grads_1 = self.V_DP_1(model_coords[..., :-1].t())
                        gt_decomposed_values_2, gt_decomposed_grads_2 = self.V_DP_2(model_coords[..., :-1].t())
                    else:
                        gt_decomposed_values_1 = self.V_DP_1(model_coords[..., :-1].t())
                        gt_decomposed_values_2 = self.V_DP_2(model_coords[..., :-1].t())
                        gt_decomposed_grads_1, gt_decomposed_grads_2 = torch.empty(0), torch.empty(0)
            else:
                gt_decomposed_values_1, gt_decomposed_values_2, gt_decomposed_grads_1, gt_decomposed_grads_2 = torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)
                    
                # FIXME: for BRAT decomp, need V_DP_2 == avoid_fn (NOT avoid_value)

            if self.lam_slice_super and self.lambda_var:
                norm_lambda_target_hi = self.dynamics.lambda_target_hi / self.dynamics.state_var[-1]
                norm_lambda_target_lo = self.dynamics.lambda_target_lo / self.dynamics.state_var[-1]
                model_coords_poslam = torch.cat((model_coords[..., :-1], norm_lambda_target_hi + 0*model_coords[..., -2:-1]), -1) 
                model_coords_neglam = torch.cat((model_coords[..., :-1], norm_lambda_target_lo + 0*model_coords[..., -2:-1]), -1) 
            else:
                model_coords_poslam, model_coords_neglam = model_coords, model_coords

        if self.pretrain:
            dirichlet_masks = torch.ones(model_coords.shape[0]) > 0
        else:
            # only enforce initial conditions around self.tMin
            dirichlet_masks = (model_coords[:, 0] == self.tMin)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.super_pretrain:
            self.super_pretrain_counter += 1
        elif self.counter < self.counter_end:
            self.counter += 1

        if self.memory_tracking:
            print(f"getitem - end, torch.cuda.memory_allocated: {torch.cuda.memory_allocated()/1000000:2.2f} MB")
            print(f"getitem - end, torch.cuda.memory_reserved:  {torch.cuda.memory_reserved()/1000000:2.2f} MB")
            print()    

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False
            self.pretrained = True
            print("\n\n ----------------  FINISHED BC PRETRAINING  ------------------- \n")

        if self.super_pretrain and self.super_pretrain_counter == self.super_pretrain_iters:
            self.super_pretrain = False
            self.super_pretrained = True
            print("\n\n ---------------- FINISHED SUPERVISION PRETRAINING ------------------- \n")


        if self.dynamics.loss_type == 'brt_hjivi':
            return {'model_coords': model_coords}, {'boundary_values': boundary_values, 'dirichlet_masks': dirichlet_masks}
        
        elif self.dynamics.loss_type == 'brt_hjivi_hopf':
            ## UGLY
            if (not(self.use_bank) or self.super_pretrain_counter == 0) and not self.solve_grad:
                return {'model_coords': model_coords}, {'boundary_values': boundary_values, 'dirichlet_masks': dirichlet_masks, 'hopf_values': hopf_values}
            elif (not self.use_bank or self.super_pretrain_counter == 0) and (not self.zerolambda_LS or self.pretrain or self.super_pretrain or self.super_pretrain_counter == self.super_pretrain_iters):
                return {'model_coords': model_coords}, {'boundary_values': boundary_values, 'dirichlet_masks': dirichlet_masks, 'hopf_values': hopf_values, 'hopf_grads': hopf_grads}
            elif not self.solve_grad:
                return {'model_coords': model_coords}, {'boundary_values': boundary_values, 'dirichlet_masks': dirichlet_masks, 'hopf_values': hopf_values, 'model_coords_hopf': model_coords_hopf}
            else:
                return {'model_coords': model_coords}, {'boundary_values': boundary_values, 'dirichlet_masks': dirichlet_masks, 'hopf_values': hopf_values, 'model_coords_hopf': model_coords_hopf, 'hopf_grads': hopf_grads}
        
        elif self.dynamics.loss_type == 'brat_hjivi':
            return {'model_coords': model_coords}, {'boundary_values': boundary_values, 'reach_values': reach_values, 'avoid_values': avoid_values, 'dirichlet_masks': dirichlet_masks}
        
        elif self.dynamics.loss_type == 'mulob_hjivi':
            return {'model_coords': model_coords}, {'boundary_values': boundary_values, 'bc_values_1': bc_values_1, 'bc_values_2': bc_values_2, 
                                                    'gt_decomposed_values_1':gt_decomposed_values_1, 'gt_decomposed_values_2':gt_decomposed_values_2, 
                                                    'gt_decomposed_grads_1':gt_decomposed_grads_1, 'gt_decomposed_grads_2':gt_decomposed_grads_2, 
                                                    'model_coords_poslam':model_coords_poslam, 'model_coords_neglam':model_coords_neglam,
                                                    'dirichlet_masks': dirichlet_masks}
        else:
            raise NotImplementedError
        
    def load_julia_DP_interpolation(self):
        
        ## julia code opt returns values or values and grads
        jl.seval("using JLD, JLD2, Interpolations")
        fast_interp_exec = """
        function fast_interp(_V_itp, tXg; compute_grad=false)
            Vg = zeros(size(tXg,2))
            for i=1:length(Vg); Vg[i] = _V_itp(tXg[:,i][end:-1:1]...); end # (assumes t in first row)
            if !compute_grad
                return Vg
            else
                G = zeros(size(tXg,2), size(tXg,1)-1)
                for i=1:size(G,1); G[i,:] = Interpolations.gradient(_V_itp, tXg[:,i][end:-1:1]...)[end-1:-1:1]; end # (assumes t in first row)
                return Vg, G
            end
        end
        """
        self.fast_interp = jl.seval(fast_interp_exec)
        ## FIXME: rename "tXg"-> "tX" and some of "hopf_loss" -> "linear"
        
        if self.N == 2:

            ## Load 2D DP Solution
            # self.V_hopf_itp = jl.load("lin2d_hopf_interp_linear.jld")["V_itp"] ## (using gt)
            self.V_hopf_itp = LessLinear2D_interpolations = jl.load(self.llnd_path + f"interps/LessLinear2D1i_interpolations_res1e-2_r{int(100*self.dynamics.goalR_2d)}e-2_c{int(abs(self.dynamics.gamma))}.jld", "LessLinear2D_interpolations")["g0_m0_a0"]
            self.V_hopf = lambda tXg: torch.from_numpy(self.fast_interp(self.V_hopf_itp, tXg.numpy()).to_numpy())
            if self.solve_grad:
                self.V_hopf_grad = lambda tXg: torch.from_numpy(self.fast_interp(self.V_hopf_itp, tXg.numpy(), compute_grad=True).to_numpy())
        
        elif self.N > 2:
            
            ## Load 2D DP Solution
            if self.dynamics.gamma != 0:
                LessLinear2D_interpolations = jl.load(self.llnd_path + f"interps/LessLinear2D1i_interpolations_res1e-2_r{int(100*self.dynamics.goalR_2d)}e-2_c{int(abs(self.dynamics.gamma))}.jld", "LessLinear2D_interpolations")
            else:
                LessLinear2D_interpolations = jl.load(self.llnd_path + f"interps/LessLinear2D1i_interpolations_res1e-2_r{int(100*self.dynamics.goalR_2d)}e-2_c5.jld", "LessLinear2D_interpolations")
            self.V_hopf_itp = LessLinear2D_interpolations["g0_m0_a0"] ## (using gt)
            
            ## Capacity Test: Load Ground Truth for Supervised-Learning
            if self.capacity_test:
                model_key = "g" + str(int(self.dynamics.gamma)) + "_m" + str(int(self.dynamics.mu)) + "_a"  + str(int(self.dynamics.alpha))
                self.V_hopf_itp = LessLinear2D_interpolations[model_key]

            ## ND Projection & Interpolation of Loaded 2D DP Solution
            def V_N_hopf_itp(tXg):
                V = 0 * tXg[0,:]
                for i in range(self.N-1):
                    V += torch.from_numpy(self.fast_interp(self.V_hopf_itp, tXg[[0, 1, 2+i], :].numpy()).to_numpy())
                return V
            self.V_hopf = V_N_hopf_itp ## TODO: rename once everything works with actual solve

            if self.solve_grad:
                def V_N_hopf_grad_itp(tXg):
                    V = 0 * tXg[0,:]
                    DV = 0 * tXg[1:,:].t()
                    for i in range(self.N-1):
                        Vi, DVi = self.fast_interp(self.V_hopf_itp, tXg[[0, 1, 2+i], :].numpy(), compute_grad=True)
                        V += torch.from_numpy(Vi.to_numpy())
                        DV[:, [0, 1+i]] += torch.from_numpy(DVi.to_numpy()) # assumes xN first
                    return V, DV
                self.V_hopf_grad = V_N_hopf_grad_itp
    
    def init_groundtruth_tests(self, load_lambda_var=False, make_benchmark_gts=False, manual_load=False, make_gt_solutions=False, decomposed=True, fd_grad=False):
        
        ## Manual load (WandB sweeps need this if using juliacall)
        if manual_load:
            pass
        
        ## Python Ground Truth Solutions
        elif hasattr(self.dynamics, "name") and self.dynamics.name in ["Conveyor","Canoe"]:
            
            # TODO: just for testing, make parsed arg later
            bd_tag = "bdbc" if self.dynamics.bounded_bc else "ubdbc"
            if self.dynamics.name == "Conveyor":
                # self.gt_key = "bounded/axes/Conveyor2D_BRAAT_bdbc_axes_lin"
                self.gt_key = f"bounded/{self.dynamics.avoid_type}/Conveyor2D_BRAAT_{bd_tag}_{self.dynamics.avoid_type}_lin"
            elif self.dynamics.name == "Canoe":
                self.gt_key = f"bounded/Canoe2D_BRRT_{bd_tag}_tv"

            ## Load HJR solutions
            if not make_gt_solutions:

                print(f"Loading *{self.gt_key}* ground truth solutions (hj_reachability.py)...")

                grid_params = np.load(f"value_fns/{self.dynamics.name}/{self.dynamics.name}2D_base_params.npz")
                
                solution_grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(
                    grid_params["lbs"] - grid_params["grid_pad"],
                    grid_params["ubs"] + grid_params["grid_pad"]), 
                    [grid_params["grid_L"] for _ in range(2)])

                solution_times = grid_params["times"]

                self.V_DP_2d = np.load(f"value_fns/{self.dynamics.name}/solutions/{self.gt_key}_V.npz")["V"]

                if self.dynamics.name == "Conveyor":
                    
                    self.V_DP_2d_1 = np.load(f"value_fns/{self.dynamics.name}/solutions/{self.gt_key}_Vr.npz")["Vr"]
                    self.V_DP_2d_2 = np.load(f"value_fns/{self.dynamics.name}/solutions/{self.gt_key}_Va.npz")["Va"]

                elif self.dynamics.name == "Canoe":
                    
                    self.V_DP_2d_1 = np.load(f"value_fns/{self.dynamics.name}/solutions/{self.gt_key}_V1.npz")["V1"]
                    self.V_DP_2d_2 = np.load(f"value_fns/{self.dynamics.name}/solutions/{self.gt_key}_V2.npz")["V2"]
                
                else:
                    raise NotImplementedError
                
            ## Solve hj_reachability.py for DP Solutions
            else:
                # print("Generating ground truth solutions with dynamic programming (hj_reachability.py)...")
                # solution_DP = solve_hjr_DP()
                raise NotImplementedError

            ## Ground Truth interpolated composition of values and gradients
            def V_N_DP_itp_combo(tXg, V_DP_base, grid, time_grid, compute_grad=False, shared_x0=True, dim_sub=1, flip_sign=False):
                
                # V = 0 * tXg[0,:]
                i = 1 # main diagonal only (any subsys i is equiv on diag)
                state_key = [0, 1, 2] # time, first two states

                tXg[[0], :] = -tXg[[0], :] # DR convention to use positive times
                sign_mag = -1. if flip_sign else 1.

                # if shared_x0:
                #     base = [0, 1]
                #     sub_range = list(range(1 + dim_sub*i, 1 + dim_sub*(i+1)))
                # else:
                #     base = [0]
                #     sub_range = list(range(dim_sub*i, dim_sub*(i+1)+1))
                # state_key = base + sub_range
                # could try N-sum test for comp?
                                
                if not compute_grad:
                    values_itp = grid.interpolate_timespace_batch(V_DP_base, time_grid, tXg[state_key, :].t().numpy(), return_grad=False)
                    values_itp_tensor = torch.from_numpy(values_itp.__array__().copy())
                    
                    return sign_mag * values_itp_tensor
                
                ## Interpolate Gradient
                else:
                    # DV = 0 * tXg[1:,:].t()
                    values_itp, grads_itp = grid.interpolate_timespace_batch(V_DP_base, time_grid, tXg[state_key, :].t().numpy(), return_grad=True) 
                
                    values_itp_tensor = torch.from_numpy(values_itp.__array__().copy())
                    grads_itp_tensor = torch.from_numpy(grads_itp.__array__().copy()) # NOTE: using time grads!
                    
                    return sign_mag * values_itp_tensor, sign_mag * grads_itp_tensor
                    
            ## Define multiobjective and decomposed interpolation fns
            def V_N_DP_itp(tXg):
                return V_N_DP_itp_combo(tXg, self.V_DP_2d, solution_grid, solution_times, compute_grad=False, shared_x0=self.dynamics.shared_x0, dim_sub=self.dynamics.dim_sub)
            
            def V_N_DP_1_itp(tXg):
                return V_N_DP_itp_combo(tXg, self.V_DP_2d_1, solution_grid, solution_times, compute_grad=False, shared_x0=self.dynamics.shared_x0, dim_sub=self.dynamics.dim_sub)

            def V_N_DP_2_itp(tXg):
                flip_sign = hasattr(self.dynamics, 'avoid_only') and self.dynamics.avoid_only
                return V_N_DP_itp_combo(tXg, self.V_DP_2d_2, solution_grid, solution_times, compute_grad=False, shared_x0=self.dynamics.shared_x0, dim_sub=self.dynamics.dim_sub, flip_sign=flip_sign)
            
            self.V_DP = V_N_DP_itp
            self.V_DP_1 = V_N_DP_1_itp
            self.V_DP_2 = V_N_DP_2_itp

            if self.solve_grad:

                def V_N_DP_itp_grad(tXg):
                    return V_N_DP_itp_combo(tXg, self.V_DP_2d, solution_grid, solution_times, compute_grad=True, shared_x0=self.dynamics.shared_x0, dim_sub=self.dynamics.dim_sub)

                def V_N_DP_1_itp_grad(tXg):
                    return V_N_DP_itp_combo(tXg, self.V_DP_2d_1, solution_grid, solution_times, compute_grad=True, shared_x0=self.dynamics.shared_x0, dim_sub=self.dynamics.dim_sub)

                def V_N_DP_2_itp_grad(tXg):
                    flip_sign = hasattr(self.dynamics, 'avoid_only') and self.dynamics.avoid_only
                    return V_N_DP_itp_combo(tXg, self.V_DP_2d_2, solution_grid, solution_times, compute_grad=True, shared_x0=self.dynamics.shared_x0, dim_sub=self.dynamics.dim_sub, flip_sign=flip_sign)

                self.V_DP_grad = V_N_DP_itp_grad
                self.V_DP_1_grad = V_N_DP_1_itp_grad
                self.V_DP_2_grad = V_N_DP_2_itp_grad
            
            # If decomposed learning, score correctly
            if (hasattr(self.dynamics, 'reach_only') and self.dynamics.reach_only) or (hasattr(self.dynamics, 'reach_1_only') and self.dynamics.reach_1_only):
                self.V_DP = self.V_DP_1
                if self.solve_grad:
                    self.V_DP_grad = self.V_DP_1_grad
            
            elif (hasattr(self.dynamics, 'avoid_only') and self.dynamics.avoid_only) or (hasattr(self.dynamics, 'reach_2_only') and self.dynamics.reach_2_only):
                self.V_DP = self.V_DP_2
                if self.solve_grad:
                    self.V_DP_grad = self.V_DP_2_grad

        ## Julia-Based Ground Truth Solutions (specific to "Linear Semi-Supervision" paper)
        else:

            jl.seval("using JLD, JLD2, Interpolations")
            fast_interp_exec = """
            function fast_interp(_V_itp, tXg; compute_grad=false)
                Vg = zeros(size(tXg,2))
                for i=1:length(Vg); Vg[i] = _V_itp(tXg[:,i][end:-1:1]...); end # (assumes t in first row)
                if !compute_grad
                    return Vg
                else
                    G = zeros(size(tXg,2), size(tXg,1)-1)
                    for i=1:size(G,1); G[i,:] = Interpolations.gradient(_V_itp, tXg[:,i][end:-1:1]...)[end-1:-1:1]; end # (assumes t in first row)
                    return Vg, G
                end
            end
            """
            if not(hasattr(self, 'fast_interp')):
                self.fast_interp = jl.seval(fast_interp_exec)

            if self.dynamics.gamma != 0:
                LessLinear2D_interpolations = jl.load(self.llnd_path + f"interps/LessLinear2D1i_interpolations_res1e-2_r{int(100*self.dynamics.goalR_2d)}e-2_c{int(abs(self.dynamics.gamma))}.jld", "LessLinear2D_interpolations")
            else:
                LessLinear2D_interpolations = jl.load(self.llnd_path + f"interps/LessLinear2D1i_interpolations_res1e-2_r{int(100*self.dynamics.goalR_2d)}e-2_c5.jld", "LessLinear2D_interpolations")
            
            model_key = "g" + str(int(self.dynamics.gamma)) + "_m" + str(int(self.dynamics.mu)) + "_a"  + str(int(self.dynamics.alpha))
            self.V_DP_itp = LessLinear2D_interpolations[model_key]
            
            if self.N == 2:
                self.V_DP = lambda tXg: torch.from_numpy(self.fast_interp(self.V_DP_itp, tXg.numpy()).to_numpy())
                if self.solve_grad:
                    self.V_DP_grad = lambda tXg: torch.from_numpy(self.fast_interp(self.V_DP_itp, tXg.numpy(), compute_grad=True).to_numpy())

            elif self.N > 2:
                def V_N_DP_itp_combo_julia(tXg):
                    V = 0 * tXg[0,:]
                    for i in range(self.N-1):
                        V += torch.from_numpy(self.fast_interp(self.V_DP_itp, tXg[[0, 1, 2+i], :].numpy()).to_numpy())
                    return V
                self.V_DP = V_N_DP_itp_combo_julia

                self.V_DP_linear_itp = LessLinear2D_interpolations["g0_m0_a0"]
                def V_N_DP_linear_itp_combo_julia(tXg):
                    V = 0 * tXg[0,:]
                    for i in range(self.N-1):
                        V += torch.from_numpy(self.fast_interp(self.V_DP_linear_itp, tXg[[0, 1, 2+i], :].numpy()).to_numpy())
                    return V
                self.V_DP_linear = V_N_DP_linear_itp_combo_julia # for plotting the BRT of Linear Solution

                if self.solve_grad:
                    def V_N_DP_itp_grad_combo_julia(tXg):
                        V = 0 * tXg[0,:]
                        DV = 0 * tXg[1:,:].t()
                        for i in range(self.N-1):
                            Vi, DVi = self.fast_interp(self.V_DP_itp, tXg[[0, 1, 2+i], :].numpy(), compute_grad=True)
                            V += torch.from_numpy(Vi.to_numpy())
                            DV[:, [0, 1+i]] += torch.from_numpy(DVi.to_numpy()) # assumes xN first
                        return V, DV
                    self.V_DP_grad = V_N_DP_itp_grad_combo_julia

                if load_lambda_var:
                    LessLinear2D_interpolations_lam = jl.load(self.llnd_path + f"interps/LessLinear2D1i_interpolations_res1e-2_r{int(100*self.dynamics.goalR_2d)}e-2_c{int(abs(self.dynamics.gamma))}_lambdavar.jld", "LessLinear2D_interpolations")
                    self.V_DP_inlam1_itp = LessLinear2D_interpolations_lam[model_key + f"_lp{int(10 * self.lambda_int_1):1d}"]
                    self.V_DP_inlam2_itp = LessLinear2D_interpolations_lam[model_key + f"_lp{int(10 * self.lambda_int_2):1d}"]
                    self.V_DP_inlam3_itp = LessLinear2D_interpolations_lam[model_key + f"_lp{int(10 * self.lambda_int_3):1d}"]
                    def V_N_DP_inlam1_itp_combo_julia(tXg):
                        V = 0 * tXg[0,:]
                        for i in range(self.N-1):
                            V += torch.from_numpy(self.fast_interp(self.V_DP_inlam1_itp, tXg[[0, 1, 2+i], :].numpy()).to_numpy())
                        return V
                    def V_N_DP_inlam2_itp_combo_julia(tXg):
                        V = 0 * tXg[0,:]
                        for i in range(self.N-1):
                            V += torch.from_numpy(self.fast_interp(self.V_DP_inlam2_itp, tXg[[0, 1, 2+i], :].numpy()).to_numpy())
                        return V
                    def V_N_DP_inlam3_itp_combo_julia(tXg):
                        V = 0 * tXg[0,:]
                        for i in range(self.N-1):
                            V += torch.from_numpy(self.fast_interp(self.V_DP_inlam3_itp, tXg[[0, 1, 2+i], :].numpy()).to_numpy())
                        return V
                    self.V_DP_inlam1 = V_N_DP_inlam1_itp_combo_julia # for plotting the BRT of Linear Solution
                    self.V_DP_inlam2 = V_N_DP_inlam2_itp_combo_julia # for plotting the BRT of Linear Solution
                    self.V_DP_inlam3 = V_N_DP_inlam3_itp_combo_julia # for plotting the BRT of Linear Solution
                
                if make_benchmark_gts:
                    self.V_DP_itp_b1 = LessLinear2D_interpolations["g20_m0_a0"]
                    self.V_DP_itp_b2 = LessLinear2D_interpolations["g-20_m0_a0"]
                    self.V_DP_itp_b3 = LessLinear2D_interpolations["g-20_m-20_a1"]
                    self.V_DP_itp_b4 = LessLinear2D_interpolations["g20_m-20_a1"]

                    def V_N_DP_b1_itp_grad_combo_julia(tXg):
                        V = 0 * tXg[0,:]
                        DV = 0 * tXg[1:,:].t()
                        for i in range(self.N-1):
                            Vi, DVi = self.fast_interp(self.V_DP_itp_b1, tXg[[0, 1, 2+i], :].numpy(), compute_grad=True)
                            V += torch.from_numpy(Vi.to_numpy())
                            DV[:, [0, 1+i]] += torch.from_numpy(DVi.to_numpy()) # assumes xN first
                        return V, DV
                    
                    def V_N_DP_b2_itp_grad_combo_julia(tXg):
                        V = 0 * tXg[0,:]
                        DV = 0 * tXg[1:,:].t()
                        for i in range(self.N-1):
                            Vi, DVi = self.fast_interp(self.V_DP_itp_b2, tXg[[0, 1, 2+i], :].numpy(), compute_grad=True)
                            V += torch.from_numpy(Vi.to_numpy())
                            DV[:, [0, 1+i]] += torch.from_numpy(DVi.to_numpy()) # assumes xN first
                        return V, DV
                    
                    def V_N_DP_b3_itp_grad_combo_julia(tXg):
                        V = 0 * tXg[0,:]
                        DV = 0 * tXg[1:,:].t()
                        for i in range(self.N-1):
                            Vi, DVi = self.fast_interp(self.V_DP_itp_b3, tXg[[0, 1, 2+i], :].numpy(), compute_grad=True)
                            V += torch.from_numpy(Vi.to_numpy())
                            DV[:, [0, 1+i]] += torch.from_numpy(DVi.to_numpy()) # assumes xN first
                        return V, DV
                    
                    def V_N_DP_b4_itp_grad_combo_julia(tXg):
                        V = 0 * tXg[0,:]
                        DV = 0 * tXg[1:,:].t()
                        for i in range(self.N-1):
                            Vi, DVi = self.fast_interp(self.V_DP_itp_b4, tXg[[0, 1, 2+i], :].numpy(), compute_grad=True)
                            V += torch.from_numpy(Vi.to_numpy())
                            DV[:, [0, 1+i]] += torch.from_numpy(DVi.to_numpy()) # assumes xN first
                        return V, DV      
                                 
                    self.V_DP_grad_b1_julia = V_N_DP_b1_itp_grad_combo_julia
                    self.V_DP_grad_b2_julia = V_N_DP_b2_itp_grad_combo_julia
                    self.V_DP_grad_b3_julia = V_N_DP_b3_itp_grad_combo_julia
                    self.V_DP_grad_b4_julia = V_N_DP_b4_itp_grad_combo_julia

        ## Define a fixed spatiotemporal grid to score MSE & Jaccard
        
        # xig_1, xig_2 = torch.arange(-0.99, 1.01, 0.02), torch.arange(-0.99, 1.01, 0.02) # 100 x 100  
        xig_1, xig_2 = torch.arange(-1., 1.02, 0.02), torch.arange(-1., 1.02, 0.02) # 101 x 101, this breaks old (bad) hopf validateND 
        if hasattr(self.dynamics, 'state_scale_score'):
            xig_1 = xig_1 * (self.dynamics.state_scale_score / self.dynamics.state_scale)
            xig_2 = xig_2 * (self.dynamics.state_scale_score / self.dynamics.state_scale)
        
        grid_L = xig_1.size()[0]
        self.X1g, self.X2g = torch.meshgrid(xig_1, xig_2)
        self.model_states_grid_2d = torch.cat((self.X1g.ravel().reshape((1,grid_L**2)), self.X2g.ravel().reshape((1,grid_L**2))), dim=0).t()
        self.n_grid_pts_2d = grid_L**2
        self.n_grid_t_pts, self.n_grid_t_pts_hi = 5, 20

        ## Make a low and high res grids wrt time
        if self.N == 2:

            times = torch.full((self.n_grid_pts_2d, 1), self.tMin) # TODO: remove first time-point if model='exact'
            self.model_coords_grid_allt = torch.cat((times, self.model_states_grid_2d), dim=1) 
            # self.model_coords_grid_allt_hi = torch.cat((times, self.model_states_grid_2d), dim=1) 

            for i in range(self.n_grid_t_pts-1):
                times = torch.full((self.n_grid_pts_2d, 1), (i+1)*(self.tMax - self.tMin)/(self.n_grid_t_pts-1))
                new_coords = torch.cat((times, self.model_states_grid_2d), dim=1) 
                self.model_coords_grid_allt = torch.cat((self.model_coords_grid_allt, new_coords), dim=0) 
            # for i in range(self.n_grid_t_pts_hi-1):
            #     times = torch.full((self.n_grid_pts_2d, 1), (i+1)*(self.tMax - self.tMin)/(self.n_grid_t_pts_hi-1))
            #     new_coords = torch.cat((times, self.model_states_grid_2d), dim=1) 
            #     self.model_coords_grid_allt_hi = torch.cat((self.model_coords_grid_allt_hi, new_coords), dim=0) 

            self.n_grid_pts = self.n_grid_pts_2d

            if not self.lambda_var:
                self.model_states_grid = self.model_states_grid_2d
            else:
                self.model_states_grid = torch.cat((self.model_states_grid_2d, self.dynamics.lambda_target * torch.ones_like(self.model_states_grid_2d[..., 0:1])),-1)
                self.model_coords_grid_allt = torch.cat((self.model_coords_grid_allt, self.dynamics.lambda_target * torch.ones_like(self.model_coords_grid_allt[..., 0:1])),-1)
        
        ## In N dims, define 2D grid on the Main Diagonal (any subsys i is equiv on diag)
        elif self.N > 2:
            
            # TODO: three planes is artifact, using one would be simpler
            score_plane1 = torch.zeros(self.n_grid_pts_2d, self.dynamics.state_dim)
            score_plane2 = torch.zeros(self.n_grid_pts_2d, self.dynamics.state_dim) + 1/300
            score_plane3 = torch.zeros(self.n_grid_pts_2d, self.dynamics.state_dim) + 2/300

            ## (N-1)d Main Diagonal; x0 shared state and N-1 states repeated
            if not hasattr(self.dynamics, "shared_x0") or self.dynamics.shared_x0:
                score_plane1[:, 0] = score_plane1[:, 0] + self.model_states_grid_2d[:, 0]
                score_plane2[:, 0] = score_plane2[:, 0] + self.model_states_grid_2d[:, 0]
                score_plane3[:, 0] = score_plane3[:, 0] + self.model_states_grid_2d[:, 0]

                if not load_lambda_var:
                    xixj = (self.model_states_grid_2d[:, 1] * torch.ones(self.dynamics.state_dim-1, self.n_grid_pts_2d)).t()
                    
                    score_plane1[:, 1:] = score_plane1[:, 1:] + xixj
                    score_plane2[:, 1:] = score_plane2[:, 1:] + xixj
                    score_plane3[:, 1:] = score_plane3[:, 1:] + xixj

                else:
                    xixj = (self.model_states_grid_2d[:, 1] * torch.ones(self.dynamics.N-1, self.n_grid_pts_2d)).t()
                    
                    score_plane1[:, 1:] = score_plane1[:, 1:-1] + xixj
                    score_plane2[:, 1:] = score_plane2[:, 1:-1] + xixj
                    score_plane3[:, 1:] = score_plane3[:, 1:-1] + xixj

                    # Scoring only on specific lambda slice at desired solution
                    score_plane1[:, -1] = self.dynamics.lambda_target * torch.ones(self.n_grid_pts_2d) 
                    score_plane2[:, -1] = self.dynamics.lambda_target * torch.ones(self.n_grid_pts_2d)
                    score_plane3[:, -1] = self.dynamics.lambda_target * torch.ones(self.n_grid_pts_2d)
            
            ## Nd Main Diagonal; Nh pairs of repeated states
            else:
                xixj = self.model_states_grid_2d.repeat(1, self.dynamics.Nh)
            
                if not load_lambda_var:
                    score_plane1 = score_plane1 + xixj
                    score_plane2 = score_plane2 + xixj
                    score_plane3 = score_plane3 + xixj

                else:
                    score_plane1[:, :-1] = score_plane1[:, :-1] + xixj
                    score_plane2[:, :-1] = score_plane2[:, :-1] + xixj
                    score_plane3[:, :-1] = score_plane3[:, :-1] + xixj

                    # lambda = 1 (scoring only NL approx.)
                    score_plane1[:, -1] = self.dynamics.lambda_target * torch.ones(self.n_grid_pts_2d) 
                    score_plane2[:, -1] = self.dynamics.lambda_target * torch.ones(self.n_grid_pts_2d)
                    score_plane3[:, -1] = self.dynamics.lambda_target * torch.ones(self.n_grid_pts_2d)

            self.model_states_grid = torch.cat((score_plane1, score_plane2, score_plane3), dim=0)
            self.n_grid_pts = 3 * self.n_grid_pts_2d
            self.model_states_grid_one_plane = score_plane1

            times = torch.full((self.n_grid_pts, 1), self.tMin) # TODO: remove first time-point if model='exact'
            self.model_coords_grid_allt = torch.cat((times, self.model_states_grid), dim=1) 
            # self.model_coords_grid_allt_hi = torch.cat((times, self.model_states_grid), dim=1) 

            for i in range(self.n_grid_t_pts-1):
                times = torch.full((self.n_grid_pts, 1), (i+1)*(self.tMax - self.tMin)/(self.n_grid_t_pts-1))
                new_coords = torch.cat((times, self.model_states_grid), dim=1) 
                self.model_coords_grid_allt = torch.cat((self.model_coords_grid_allt, new_coords), dim=0) 
            # for i in range(self.n_grid_t_pts_hi-1):
            #     times = torch.full((self.n_grid_pts, 1), (i+1)*(self.tMax - self.tMin)/(self.n_grid_t_pts_hi-1))
            #     new_coords = torch.cat((times, self.model_states_grid), dim=1) 
            #     self.model_coords_grid_allt_hi = torch.cat((self.model_coords_grid_allt_hi, new_coords), dim=0) 

        ## Precompute value, gradient & safe-set on grid for ground truth

        if self.solve_grad:
            self.values_DP_grid, self.value_grads_DP_grid = self.V_DP_grad(self.dynamics.input_to_coord(self.model_coords_grid_allt).t())
            self.values_DP_grid, self.value_grads_DP_grid = self.values_DP_grid.cuda(), self.value_grads_DP_grid.cuda()
        else:
            self.values_DP_grid = self.V_DP(self.dynamics.input_to_coord(self.model_coords_grid_allt).t()).cuda()

        self.values_DP_grid_sub0_ixs = torch.argwhere(self.values_DP_grid <= 0).flatten().cuda()

        if self.lambda_var and self.dynamics.name == "LessLinear":
            self.values_DP_linear_grid = self.V_DP_linear(self.dynamics.input_to_coord(self.model_coords_grid_allt).t()).cuda()
            self.values_DP_grid_inlam1 = self.V_DP_inlam1(self.dynamics.input_to_coord(self.model_coords_grid_allt).t()).cuda()
            self.values_DP_grid_inlam2 = self.V_DP_inlam2(self.dynamics.input_to_coord(self.model_coords_grid_allt).t()).cuda()
        
        elif self.lambda_var and self.dynamics.name in ["Conveyor", "Canoe"]:
            self.values_DP_1_grid = self.V_DP_1(self.dynamics.input_to_coord(self.model_coords_grid_allt).t()).cuda()
            self.values_DP_2_grid = self.V_DP_2(self.dynamics.input_to_coord(self.model_coords_grid_allt).t()).cuda()

        # self.values_DP_grid_hi = self.V_DP(self.dynamics.input_to_coord(self.model_coords_grid_allt_hi).t()).cuda()
        # self.values_DP_grid_sub0_ixs_hi = torch.argwhere(self.values_DP_grid_hi <= 0).flatten().cuda()

        self.model_coords_grid_allt = self.model_coords_grid_allt.cuda()
        # self.model_coords_grid_allt_hi = self.model_coords_grid_allt_hi.cuda()
        self.model_states_grid = self.model_states_grid.cuda()

        ## Isolated loading test
        # if not self.lambda_var:
        #     test_times = torch.full((self.n_grid_pts_2d, 1), 2.) if self.dynamics.N>2 else torch.full((self.n_grid_pts_2d, 1), 2.)
        #     test_states_grid = score_plane1 if self.dynamics.N>2 else self.model_states_grid_2d
        #     self.interp_check(test_times, test_states_grid, grid_L, grid_params, solution_grid)

    def interp_check(self, test_times, test_states_grid, grid_L, grid_params, solution_grid, save_plot=False):
        
        if not hasattr(self.dynamics, "name") or self.dynamics.name not in ["Conveyor","Canoe"]:
            raise NotImplementedError

        # test_coords_grid = torch.cat((test_times, test_states_grid.cpu()), dim=1) # for ITP testing
        test_coords_grid = torch.cat((0 * test_times, test_states_grid.cpu()), dim=1) # for BC testing

        ## Solve Interpolated Values on Main Diagonal Grid
        n_grid_len = grid_L
        plot_values_V_DP = self.V_DP(self.dynamics.input_to_coord(test_coords_grid).t()).reshape(n_grid_len, n_grid_len)
        plot_values_V_DP_1 = self.V_DP_1(self.dynamics.input_to_coord(test_coords_grid).t()).reshape(n_grid_len, n_grid_len)
        plot_values_V_DP_2 = self.V_DP_2(self.dynamics.input_to_coord(test_coords_grid).t()).reshape(n_grid_len, n_grid_len)

        ## Solve BC on Main Diagonal
        test_states_grid_scaled = self.dynamics.input_to_coord(test_coords_grid)[:, 1:]
        plot_values_bc_t0 = self.dynamics.boundary_fn(test_states_grid_scaled, 0*test_times).reshape(n_grid_len, n_grid_len)
        plot_values_bc_t2 = self.dynamics.boundary_fn(test_states_grid_scaled, test_times).reshape(n_grid_len, n_grid_len)
        
        if self.dynamics.name == "Conveyor":
            plot_values_bc1_t0 = self.dynamics.reach_fn(test_states_grid_scaled, 0*test_times).reshape(n_grid_len, n_grid_len)
            plot_values_bc2_t0 = -self.dynamics.avoid_fn(test_states_grid_scaled, 0*test_times).reshape(n_grid_len, n_grid_len)
            plot_values_bc1_t2 = self.dynamics.reach_fn(test_states_grid_scaled, test_times).reshape(n_grid_len, n_grid_len)
            plot_values_bc2_t2 = -self.dynamics.avoid_fn(test_states_grid_scaled, test_times).reshape(n_grid_len, n_grid_len)
        
        elif self.dynamics.name == "Canoe":
            plot_values_bc1_t0 = self.dynamics.reach_fn_1(test_states_grid_scaled, 0*test_times).reshape(n_grid_len, n_grid_len)
            plot_values_bc2_t0 = self.dynamics.reach_fn_2(test_states_grid_scaled, 0*test_times).reshape(n_grid_len, n_grid_len)
            plot_values_bc1_t2 = self.dynamics.reach_fn_1(test_states_grid_scaled, test_times).reshape(n_grid_len, n_grid_len)
            plot_values_bc2_t2 = self.dynamics.reach_fn_2(test_states_grid_scaled, test_times).reshape(n_grid_len, n_grid_len)

        cmap_name = "RdBu_r"
        # vmin, vmax = -0.075, 0.075
        vmin, vmax = -0.5, 0.5
        # vmin, vmax = -2., 2.
        levels = np.linspace(vmin, vmax)
        n_bins_high = round(256 * vmax/(vmax - vmin))
        scaled_colors = np.vstack((mpl.colormaps[cmap_name](np.linspace(0., 0.4, 256-n_bins_high)), mpl.colormaps[cmap_name](np.linspace(0.6, 1., n_bins_high))))
        RdWhBl_vscaled = mpl.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', scaled_colors)
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))
        fig.suptitle(f"V DP - Ground Truth Test")

        xlims = (grid_params["lbs"][0], grid_params["ubs"][0])
        ylims = (grid_params["lbs"][1], grid_params["ubs"][1])

        names = ["V2", "V", "V1"]
        # plot_values_raw = [self.V_DP_2d_2[-1].T, self.V_DP_2d[-1].T, self.V_DP_2d_1[-1].T] # for ITP
        plot_values_raw = [self.V_DP_2d_2[0].T, self.V_DP_2d[0].T, self.V_DP_2d_1[0].T] # for BC
        plot_values_itp = [plot_values_V_DP_2, plot_values_V_DP, plot_values_V_DP_1]

        plot_values_bcs_t0 = [plot_values_bc2_t0, plot_values_bc_t0, plot_values_bc1_t0] # for BC
        plot_values_bcs_t2 = [plot_values_bc2_t2, plot_values_bc_t2, plot_values_bc1_t2] # for BC

        for k in range(3):

            # Plot raw values for ITP
            axes[0][k].set_title(f"{names[k]} raw")
            axes[0][k].contourf(solution_grid.coordinate_vectors[0], solution_grid.coordinate_vectors[1],
                            plot_values_raw[k],
                            levels=levels,
                            extend="both",
                            cmap=RdWhBl_vscaled)
            axes[0][k].contour(solution_grid.coordinate_vectors[0], solution_grid.coordinate_vectors[1],
                            plot_values_raw[k], levels=0, colors="black", linewidths=2)
            axes[0][k].set_xlim(xlims)
            axes[0][k].set_ylim(ylims)
            axes[0][k].set_aspect('equal')

            # Plot interpolation
            axes[1][k].set_title(f"{names[k]} itp")
            axes[1][k].contourf(self.X1g * self.dynamics.state_var[0] + self.dynamics.state_mean[0], self.X2g * self.dynamics.state_var[1] + self.dynamics.state_mean[1],
                            plot_values_itp[k],
                            levels=levels,
                            extend="both",
                            cmap=RdWhBl_vscaled)
            axes[1][k].contour(self.X1g * self.dynamics.state_var[0] + self.dynamics.state_mean[0], self.X2g * self.dynamics.state_var[1] + self.dynamics.state_mean[1],
                            plot_values_itp[k], levels=0, colors="black", linewidths=2)
            axes[1][k].set_xlim(xlims)
            axes[1][k].set_ylim(ylims)
            axes[1][k].set_aspect('equal')

            axes[1][k].contour(self.X1g * self.dynamics.state_var[0] + self.dynamics.state_mean[0], self.X2g * self.dynamics.state_var[1] + self.dynamics.state_mean[1],
                            plot_values_bcs_t0[k], levels=0, colors="magenta", linewidths=2)

            # Plot dynamics bc
            axes[2][k].set_title(f"{names[k]} BC")
            axes[2][k].contourf(self.X1g * self.dynamics.state_var[0] + self.dynamics.state_mean[0], self.X2g * self.dynamics.state_var[1] + self.dynamics.state_mean[1],
                            plot_values_bcs_t2[k],
                            levels=levels,
                            extend="both",
                            cmap=RdWhBl_vscaled)
            axes[2][k].contour(self.X1g * self.dynamics.state_var[0] + self.dynamics.state_mean[0], self.X2g * self.dynamics.state_var[1] + self.dynamics.state_mean[1],
                            plot_values_bcs_t2[k], levels=0, colors="black", linewidths=2)
            axes[2][k].set_xlim(xlims)
            axes[2][k].set_ylim(ylims)
            axes[2][k].set_aspect('equal')

            # TODO: test reach / avoid bc fns here, in N-dimensions (could also check N-dim itp combo works)
            # TODO: then write vanilla BRAAT & BRRT loss fn's, & test w/ vanilla
            # TODO: then write supervision BRAAT & BRRT loss fn's, & test w/ vanilla
            # TODO: then implement various models, and test w/ DR

        plt.savefig(f"plots/mulob_tests/{self.dynamics.name}_test_bc_plot_t2check.png")

        print(f"Mean Error for bc 1: {(plot_values_bc1_t0 - plot_values_V_DP_1).abs().mean():2.0e}")
        print(f"Mean Error for bc 2: {(plot_values_bc2_t0 - plot_values_V_DP_2).abs().mean():2.0e}")
        print(f"Mean Error for bc: {(plot_values_bc_t0 - plot_values_V_DP).abs().mean():2.0e}")
        plt.close()
        return
    
    def make_DP_bank(self):
        
        print("\nMaking a Static Bank of Interpolated Points ...")
        bank = torch.zeros(self.bank_total, 2*(self.dynamics.state_dim)+3) # cols: time (1), state (2 - N+1), boundary value (N+2), value (N+3), spatial grad (N+4 - end)
        bank[:, 1:self.dynamics.state_dim+1] = torch.zeros(self.bank_total, self.dynamics.state_dim).uniform_(-1, 1) 
        # TODO better sampling: latin hypercube? sparse grid? near boundary? uncertainty model?

        step = self.numpoints 
        with tqdm(total=self.numblocks) as pbar:
            for i in range(0, self.bank_total, step):

                # Make T & X (model_coords)
                bank[i:i+step, 0:self.dynamics.state_dim+1] = torch.cat((torch.full((step, 1), (i//step)*(self.tMax-self.tMin)/(self.numblocks-1)),
                                                            bank[i:i+step, 1:self.dynamics.state_dim+1]), dim=1)
                
                # Solve Boundary & Hopf Value 
                coords_io = self.dynamics.input_to_coord(bank[i:i+step, 0:self.dynamics.state_dim+1])
                states_io, times_io = coords_io[..., 1:], coords_io[..., 0:1]
                bank[i:i+step, self.dynamics.state_dim+1] = self.dynamics.boundary_fn(states_io, times_io)
                if self.solve_grad:
                    bank[i:i+step, self.dynamics.state_dim+2], bank[i:i+step, self.dynamics.state_dim+3:] = self.V_hopf_grad(self.dynamics.input_to_coord(bank[i:i+step, 0:self.dynamics.state_dim+1]).t())
                else:
                    bank[i:i+step, self.dynamics.state_dim+2] = self.V_hopf(self.dynamics.input_to_coord(bank[i:i+step, 0:self.dynamics.state_dim+1]).t())

                pbar.update(1)

        ## Save Evaluated Bank and Delete it from Memory
        print("Done. Written to " + self.bank_name + ".\n")
        np.save(self.llnd_path + "banks/" + self.bank_name, bank)
        del(bank)
        gc.collect()

    def make_hopf_bank(self):
        print("\nMaking a Dynamic Bank of Hopf Points ...")     

        ## Solve bank starter (blocks until completion)
        self.hjpool.solve_bank_starter(self.hopf_bank_params, n_splits=self.hopf_starter_numsplits, print_sample=False)
        self.solved_hopf_pts = self.hopf_bank_params["n_starter"]
        self.max_solved_ix = min(self.solved_hopf_pts, self.hopf_bank_params["n_total"])

        ## Refine Bank 
        # (this is inefficient unless we have synergy, otherwise TODO hopf solve samples in curriculum-like fashion)
        if self.refine_bank:
            if self.hopf_bank_params["n_total"] != self.hopf_bank_params["n_starter"] and self.hopf_bank_params["n_total"] != self.hopf_bank_params["n_deposit"]: raise AssertionError("Bank refinement needs equivalent bank params")
            
            self.shm_states = SharedMemory(name=self.hjpool.shm_states_id)
            self.bank = np.ndarray(self.hjpool.shm_states_shape, dtype=np.float32, buffer=self.shm_states.buf)

            self.refined_bank = np.zeros(self.hjpool.shm_states_shape, dtype=np.float32)
            n_spatial = int(self.bank_total * self.hopf_time_step)
            n_spatial_split = int(n_spatial / self.hopf_starter_numsplits)
            n_split_size = int(self.bank_total / self.hopf_starter_numsplits)
            n_tp = int(1/self.hopf_time_step)
            c_refines = 0

            for j in range(self.hopf_starter_numsplits):
                for i in range(n_spatial_split):
                    bix = j * n_split_size + i + np.random.randint(n_tp) * n_spatial_split
                    rbix = c_refines * n_spatial + j * n_spatial_split + i
                    self.refined_bank[rbix, :] = self.bank[bix, :]
            c_refines += 1

            for k in range(n_tp-1):
                refine_start = time.time()
                self.hjpool.solve_bank_deposit(model=None, n_splits=self.hopf_deposit_numsplits, blocking=True) ## BUG: concise causes segf...?
                for j in range(self.hopf_starter_numsplits):
                    for i in range(n_spatial_split):
                        bix = j * n_split_size + i + np.random.randint(n_tp) * n_spatial_split
                        rbix = c_refines * n_spatial + j * n_spatial_split + i
                        self.refined_bank[rbix, :] = self.bank[bix, :]
                c_refines += 1
                refine_time = (time.time() - refine_start)
                print(f"Completed ({c_refines}/{n_tp}), est. time left: {refine_time*(n_tp  - c_refines)/60:<4.1f} mins")
    
    def load_bank(self):

        if not self.solve_hopf:
            self.bank = np.load(self.llnd_path + "banks/" + self.bank_name, mmap_mode='r')
        
        else:
            self.shm_states = SharedMemory(name=self.hjpool.shm_states_id)
            self.shm_algdat = SharedMemory(name=self.hjpool.shm_algdat_id)
            self.bank = np.ndarray(self.hjpool.shm_states_shape, dtype=np.float32, buffer=self.shm_states.buf)
            self.alg_data = np.ndarray(self.hjpool.shm_algdat_shape, dtype=np.float32, buffer=self.shm_algdat.buf)

            if self.just_make_hopf_bank:
                print("Done. Written to " + self.bank_name + ".\n")
                if self.refine_bank:
                    np.save(self.llnd_path + "banks/" + self.bank_name, self.refined_bank)
                else:
                    np.save(self.llnd_path + "banks/" + self.bank_name, self.bank)
                np.save(self.llnd_path + "banks/alg_data_for_" + self.bank_name, self.alg_data)
                self.hjpool.dispose()
                sys.exit()
        
        if not self.solve_hopf or self.hopf_bank_params["n_starter"] == self.hopf_bank_params["n_total"]:
            self.bank_index = torch.from_numpy(np.random.permutation(self.bank_total)) # random shuffle for sample bank
            self.block_counter = 0

    def make_bank_name(self):
        if self.bank_total >= 1e6:
            qty_tag = str(int(self.bank_total // 1e6)) + 'M'
        elif self.bank_total >= 1e3:
            qty_tag = str(int(self.bank_total // 1e3)) + 'K'
        else:
            qty_tag = str(self.bank_total)

        if self.solve_hopf:
            make_tag = "Hopf_mp_" 
            if self.refine_bank:
                make_tag = make_tag + "refined_"
        else:
            make_tag = "DPitp_"
        return "Bank_" + make_tag + str(self.N)+"D_"+ qty_tag + "pts_r" + str(int(100 * self.dynamics.goalR_2d)) + "e-2_g" + str(int(self.dynamics.gamma)) + "m" + str(int(self.dynamics.mu)) + "a" + str(int(self.dynamics.alpha)) + ".npy"
