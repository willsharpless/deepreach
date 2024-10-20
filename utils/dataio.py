from juliacall import Main as jl, convert as jlconvert
import torch
import numpy as np
from torch.utils.data import Dataset
import time
import os, sys
import gc
from tqdm.autonotebook import tqdm
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import julia_multiproc

# uses model input and real boundary fn
class ReachabilityDataset(Dataset):
    def __init__(self, dynamics, numpoints, pretrain, pretrain_iters, tMin, tMax, counter_start, counter_end, num_src_samples, num_target_samples, 
                 use_hopf=False, hopf_pretrain=False, hopf_pretrain_iters=0, record_gt_metrics=False, solve_grad=False,
                 dp_manual_load=False, load_packet=None, no_curriculum=False, use_bank=False, bank_name=None, capacity_test=False,
                 solve_hopf=False, hopf_warm_start=False, hopf_time_step=1e-2, num_hopf_workers=10, hopf_starter_numsplits=1000, hopf_deposit_numsplits=1000,
                #  solve_hopf=False, hopf_warm_start=False, hopf_time_step=1e-1, num_hopf_workers=10, hopf_starter_numsplits=200, hopf_deposit_numsplits=200,
                 hopf_opt_p = {"vh":0.01, "stepsz":1, "tol":1e-3, "decay_stepsz":100, "conv_runs_rqd":1, "max_runs":1, "max_its":100},
                #  hopf_bank_params = {"n_total":int(4e6), "n_starter":int(4e6), "n_deposit":int(4e6)}, # to make static bank
                 hopf_bank_params = {"n_total":int(1e5), "n_starter":int(1e5), "n_deposit":int(1e5)}, # dynamic refresh
                #  hopf_bank_params = {"n_total":int(4e6), "n_starter":int(4e6), "n_deposit":int(2e6)}, 
                #  hopf_bank_params = {"n_total":int(1e5), "n_starter":int(1e5), "n_deposit":int(1e3)}, 
                 hopf_bank_save_quit=False, refine_bank=False,
                 loaded_model=None,
                 ):

        self.dynamics = dynamics
        self.numpoints = numpoints
        self.pretrain = pretrain
        self.pretrain_counter = 0
        self.pretrain_iters = pretrain_iters
        self.tMin = tMin 
        self.tMax = tMax 
        self.counter = counter_start 
        self.counter_end = counter_end 
        self.num_src_samples = num_src_samples
        self.num_target_samples = num_target_samples

        self.use_hopf = use_hopf # FIXME: old name, means use linear data (not necessarily solve hopf formula)
        self.solve_grad = solve_grad
        self.hopf_pretrain = use_hopf and hopf_pretrain
        self.hopf_pretrained = use_hopf and hopf_pretrain
        self.hopf_pretrain_counter = 0
        self.hopf_pretrain_iters = hopf_pretrain_iters
        self.record_gt_metrics = record_gt_metrics
        self.no_curriculum = no_curriculum
        self.N = dynamics.N
        self.capacity_test = capacity_test
        self.llnd_path = "value_fns/LessLinear/"

        self.solve_hopf = solve_hopf # triggers online hopf formula solving via HopfReachability.jl
        self.num_hopf_workers = num_hopf_workers
        self.hopf_warm_start = hopf_warm_start
        self.hopf_time_step = hopf_time_step
        self.hopf_opt_p = hopf_opt_p
        self.hopf_bank_params = hopf_bank_params
        self.hopf_starter_numsplits = hopf_starter_numsplits
        self.hopf_deposit_numsplits = hopf_deposit_numsplits
        # self.hopf_time_step = 1e-3 #FIXME load from run_exp.py  
        # self.hopf_opt_p = {"vh":0.01, "stepsz":1, "tol":1e-3, "decay_stepsz":100, "conv_runs_rqd":1, "max_runs":1, "max_its":100} #TODO load from run_exp.py 
        # self.hopf_bank_params = {"n_total":2e6, "n_starter":1e6, "n_deposit":1e5} # TODO load from run_exp.py  
        # self.hopf_bank_params = {"n_total":12, "n_starter":8, "n_deposit":2} # small test set
        # self.hopf_time_step = 1e-2
        # self.hopf_bank_params = {"n_total":4000, "n_starter":1000, "n_deposit":1000} # medium test set
        # self.hopf_bank_params = {"n_total":40000, "n_starter":1000, "n_deposit":1000} # big test set
        # self.hopf_starter_numsplits = 10
        # self.hopf_deposit_numsplits = 10
        
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

        if self.bank_total >= 1e6:
            qty_tag = str(int(self.bank_total // 1e6)) + 'M'
        elif self.bank_total >= 1e3:
            qty_tag = str(int(self.bank_total // 1e3)) + 'K'
        else:
            qty_tag = str(self.bank_total)

        if bank_name is None or bank_name == 'none': 
            if self.solve_hopf:
                make_tag = "Hopf_mp_" 
                if self.refine_bank:
                    make_tag = make_tag + "refined_"
            else:
                make_tag = "DPitp_"
            bank_name = "Bank_" + make_tag + str(self.N)+"D_"+ qty_tag + "pts_r" + str(int(100 * dynamics.goalR_2d)) + "e-2_g" + str(int(dynamics.gamma)) + "m" + str(int(dynamics.mu)) + "a" + str(int(dynamics.alpha)) + ".npy"
        self.bank_name = bank_name
        self.mp_bank = "mp" in bank_name
        self.hopf_bank_save_quit = hopf_bank_save_quit

        # Dynamic Programming Manual Load (added this to skirt WandB sweep + PyCall imcompatibility but still not working)
        if dp_manual_load: 
            self.V_hopf_itp, self.fast_interp, self.V_hopf, self.V_DP_itp, self.V_DP = load_packet

        # Load a pretrained model for hopf supervision
        self.loaded_model = loaded_model
        if loaded_model: self.loaded_model = loaded_model.cuda()
        self.load_hopf_model = loaded_model is not None
        
        ## Compute Linear Value from Model (if hopf loss)
        if use_hopf and not(dp_manual_load):

            if self.load_hopf_model:
                pass # done

            ## Load Linear DP Solution for Proof-of-Concept (Faster & Higher-Fidelity than actual Hopf Solving)
            elif not self.solve_hopf:
                
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
                    self.V_hopf_itp = jl.load("lin2d_hopf_interp_linear.jld")["V_itp"] ## (using gt)
                    self.V_hopf = lambda tXg: torch.from_numpy(self.fast_interp(self.V_hopf_itp, tXg.numpy()).to_numpy())
                    if self.solve_grad:
                        self.V_hopf_grad = lambda tXg: torch.from_numpy(self.fast_interp(self.V_hopf_itp, tXg.numpy(), compute_grad=True).to_numpy())
                
                elif self.N > 2:
                    
                    ## Load 2D DP Solution
                    # LessLinear2D_interpolations = jl.load(self.llnd_path + "interps/old/LessLinear2D1i_interpolations_res1e-2_r15e-2.jld", "LessLinear2D_interpolations")
                    LessLinear2D_interpolations = jl.load(self.llnd_path + "interps/LessLinear2D1i_interpolations_res1e-2_r15e-2_c5.jld", "LessLinear2D_interpolations")
                    self.V_hopf_itp = LessLinear2D_interpolations["g0_m0_a0"] ## (using gt)
                    
                    ## Capacity Test: Load Ground Truth for Supervised-Learning
                    if capacity_test:
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
            
            ## Initialize Julia CPU Workers for Pool Solving Hopf Formula
            else:
                mp.set_start_method('spawn', force=True)
                if self.solve_hopf and not self.record_gt_metrics: AssertionError("Must also interpolate DP solution if solving hopf. Ask Will to fix me.") #BUG: x/0 bug when only this on?
                self.hjpool = julia_multiproc.HopfJuliaPool(self.dynamics, self.hopf_time_step, self.hopf_opt_p,  
                                                    use_hopf=True, # if (manually) False, then using DP itp
                                                    solve_grad=self.solve_grad,
                                                    hopf_warm_start=self.hopf_warm_start,
                                                    gt_metrics=self.record_gt_metrics,
                                                    num_hopf_workers=self.num_hopf_workers)      
                
        ## Get Ground Truth for Special N-Dimensional Decomposable LessLinear System
        # TODO put this into a fn jesus
        if record_gt_metrics:
            if not(dp_manual_load):

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
                
                if self.N == 2:
                    # self.V_DP_itp = jl.load(self.llnd_path + "interps/old/llin2d_g20_m-20_a1_DP_interp_linear.jld")["V_itp"]
                    model_key = "g" + str(int(self.dynamics.gamma)) + "_m" + str(int(self.dynamics.mu)) + "_a"  + str(int(self.dynamics.alpha))
                    self.V_DP_itp = jl.load(self.llnd_path + "interps/LessLinear2D1i_interpolations_res1e-2_r15e-2_c20.jld", "LessLinear2D_interpolations")[model_key]
                    self.V_DP = lambda tXg: torch.from_numpy(self.fast_interp(self.V_DP_itp, tXg.numpy()).to_numpy())
                    if self.solve_grad:
                        self.V_DP_grad = lambda tXg: torch.from_numpy(self.fast_interp(self.V_DP_itp, tXg.numpy(), compute_grad=True).to_numpy())

                elif self.N > 2:
                    # LessLinear2D_interpolations = jl.load(self.llnd_path + "interps/old/LessLinear2D1i_interpolations_res1e-2_r15e-2.jld", "LessLinear2D_interpolations")
                    LessLinear2D_interpolations = jl.load(self.llnd_path + "interps/LessLinear2D1i_interpolations_res1e-2_r15e-2_c5.jld", "LessLinear2D_interpolations")
                    
                    model_key = "g" + str(int(self.dynamics.gamma)) + "_m" + str(int(self.dynamics.mu)) + "_a"  + str(int(self.dynamics.alpha))
                    self.V_DP_itp = LessLinear2D_interpolations[model_key]
                    def V_N_DP_itp_combo(tXg):
                        V = 0 * tXg[0,:]
                        for i in range(self.N-1):
                            V += torch.from_numpy(self.fast_interp(self.V_DP_itp, tXg[[0, 1, 2+i], :].numpy()).to_numpy())
                        return V
                    self.V_DP = V_N_DP_itp_combo

                    self.V_DP_linear_itp = LessLinear2D_interpolations["g0_m0_a0"]
                    def V_N_DP_linear_itp_combo(tXg):
                        V = 0 * tXg[0,:]
                        for i in range(self.N-1):
                            V += torch.from_numpy(self.fast_interp(self.V_DP_linear_itp, tXg[[0, 1, 2+i], :].numpy()).to_numpy())
                        return V
                    self.V_DP_linear = V_N_DP_linear_itp_combo # for plotting the BRT of Linear Solution

                    if self.solve_grad:
                        def V_N_DP_itp_grad_combo(tXg):
                            V = 0 * tXg[0,:]
                            DV = 0 * tXg[1:,:].t()
                            for i in range(self.N-1):
                                Vi, DVi = self.fast_interp(self.V_DP_itp, tXg[[0, 1, 2+i], :].numpy(), compute_grad=True)
                                V += torch.from_numpy(Vi.to_numpy())
                                DV[:, [0, 1+i]] += torch.from_numpy(DVi.to_numpy()) # assumes xN first
                            return V, DV
                        self.V_DP_grad = V_N_DP_itp_grad_combo

            ## Define a fixed spatiotemporal grid to score Jaccard

            self.n_grid_t_pts, self.n_grid_t_pts_hi = 5, 20
            xig = torch.arange(-0.99, 1.01, 0.02) # 100 x 100
            # xig = torch.arange(-1., 1.01, 0.01) # 201 x 201
            self.X1g, self.X2g = torch.meshgrid(xig, xig)
            self.model_states_grid = torch.cat((self.X1g.ravel().reshape((1,xig.size()[0]**2)), self.X2g.ravel().reshape((1,xig.size()[0]**2))), dim=0).t()
            self.n_grid_pts = xig.size()[0]**2

            ## Make a low and high res grids wrt time
            if self.N == 2:

                times = torch.full((self.n_grid_pts, 1), self.tMin) # TODO: remove first time-point if model='exact'
                self.model_coords_grid_allt = torch.cat((times, self.model_states_grid), dim=1) 
                self.model_coords_grid_allt_hi = torch.cat((times, self.model_states_grid), dim=1) 

                for i in range(self.n_grid_t_pts-1):
                    times = torch.full((self.n_grid_pts, 1), (i+1)*(self.tMax - self.tMin)/(self.n_grid_t_pts-1))
                    new_coords = torch.cat((times, self.model_states_grid), dim=1) 
                    self.model_coords_grid_allt = torch.cat((self.model_coords_grid_allt, new_coords), dim=0) 
                for i in range(self.n_grid_t_pts_hi-1):
                    times = torch.full((self.n_grid_pts, 1), (i+1)*(self.tMax - self.tMin)/(self.n_grid_t_pts_hi-1))
                    new_coords = torch.cat((times, self.model_states_grid), dim=1) 
                    self.model_coords_grid_allt_hi = torch.cat((self.model_coords_grid_allt_hi, new_coords), dim=0) 
            
            ## In N dimension, using same 2D grid on 3 slices of total space
            elif self.N > 2:

                xnxi_plane = torch.zeros(self.n_grid_pts, self.N)
                xnxi_plane[:, 0] = self.model_states_grid[:, 0]
                xnxi_plane[:, 1] = self.model_states_grid[:, 1]
                xixj_plane = torch.zeros(self.n_grid_pts, self.N)
                xixj_plane[:, 1] = self.model_states_grid[:, 0]
                xixj_plane[:, 2] = self.model_states_grid[:, 1]
                xnxixj_plane = torch.zeros(self.n_grid_pts, self.N)
                xnxixj_plane[:, 0] = self.model_states_grid[:, 0]
                xnxixj_plane[:, 1:] = (self.model_states_grid[:, 1]* torch.ones(self.N-1, self.n_grid_pts)).t()

                self.model_states_grid = torch.cat((xnxi_plane, xixj_plane, xnxixj_plane), dim=0)
                self.n_grid_pts = 3 * self.n_grid_pts

                times = torch.full((self.n_grid_pts, 1), self.tMin) # TODO: remove first time-point if model='exact'
                self.model_coords_grid_allt = torch.cat((times, self.model_states_grid), dim=1) 
                self.model_coords_grid_allt_hi = torch.cat((times, self.model_states_grid), dim=1) 

                for i in range(self.n_grid_t_pts-1):
                    times = torch.full((self.n_grid_pts, 1), (i+1)*(self.tMax - self.tMin)/(self.n_grid_t_pts-1))
                    new_coords = torch.cat((times, self.model_states_grid), dim=1) 
                    self.model_coords_grid_allt = torch.cat((self.model_coords_grid_allt, new_coords), dim=0) 
                for i in range(self.n_grid_t_pts_hi-1):
                    times = torch.full((self.n_grid_pts, 1), (i+1)*(self.tMax - self.tMin)/(self.n_grid_t_pts_hi-1))
                    new_coords = torch.cat((times, self.model_states_grid), dim=1) 
                    self.model_coords_grid_allt_hi = torch.cat((self.model_coords_grid_allt_hi, new_coords), dim=0) 

            ## Precompute value & safe-set on grid for ground truth

            if self.solve_grad:
                self.values_DP_grid, self.value_grads_DP_grid = self.V_DP_grad(self.dynamics.input_to_coord(self.model_coords_grid_allt).t())
                self.values_DP_grid, self.value_grads_DP_grid = self.values_DP_grid.cuda(), self.value_grads_DP_grid.cuda()
            else:
                self.values_DP_grid = self.V_DP(self.dynamics.input_to_coord(self.model_coords_grid_allt).t()).cuda()

            self.values_DP_linear_grid = self.V_DP_linear(self.dynamics.input_to_coord(self.model_coords_grid_allt).t()).cuda()
            self.values_DP_grid_sub0_ixs = torch.argwhere(self.values_DP_grid <= 0).flatten().cuda()

            self.values_DP_grid_hi = self.V_DP(self.dynamics.input_to_coord(self.model_coords_grid_allt_hi).t()).cuda()
            self.values_DP_grid_sub0_ixs_hi = torch.argwhere(self.values_DP_grid_hi <= 0).flatten().cuda()

            self.model_coords_grid_allt = self.model_coords_grid_allt.cuda()
            self.model_coords_grid_allt_hi = self.model_coords_grid_allt_hi.cuda()
            self.model_states_grid = self.model_states_grid.cuda()

        ## Make a bank of evaluated points, instead of evaluating online
        if self.make_bank:

            ## Make a Bank of Points from Linear DP Solution
            if not self.solve_hopf:
                print("\nMaking a Static Bank of Interpolated Points ...")
                bank = torch.zeros(self.bank_total, 2*(self.N)+3) # cols: time (1), state (2 - N+1), boundary value (N+2), value (N+3), spatial grad (N+4 - end)
                bank[:, 1:self.N+1] = torch.zeros(self.bank_total, self.N).uniform_(-1, 1) 
                # TODO better sampling: latin hypercube? sparse grid? near boundary? uncertainty model?

                step = self.numpoints 
                with tqdm(total=self.numblocks) as pbar:
                    for i in range(0, self.bank_total, step):

                        # Make T & X (model_coords)
                        bank[i:i+step, 0:self.N+1] = torch.cat((torch.full((step, 1), (i//step)*(self.tMax-self.tMin)/(self.numblocks-1)),
                                                                    bank[i:i+step, 1:self.N+1]), dim=1)
                        
                        # Solve Boundary & Hopf Value 
                        bank[i:i+step, self.N+1] = self.dynamics.boundary_fn(self.dynamics.input_to_coord(bank[i:i+step, 0:self.N+1])[..., 1:])
                        if self.solve_grad:
                            bank[i:i+step, self.N+2], bank[i:i+step, self.N+3:] = self.V_hopf_grad(self.dynamics.input_to_coord(bank[i:i+step, 0:self.N+1]).t())
                        else:
                            bank[i:i+step, self.N+2] = self.V_hopf(self.dynamics.input_to_coord(bank[i:i+step, 0:self.N+1]).t())

                        pbar.update(1)

                ## Save Evaluated Bank and Delete it from Memory
                print("Done. Written to " + self.bank_name + ".\n")
                np.save(self.llnd_path + "banks/" + self.bank_name, bank)
                del(bank)
                gc.collect()
            
            ## Make Dynamic Bank by Solving Hopf Formula Online
            else:
                print("\nMaking a Dynamic Bank of Hopf Points ...")     
                ## Solve bank starter (blocks until completion)
                self.hjpool.solve_bank_starter(self.hopf_bank_params, n_splits=self.hopf_starter_numsplits, print_sample=False)
                self.solved_hopf_pts = self.hopf_bank_params["n_starter"]
                self.max_solved_ix = min(self.solved_hopf_pts, self.hopf_bank_params["n_total"])

                ## Refine Bank 
                # (this is inefficient unless we have synergy, otherwise TODO hopf solve samples in curriculum-like fashion)
                if refine_bank:
                    if hopf_bank_params["n_total"] != hopf_bank_params["n_starter"] and hopf_bank_params["n_total"] != hopf_bank_params["n_deposit"]: raise AssertionError("Bank refinement needs equivalent bank params")
                    
                    self.shm_states = SharedMemory(name=self.hjpool.shm_states_id)
                    self.bank = np.ndarray(self.hjpool.shm_states_shape, dtype=np.float32, buffer=self.shm_states.buf)

                    self.refined_bank = np.zeros(self.hjpool.shm_states_shape, dtype=np.float32)
                    n_spatial = int(self.bank_total * self.hopf_time_step)
                    n_spatial_split = int(n_spatial / hopf_starter_numsplits)
                    n_split_size = int(self.bank_total / hopf_starter_numsplits)
                    n_tp = int(1/self.hopf_time_step)
                    c_refines = 0

                    for j in range(hopf_starter_numsplits):
                        for i in range(n_spatial_split):
                            bix = j * n_split_size + i + np.random.randint(n_tp) * n_spatial_split
                            rbix = c_refines * n_spatial + j * n_spatial_split + i
                            self.refined_bank[rbix, :] = self.bank[bix, :]
                    c_refines += 1

                    for k in range(n_tp-1):
                        refine_start = time.time()
                        self.hjpool.solve_bank_deposit(model=None, n_splits=self.hopf_deposit_numsplits, blocking=True) ## BUG: concise causes segf...?
                        for j in range(hopf_starter_numsplits):
                            for i in range(n_spatial_split):
                                bix = j * n_split_size + i + np.random.randint(n_tp) * n_spatial_split
                                rbix = c_refines * n_spatial + j * n_spatial_split + i
                                self.refined_bank[rbix, :] = self.bank[bix, :]
                        c_refines += 1
                        refine_time = (time.time() - refine_start)
                        print(f"Completed ({c_refines}/{n_tp}), est. time left: {refine_time*(n_tp  - c_refines)/60:<4.1f} mins")

        ## Load Memory Map of the Bank
        if self.use_bank:
            
            if not self.solve_hopf:
                self.bank = np.load(self.llnd_path + "banks/" + self.bank_name, mmap_mode='r')
            
            else:
                self.shm_states = SharedMemory(name=self.hjpool.shm_states_id)
                self.shm_algdat = SharedMemory(name=self.hjpool.shm_algdat_id)
                self.bank = np.ndarray(self.hjpool.shm_states_shape, dtype=np.float32, buffer=self.shm_states.buf)
                self.alg_data = np.ndarray(self.hjpool.shm_algdat_shape, dtype=np.float32, buffer=self.shm_algdat.buf)

                if self.hopf_bank_save_quit:
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
                    
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        
        ## Sample Points and Evaluate 
        # uniformly sample domain and include coordinates where source is non-zero 
        model_states = torch.zeros(self.numpoints, self.dynamics.state_dim).uniform_(-1, 1)
        if self.num_target_samples > 0:
            target_state_samples = self.dynamics.sample_target_state(self.num_target_samples)
            model_states[-self.num_target_samples:] = self.dynamics.coord_to_input(torch.cat((torch.zeros(self.num_target_samples, 1), target_state_samples), dim=-1))[:, 1:self.dynamics.state_dim+1]

        if self.pretrain:
            # only sample in time around the initial condition
            times = torch.full((self.numpoints, 1), self.tMin)
        else:
            # slowly grow time values from start time (unless Hopf)
            if self.hopf_pretrain or self.hopf_pretrained or self.no_curriculum:
                # times = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * ((self.counter + self.hopf_pretrain_counter)/(self.counter_end + self.hopf_pretrain_iters)))
                times = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin)) # during hopf pt, sample across all time?
            else:
                times = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * (self.counter/self.counter_end))
            # make sure we always have training samples at the initial time
            times[-self.num_src_samples:, 0] = self.tMin

        model_coords = torch.cat((times, model_states), dim=1)        
        if self.dynamics.input_dim > self.dynamics.state_dim + 1: # temporary workaround for having to deal with dynamics classes for parametrized models with extra inputs
            model_coords = torch.cat((model_coords, torch.zeros(self.numpoints, self.dynamics.input_dim - self.dynamics.state_dim - 1)), dim=1)

        ## Get Hopf value
        if self.use_hopf:   

            if self.pretrain or self.load_hopf_model:
                hopf_values = torch.zeros(self.numpoints)
                if self.solve_grad:
                    hopf_grads = torch.zeros(self.numpoints, self.N)

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
                model_coords_hopf = bank_sample[:, 0:self.N+1]
                hopf_values = bank_sample[:, self.N+2]

                if self.solve_grad:
                    if self.solve_hopf or self.mp_bank: ## TODO: fix static bank format to match hopf
                        hopf_grads = bank_sample[:, self.N+4:2*self.N+4]
                    else:
                        hopf_grads = bank_sample[:, self.N+3:]

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
            
        boundary_values = self.dynamics.boundary_fn(self.dynamics.input_to_coord(model_coords)[..., 1:])
        if self.dynamics.loss_type == 'brat_hjivi':
            reach_values = self.dynamics.reach_fn(self.dynamics.input_to_coord(model_coords)[..., 1:])
            avoid_values = self.dynamics.avoid_fn(self.dynamics.input_to_coord(model_coords)[..., 1:])

        if self.pretrain:
            dirichlet_masks = torch.ones(model_coords.shape[0]) > 0
        else:
            # only enforce initial conditions around self.tMin
            dirichlet_masks = (model_coords[:, 0] == self.tMin)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.hopf_pretrain:
            self.hopf_pretrain_counter += 1
        elif self.counter < self.counter_end:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        if self.hopf_pretrain and self.hopf_pretrain_counter == self.hopf_pretrain_iters:
            self.hopf_pretrain = False

        if self.dynamics.loss_type == 'brt_hjivi':
            return {'model_coords': model_coords}, {'boundary_values': boundary_values, 'dirichlet_masks': dirichlet_masks}
        elif self.dynamics.loss_type == 'brt_hjivi_hopf':
            ## UGLY
            if (not(self.use_bank) or self.hopf_pretrain_counter == 0) and not self.solve_grad:
                return {'model_coords': model_coords}, {'boundary_values': boundary_values, 'dirichlet_masks': dirichlet_masks, 'hopf_values': hopf_values}
            elif not(self.use_bank) or self.hopf_pretrain_counter == 0:
                return {'model_coords': model_coords}, {'boundary_values': boundary_values, 'dirichlet_masks': dirichlet_masks, 'hopf_values': hopf_values, 'hopf_grads': hopf_grads}
            elif not self.solve_grad:
                return {'model_coords': model_coords}, {'boundary_values': boundary_values, 'dirichlet_masks': dirichlet_masks, 'hopf_values': hopf_values, 'model_coords_hopf': model_coords_hopf}
            else:
                return {'model_coords': model_coords}, {'boundary_values': boundary_values, 'dirichlet_masks': dirichlet_masks, 'hopf_values': hopf_values, 'model_coords_hopf': model_coords_hopf, 'hopf_grads': hopf_grads}
        elif self.dynamics.loss_type == 'brat_hjivi':
            return {'model_coords': model_coords}, {'boundary_values': boundary_values, 'reach_values': reach_values, 'avoid_values': avoid_values, 'dirichlet_masks': dirichlet_masks}
        else:
            raise NotImplementedError