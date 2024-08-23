from juliacall import Main as jl, convert as jlconvert
import torch
import numpy as np
from torch.utils.data import Dataset
import time
import os
import gc
from tqdm.autonotebook import tqdm

# uses model input and real boundary fn
class ReachabilityDataset(Dataset):
    def __init__(self, dynamics, numpoints, pretrain, pretrain_iters, tMin, tMax, counter_start, counter_end, num_src_samples, num_target_samples, 
                 use_hopf=False, hopf_pretrain=False, hopf_pretrain_iters=0, record_set_metrics=False,
                 manual_load=False, load_packet=None, no_curriculum=False, use_bank=False, bank_name=None, capacity_test=False):
        
        # print("Into the dataset!")

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

        self.use_hopf = use_hopf
        self.hopf_pretrain = use_hopf and hopf_pretrain
        self.hopf_pretrained = use_hopf and hopf_pretrain
        self.hopf_pretrain_counter = 0
        self.hopf_pretrain_iters = hopf_pretrain_iters
        self.record_set_metrics = record_set_metrics
        self.no_curriculum = no_curriculum
        self.N = dynamics.N
        self.capacity_test = capacity_test
        self.llnd_path = "value_fns/LessLinear/"

        self.use_bank = use_bank
        if bank_name is None or bank_name == 'none': bank_name = "Bank_"+str(self.N)+"D_"+str(self.numpoints//10000)+"Mpts.npy"
        self.bank_name = bank_name
        self.make_bank = use_bank and not(os.path.isfile(self.llnd_path + "banks/" + self.bank_name))
        self.numblocks = 101
        self.bank_total = numpoints * self.numblocks

        if manual_load: # added this to skirt WandB sweep + PyCall imcompatibility (not working)
            self.V_hopf_itp, self.fast_interp, self.V_hopf, self.V_DP_itp, self.V_DP = load_packet
        
        ## Compute Hopf value from interpolant (if hopf loss)
        
        if use_hopf and not(manual_load):

            jl.seval("using JLD, JLD2, Interpolations")
            fast_interp_exec = """
            function fast_interp(_V_itp, tXg)
                Vg = zeros(size(tXg,2))
                for i=1:length(Vg); Vg[i] = _V_itp(tXg[:,i][end:-1:1]...); end # (assumes t in first row)
                return Vg
            end
            """
            self.fast_interp = jl.seval(fast_interp_exec)
            
            if self.N == 2:
                self.V_hopf_itp = jl.load("lin2d_hopf_interp_linear.jld")["V_itp"]
                self.V_hopf = lambda tXg: torch.from_numpy(self.fast_interp(self.V_hopf_itp, tXg.numpy()).to_numpy())
            
            ## FIXME : using DP right now for early Ndim testing, switch to hopf solution in future
            elif self.N > 2:

                LessLinear2D_interpolations = jl.load("LessLinear2D1i_interpolations_res1e-2_r15e-2.jld", "LessLinear2D_interpolations")

                self.V_hopf_itp = LessLinear2D_interpolations["g0_m0_a0"]
                if capacity_test:
                    model_key = "g" + str(int(self.dynamics.gamma)) + "_m" + str(int(self.dynamics.mu)) + "_a"  + str(int(self.dynamics.alpha))
                    self.V_hopf_itp = LessLinear2D_interpolations[model_key]

                def V_N_hopf_itp(tXg):
                    V = 0 * tXg[0,:]
                    for i in range(self.N-1):
                        V += torch.from_numpy(self.fast_interp(self.V_hopf_itp, tXg[[0, 1, 2+i], :].numpy()).to_numpy())
                    return V
                self.V_hopf = V_N_hopf_itp ## TODO: actually use hopf solution (as in N==2 case)

                # fast_interp_exec = """
                # function fast_interp_N(_V_itp, tXg)
                #     Vg = zeros(size(tXg,2))
                #     for i=1:length(Vg); 
                #     Vg[i] = sum(_V_itp(tXg[[1,2,2+j],i][end:-1:1]...) for j=1:size(tXg,1)-3); end # (assumes t in first row)
                #     return Vg
                # end
                # """
                # self.fast_interp = jl.seval(fast_interp_exec)
                # self.V_hopf = lambda tXg: torch.from_numpy(self.fast_interp(self.V_hopf_itp, tXg.numpy()).to_numpy())
                
        if record_set_metrics:
            if not(manual_load):

                jl.seval("using JLD, JLD2, Interpolations")
                fast_interp_exec = """
                function fast_interp(_V_itp, tXg)
                    Vg = zeros(size(tXg,2))
                    for i=1:length(Vg); Vg[i] = _V_itp(tXg[:,i][end:-1:1]...); end # (assumes t in first row)
                    return Vg
                end
                """
                if not(hasattr(self, 'fast_interp')):
                    self.fast_interp = jl.seval(fast_interp_exec)
                
                if self.N == 2:
                    self.V_DP_itp = jl.load(self.llnd_path + "interps/old/llin2d_g20_m-20_a1_DP_interp_linear.jld")["V_itp"]
                    self.V_DP = lambda tXg: torch.from_numpy(self.fast_interp(self.V_DP_itp, tXg.numpy()).to_numpy())
                
                elif self.N > 2:
                    LessLinear2D_interpolations = jl.load(self.llnd_path + "interps/old/LessLinear2D1i_interpolations_res1e-2_r15e-2.jld", "LessLinear2D_interpolations")
                    # LessLinear2D_interpolations = jl.load(rel_path + "old/LessLinear2D1i_interpolations_res1e-2_r4e-1_el_1_5.jld", "LessLinear2D_interpolations")
                    # self.V_DP_itp = LessLinear2D_interpolations["g0_m0_a0"] #linear
                    # self.V_DP_itp = LessLinear2D_interpolations["g20_m0_a0"] #level 1
                    # self.V_DP_itp = LessLinear2D_interpolations["g20_m-20_a1"] #level 2
                    
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
                    self.V_DP_linear = V_N_DP_linear_itp_combo

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

            print("\nMaking a Bank of Evaluated Points ...")
            bank = torch.zeros(self.bank_total, self.N+3) # cols: time (1), state (2 - N+1), boundary value (N+2), value (N+3)
            bank[:, 1:self.N+1] = torch.zeros(self.bank_total, self.N).uniform_(-1, 1) 
            # TODO better sampling: latin hypercube? sparse grid? near boundary? on scored planes (is this cheating)?

            step = self.numpoints 
            with tqdm(total=self.numblocks) as pbar:
                for i in range(0, self.bank_total, step):

                    # Make T & X (model_coords)
                    bank[i:i+step, 0:self.N+1] = torch.cat((torch.full((step, 1), (i//step)*(self.tMax-self.tMin)/(self.numblocks-1)),
                                                                bank[i:i+step, 1:self.N+1]), dim=1)
                    
                    # Solve Boundary & Hopf Value 
                    bank[i:i+step, self.N+1] = self.dynamics.boundary_fn(self.dynamics.input_to_coord(bank[i:i+step, 0:self.N+1])[..., 1:])
                    bank[i:i+step, self.N+2] = self.V_hopf(self.dynamics.input_to_coord(bank[i:i+step, 0:self.N+1]).t())
                    pbar.update(1)

            ## Save Evaluated Bank and Delete it from Memory
            print("Done. Written to " + self.bank_name + ".\n")
            np.save(self.llnd_path + "banks/" + self.bank_name, bank)
            del(bank)
            gc.collect()
        
        ## Load Memory Map of the Bank
        if self.use_bank:

            self.bank = np.load(self.llnd_path + "banks/" + self.bank_name, mmap_mode='r')
            self.bank_index = torch.from_numpy(np.random.permutation(self.bank_total)) # random shuffle for sampling (should we also mmap this?)
            self.block_counter = 0
        
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        
        # ## Sample Points and Evaluate
        # if not(self.use_bank) or self.pretrain:

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

        ## Solve Hopf value
        if self.use_hopf:   

            if self.pretrain: # or when hopf loss is not being used (once turned off in regular training)
                hopf_values = torch.zeros(self.numpoints) #saves time

            ## Sample Bank of Hopf-Evaluated Points
            elif self.use_bank:

                sample_index = self.bank_index[self.block_counter*self.numpoints:(self.block_counter+1)*self.numpoints]
                bank_sample = torch.from_numpy(self.bank[sample_index, :])

                model_coords_hopf = bank_sample[:, 0:self.N+1] # separate states so pde loss is not restricted to small bank
                hopf_values = bank_sample[:, self.N+2]

                self.block_counter += 1
                if self.block_counter == self.numblocks:
                    self.bank_index = torch.from_numpy(np.random.permutation(self.bank_total)) # reshuffle
                    self.block_counter = 0
            
            ## Compute Value from Hopf-Interpolation
            else:
                try: hopf_values = self.V_hopf(self.dynamics.input_to_coord(model_coords).t()) # is slow in high d (yields)
                except: hopf_values = self.V_hopf(0.999 * self.dynamics.input_to_coord(model_coords).t()) # rare FP issue
            
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
            if not(self.use_bank) or self.hopf_pretrain_counter == 0:
                return {'model_coords': model_coords}, {'boundary_values': boundary_values, 'dirichlet_masks': dirichlet_masks, 'hopf_values': hopf_values}
            else:
                return {'model_coords': model_coords}, {'boundary_values': boundary_values, 'dirichlet_masks': dirichlet_masks, 'hopf_values': hopf_values, 'model_coords_hopf': model_coords_hopf}
        elif self.dynamics.loss_type == 'brat_hjivi':
            return {'model_coords': model_coords}, {'boundary_values': boundary_values, 'reach_values': reach_values, 'avoid_values': avoid_values, 'dirichlet_masks': dirichlet_masks}
        else:
            raise NotImplementedError