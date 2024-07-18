from juliacall import Main as jl, convert as jlconvert
import torch
from torch.utils.data import Dataset

# uses model input and real boundary fn
class ReachabilityDataset(Dataset):
    def __init__(self, dynamics, numpoints, pretrain, pretrain_iters, tMin, tMax, counter_start, counter_end, num_src_samples, num_target_samples, 
                 use_hopf=False, hopf_pretrain=False, hopf_pretrain_iters=0, hopf_loss_decay=False, hopf_loss_decay_w=0., diff_con_loss_incr=False, record_set_metrics=False,
                 manual_load=False, load_packet=None, no_curriculum=False):
        
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
        self.hopf_loss_decay = hopf_loss_decay
        self.hopf_loss_decay_w = hopf_loss_decay_w
        self.diff_con_loss_incr = hopf_loss_decay and diff_con_loss_incr
        self.record_set_metrics = record_set_metrics
        self.no_curriculum = no_curriculum

        if manual_load:
            self.V_hopf_itp, self.fast_interp, self.V_hopf, self.V_DP_itp, self.V_DP = load_packet

        # print("About to call julia")

        ## compute Hopf value interpolant (if hopf loss)
        # using dynamics load/solve corresponding HopfReachability.jl code to get interpolation solution
        if use_hopf and not(manual_load):
            jl.seval("using JLD2, Interpolations")
            self.V_hopf_itp = jl.load("lin2d_hopf_interp_linear.jld")["V_itp"]
            fast_interp_exec = """
            function fast_interp(_V_itp, tXg)
                # assumes tXg has time in first row
                Vg = zeros(size(tXg,2))
                for i=1:length(Vg)
                    Vg[i] = _V_itp(tXg[:,i][end:-1:1]...)
                end
                return Vg
            end
            """
            self.fast_interp = jl.seval(fast_interp_exec)
            self.V_hopf = lambda tXg: torch.from_numpy(self.fast_interp(self.V_hopf_itp, tXg.numpy()).to_numpy())
            
        # print("About to call julia again")

        if record_set_metrics:
            if not(manual_load):
                jl.seval("using JLD2, Interpolations")
                # self.V_DP_itp = jl.load("llin2d_g20_m0_a0_DP_interp_linear.jld")["V_itp"]
                self.V_DP_itp = jl.load("llin2d_g20_m-20_a1_DP_interp_linear.jld")["V_itp"]
                # self.V_DP_itp = jl.load("llin2d_g20_m20_a-20_DP_interp_linear.jld")["V_itp"]
                
                fast_interp_exec = """
                function fast_interp(_V_itp, tXg)
                    # assumes tXg has time in first row
                    Vg = zeros(size(tXg,2))
                    for i=1:length(Vg)
                        Vg[i] = _V_itp(tXg[:,i][end:-1:1]...)
                    end
                    return Vg
                end
                """
                if not(hasattr(self, 'fast_interp')):
                    self.fast_interp = jl.seval(fast_interp_exec)
                self.V_DP = lambda tXg: torch.from_numpy(self.fast_interp(self.V_DP_itp, tXg.numpy()).to_numpy())
                

            self.n_grid_t_pts, self.n_grid_t_pts_hi = 5, 20
            xig = torch.arange(-0.99, 1.01, 0.02) # 100 x 100
            X1g, X2g = torch.meshgrid(xig, xig)
            self.model_states_grid = torch.cat((X1g.ravel().reshape((1,xig.size()[0]**2)), X2g.ravel().reshape((1,xig.size()[0]**2))), dim=0).t()
            self.n_grid_pts = xig.size()[0]**2

            times = torch.full((self.n_grid_pts, 1), self.tMin)
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

            # print("About to move stuff to CUDA")

            self.values_DP_grid = self.V_DP(self.dynamics.input_to_coord(self.model_coords_grid_allt).t()).cuda()
            self.values_DP_grid_sub0_ixs = torch.argwhere(self.values_DP_grid <= 0).flatten().cuda()

            self.values_DP_grid_hi = self.V_DP(self.dynamics.input_to_coord(self.model_coords_grid_allt_hi).t()).cuda()
            self.values_DP_grid_sub0_ixs_hi = torch.argwhere(self.values_DP_grid_hi <= 0).flatten().cuda()

            self.model_coords_grid_allt = self.model_coords_grid_allt.cuda()
            self.model_coords_grid_allt_hi = self.model_coords_grid_allt_hi.cuda()
            self.model_states_grid = self.model_states_grid.cuda()
        
    def __len__(self):
        return 1

    def __getitem__(self, idx):
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

        boundary_values = self.dynamics.boundary_fn(self.dynamics.input_to_coord(model_coords)[..., 1:])
        if self.dynamics.loss_type == 'brat_hjivi':
            reach_values = self.dynamics.reach_fn(self.dynamics.input_to_coord(model_coords)[..., 1:])
            avoid_values = self.dynamics.avoid_fn(self.dynamics.input_to_coord(model_coords)[..., 1:])
        
        ## Compute Hopf value
        # compute find/solve value at model_coords with Hopf (from preloaded interpolation for now)
        if self.use_hopf:
            try:
                hopf_values = self.V_hopf(self.dynamics.input_to_coord(model_coords).t()) # 2x for diff in val fn
            except:
                # TODO: interpolate outside of range in future (FP issue)
                hopf_values = self.V_hopf(0.999 * self.dynamics.input_to_coord(model_coords).t())

        
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
            return {'model_coords': model_coords}, {'boundary_values': boundary_values, 'dirichlet_masks': dirichlet_masks, 'hopf_values': hopf_values}
        elif self.dynamics.loss_type == 'brat_hjivi':
            return {'model_coords': model_coords}, {'boundary_values': boundary_values, 'reach_values': reach_values, 'avoid_values': avoid_values, 'dirichlet_masks': dirichlet_masks}
        else:
            raise NotImplementedError