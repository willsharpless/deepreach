from juliacall import Main as jl, convert as jlconvert
import torch
from torch.utils.data import Dataset

# uses model input and real boundary fn
class ReachabilityDataset(Dataset):
    def __init__(self, dynamics, numpoints, pretrain, pretrain_iters, tMin, tMax, counter_start, counter_end, num_src_samples, num_target_samples, use_hopf=False, hopf_pretrain=False, hopf_pretrain_iters=0):
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
        self.hopf_pretrain_counter = 0
        self.hopf_pretrain_iters = hopf_pretrain_iters

        ## compute Hopf value interpolant (if hopf loss)
        # using dynamics load/solve corresponding HopfReachability.jl code to get interpolation solution
        if use_hopf:
            jl.seval("using JLD2, Interpolations")
            self.V_hopf_itp = jl.load("lin2d_hopf_interp_linear.jld")["V_itp"]
            fast_interp_exec = """
            function fast_interp(_V_itp, tXg, method="grid")
                # assumes tXg has time in first row
                if method == "grid"
                    Vg = zeros(size(tXg,2))
                    for i=1:length(Vg)
                        Vg[i] = _V_itp(tXg[:,i][end:-1:1]...)
                    end
                else
                    Vg = ScatteredInterpolation.evaluate(_V_itp, tXg)
                end
                return Vg
            end
            """
            self.fast_interp = jl.seval(fast_interp_exec)
            self.V_hopf = lambda tXg: torch.from_numpy(self.fast_interp(self.V_hopf_itp, tXg.numpy()).to_numpy())
        
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
            # slowly grow time values from start time
            times = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * ((self.counter + self.hopf_pretrain_counter)/(self.counter_end + self.hopf_pretrain_iters)))
            # make sure we always have training samples at the initial time
            times[-self.num_src_samples:, 0] = self.tMin

        model_coords = torch.cat((times, model_states), dim=1)        
        if self.dynamics.input_dim > self.dynamics.state_dim + 1: # temporary workaround for having to deal with dynamics classes for parametrized models with extra inputs
            model_coords = torch.cat((model_coords, torch.zeros(self.numpoints, self.dynamics.input_dim - self.dynamics.state_dim - 1)), dim=1)      

        boundary_values = self.dynamics.boundary_fn(self.dynamics.input_to_coord(model_coords)[..., 1:])
        if self.dynamics.loss_type == 'brat_hjivi':
            reach_values = self.dynamics.reach_fn(self.dynamics.input_to_coord(model_coords)[..., 1:])
            avoid_values = self.dynamics.avoid_fn(self.dynamics.input_to_coord(model_coords)[..., 1:])
        
        ## TODO: compute Hopf value
        # compute find/solve value at model_coords with Hopf (prob from preloaded interpolation for now)
        # hopf_values = boundary_values.detach().clone()
        if self.use_hopf:
            try:
                hopf_values = self.V_hopf(self.dynamics.input_to_coord(model_coords).t())
            except:
                # FP error, will just interpolate outside of range in future
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