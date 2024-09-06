
from multiprocessing import Pool, Lock
from multiprocessing.shared_memory import SharedMemory
from tqdm import tqdm
import time
import numpy as np
import torch

## Multi-Processing

class HopfJuliaPool(object):
    ## By design, only one instance should be defined for class attribute stability
    ## (instance attributes do not transfer to processes, hence, need class)

    lock = None
    jl = None
    
    shm_states_name = None 
    shm_states_shape = None
    shm_algdat_name = None 
    shm_algdat_shape = None

    bank_total, bank_start, bank_invst, ts = 0, 0, 0, 0.

    solve_grad = False
    use_hopf = False
    hopf_warm_start = False

    Dynamics = None

    V_hopf = None
    V_hopf_ws = None
    V_hopf_grad = None
    V_hopf_grad_ws = None

    def __init__(self, dynamics, time_step, hopf_opt_p, bank_params, solve_grad=True, use_hopf=True, hopf_warm_start=True, num_hopf_workers=1):

        ## Initialize Julia Pool
        self.pool = Pool(num_hopf_workers, initializer=self.init_worker)
        self.jobs = []
        self.alg_iter = 0

        ## Shared Memory Params
        bank_total = bank_params["bank_total"]

        shm_states_shape = (bank_total, 1 + dynamics.N + 1 + 1 + dynamics.N) # bank_total x (time, state, bc, val, state_grad)
        shm_algdat_shape = (1000, 5) # alg_log_max x (alg_iter, job_ix, avg_comp_time, avg_mse, avg_grad_mse)

        shared_data_shapes = (shm_states_shape, shm_algdat_shape)

        ## Initialize Hopf (Blocks until done)
        self.pool.apply(self.init_solve, (dynamics, time_step, hopf_opt_p, shared_data_shapes, bank_params, solve_grad, use_hopf, hopf_warm_start))

    @classmethod
    def init_worker(cls):

        print('Initializing a process...')

        cls.lock = Lock()

        from juliacall import Main as jl, convert as jlconvert

        cls.jl = jl
        exec_load = """

        using LinearAlgebra

        include(pwd() * "/src/HopfReachability.jl");
        using .HopfReachability: Hopf_BRS, Hopf_cd, make_grid, make_target, make_set_params

        """
        cls.jl.seval(exec_load)

    @classmethod
    def init_solve(cls, dynamics, time_step, hopf_opt_p, shared_data_shapes, bank_params, solve_grad=True, use_hopf=True, hopf_warm_start=False):

        ## Store Some Class Attributes
        cls.dynamics = dynamics
        cls.solve_grad = solve_grad
        cls.use_hopf = use_hopf
        cls.hopf_warm_start = hopf_warm_start
    
        ## Define Shared Memory
        cls.shm_states_shape, cls.shm_algdat_shape = shared_data_shapes
        cls.bank_total, cls.bank_start, cls.bank_invst, cls.ts = bank_params["bank_total"], bank_params["bank_start"], bank_params["bank_invst"], time_step

        shm_states = SharedMemory(create=True, size=np.prod(cls.shm_states_shape) * np.dtype(np.float32).itemsize)
        shm_algdat = SharedMemory(create=True, size=np.prod(cls.shm_algdat_shape) * np.dtype(np.float32).itemsize)

        cls.shm_states_name, cls.shm_algdat_name = shm_states.name, shm_algdat.name

        ## Execute HopfReachability.jl's solve_BRS
        if use_hopf:
            hopf_setup_exec = f"""

            ## System & Game
            A, B₁, B₂, = {dynamics.A.numpy()}, {dynamics.B.numpy()}, {dynamics.C.numpy()}
            max_u, max_d, input_center, input_shapes = {dynamics.u_max}, {dynamics.d_max}, {dynamics.input_center.numpy()}, {dynamics.input_shape}
            # A, B₁, B₂ = -0.5*I + hcat(vcat(0, -ones(N-1,1)), zeros(N, N-1)), vcat(zeros(1, N-1), 0.4*I), vcat(zeros(1,N-1), 0.1*I) # system
            # max_u, max_d, input_center, input_shapes = 0.5, 0.3, zeros(N-1), "box"
            Q₁, c₁ = make_set_params(input_center, max_u; type=input_shapes) 
            Q₂, c₂ = make_set_params(input_center, max_d; type=input_shapes) # 𝒰 & 𝒟
            system, game = (A, B₁, B₂, Q₁, c₁, Q₂, c₂), "reach"

            ## Target
            N, r = {dynamics.N}, {dynamics.goalR}
            Q, center, radius = diagm(ones(N)), zeros(N), r
            radius_N, Q_N = sqrt(N-1) * radius, diagm(vcat(1/(N-1), inv(1) * ones(N-1)))
            target = make_target(center, radius_N; Q=Q_N, type="ellipse")

            ## Times
            TH = {time_step}
            times = collect(Th : Th : 1.);
            th = min(1e-2, Th)

            ## Optimization Parameters
            # vh, stepsz, tol, decay_stepsz, conv_runs_rqd, max_runs, max_its = 0.01, 1, 1e-3, 100, 1, 1, 100 # for N=100, gives MSE=0.85 & 27 min/60k
            # vh, stepsz, tol, decay_stepsz, conv_runs_rqd, max_runs, max_its = 0.01, 1, 1e-3, 300, 1, 1, 300 # for N=100, gives MSE=0.25 & 78 min/60k
            # vh, stepsz, tol, decay_stepsz, conv_runs_rqd, max_runs, max_its = 0.01, 1, 1e-3, 1000, 1, 1, 1000 # for N=100, gives MSE=0.13 & 253 min/60k
            opt_p_cd = ({hopf_opt_p["vh"]}, {hopf_opt_p["stepsz"]}, {hopf_opt_p["tol"]}, {hopf_opt_p["decay_stepsz"]}, {hopf_opt_p["conv_runs_rqd"]}, {hopf_opt_p["max_runs"]}, {hopf_opt_p["max_its"]})

            ## Grad Reshape Fn
            P_in_f(∇ϕX) = reshape(hcat(∇ϕX[2:end]...), size(∇ϕX[1])..., length(∇ϕX)-1)

            """
            cls.jl.seval(hopf_setup_exec)

            solve_Hopf_BRS_exec = f"""
            function solve_Hopf_BRS(X; P_in=nothing, return_grad=false)
                (XsT, ϕXsT), run_stats, opt_data, ∇ϕXsT = Hopf_BRS(system, target, times; X, th, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, P_in, warm="temporal")
                if return_grad
                    return ϕXsT[1], ϕXsT[2:end], P_in_f(∇ϕXsT)
                else
                    return ϕXsT[1], ϕXsT[2:end],
                end
            end
            """
            cls.solve_Hopf_BRS = cls.jl.seval(solve_Hopf_BRS_exec)
            cls.V_hopf = lambda tX: torch.from_numpy(cls.solve_Hopf_BRS(tX.numpy(), return_grad=False).to_numpy())
            cls.V_hopf_ws = lambda tX, P_in: torch.from_numpy(cls.solve_Hopf_BRS(tX.numpy(), P_in=P_in, return_grad=False).to_numpy())
            if solve_grad:
                cls.V_hopf_grad = lambda tX: torch.from_numpy(cls.solve_Hopf_BRS(tX.numpy(), return_grad=True).to_numpy())
                cls.V_hopf_grad_ws = lambda tX, P_in: torch.from_numpy(cls.solve_Hopf_BRS(tX.numpy(), P_in=P_in, return_grad=True).to_numpy())

        ## Interpolate the Dynamic Programming Solution
        else:
            cls.jl.seval("using JLD, JLD2, Interpolations")

            llnd_path = "value_fns/LessLinear/"
            V_itp = cls.jl.load(llnd_path + "interps/old/lin2d_hopf_interp_linear.jld")["V_itp"]

            fast_interp_exec = """

            function fast_interp(_V_itp, tX; compute_grad=false)
                V = zeros(size(tX,2))
                for i=1:length(V); V[i] = _V_itp(tX[:,i][end:-1:1]...); end # (assumes t in first row)
                J = zeros(size(tX,2))
                for i=1:length(V); V[i] = _V_itp(tX[:,i][end:-1:2]..., 0.); end
                if !compute_grad
                    return J, V
                else
                    DxV = zeros(size(tX,2), size(tX,1)-1)
                    for i=1:size(DxV,1); DxV[i,:] = Interpolations.gradient(_V_itp, tX[:,i][end:-1:1]...)[end-1:-1:1]; end # (assumes t in first row)
                    return J, V, DxV
                end
            end

            """
            fast_interp = cls.jl.seval(fast_interp_exec)
            cls.V_hopf = lambda tX: torch.from_numpy(fast_interp(V_itp, tX.numpy()).to_numpy())
            cls.V_hopf_grad = lambda tX: torch.from_numpy(fast_interp(V_itp, tX.numpy(), compute_grad=True).to_numpy())

    @classmethod
    def solve_hopf(cls, Xi, bix, tX_grad_ws=None):
        
        ## Point to Shared Memory
        shm_states = SharedMemory(name=cls.shm_states_name)
        shm_algdat = SharedMemory(name=cls.shm_algdat_name)
        bank = np.ndarray(shm_states, dtype=np.float32, buffer=shm_states.buf)
        alg_data = np.ndarray(shm_algdat, dtype=np.float32, buffer=shm_algdat.buf)

        split_size = Xi.shape[0] / cls.ts
        job_id = int(bix/split_size)
        
        ## Compute Value (and Gradient), w/w/o Learned Gradient for WS
        start_time = time.time()
        if cls.solve_grad:
            if tX_grad_ws is not None:
                J, V, DxV = cls.V_hopf_grad_ws(Xi, tX_grad_ws)
            else:
                J, V, DxV = cls.V_hopf_grad(Xi)
        else:
            if tX_grad_ws is not None:
                J, V = cls.V_hopf_grad(Xi, tX_grad_ws)
            else:
                J, V = cls.V_hopf(Xi)
        mean_time = (start_time - time.time()) / split_size

        if cls.use_hopf:
            J = np.repeat(J, int(1/cls.ts), axis=0)
            V = np.vstack(V)
            DxV = np.reshape(DxV, (cls.dynamics.N, split_size)).T()

        ## Put values into shared bank
       
        with cls.lock:
            bank[bix:bix+split_size, cls.dynamics.N+1] = J # boundary
            bank[bix:bix+split_size, cls.dynamics.N+2] = V # value
            if cls.solve_grad:
                bank[bix:bix+split_size, cls.dynamics.N+3:] = DxV # hopf-grad
            
            alg_data[job_id, 2] = mean_time # TODO: different ix for bank_invst

        return (job_id, mean_time)

    def solve_bank_start(self, n_splits=10):

        ## Point to Shared Memory
        shm_states = SharedMemory(name=self.shm_states_name)
        shm_algdat = SharedMemory(name=self.shm_algdat_name)
        bank = np.ndarray(shm_states, dtype=np.float32, buffer=shm_states.buf)
        alg_data = np.ndarray(shm_algdat, dtype=np.float32, buffer=shm_algdat.buf)

        ## split X into several jobs (X1, ... Xp)
        total_spatial_pts = self.bank_start // self.ts # maybe this should be chosen instead of bank_total
        split_spatial_pts = total_spatial_pts // n_splits
        split_size = self.bank_start // n_splits
        time_pts = 1 / self.ts
        if self.bank_start % split_spatial_pts != 0: raise AssertionError("Your bank isn't divided well, change your total or time-step") 

        with tqdm(total=n_splits) as pbar:
            
            ## Define Xi splits and store
            for i in range(0, self.bank_total, split_size):
                Xi = np.zeros(split_spatial_pts, self.dynamics.N).uniform_(-1, 1) 
                # TODO: w/ fixed Xi, check BC for use_hopf & w/o (to see alignment)

                for j in range(time_pts):
                    bank[i + j*split_spatial_pts: i + (j+1)*split_spatial_pts, 0:self.dynamics.N+1] = np.hstack((self.ts * (j+1) * np.ones(Xi.shape[0]), Xi))
                    ## TODO: here is where solve_bank_invst will look up the grads if warmstarting
                
                alg_data[i, 0], alg_data[i, 1] = self.alg_iter, i/split_size
        
            ## Execute (blocking) on all workers 
            for i in range(n_splits):
                Xi = bank[i*split_size: i*split_size + split_spatial_pts, 1:self.dynamics.N+1]
                job = self.pool.apply_async(self.solve_hopf, (Xi, i*split_size))
                self.jobs.append(job)

            for job in self.jobs:
                job_id, mean_time = job.get()
                # alg_data[job_id, 0], alg_data[job_id, 1], alg_data[job_id, 2] = self.alg_iter, job_id, mean_time

                # TODO: still need to check MSE... (MSE_solution (jl)? or ReachabilityDataset.V_DP_grad (py/jl)?)
                # I think this should happen inside (particularly for later, unless we want to implement a callback for end of every alg iter)
                
                # FIXME: I realize now DP method wont work w/ solve_hopf which takes Xi only
                # could redefine input args conditionally, maybe this should just be repurposed for MSE?
                
                # leaning towards julia in this case:
                # make_tXg(t, X) = vcat(t*ones(size(X,2))', X)
                # V_N_DP(t, X_N) = sum(fast_interp(V_DP_itp, make_tXg(t, view(X_N,[1, i+1],:))) for i=1:size(X_N,1)-1) # project to N-1 subspaces, getting val from DP interp
                # function MSE_solution(solution, times; subset=1:length(solution[2][1]), tix=1:length(times))
                #     MSE, MSEs = 0, 0. * zero(tix)
                #     for ti in tix
                #         MSEs[ti] = sum((V_N_DP(times[ti], solution[1][ti+1][:, subset]) - solution[2][ti+1][subset]).^2) / length(solution[2][ti+1][subset]) # subset could be bit array or indexes
                #         MSE += MSEs[ti] / length(times)
                #     end
                #     return MSE, MSEs
                # end


                pbar.update(1)

            self.alg_iter += 1
    
    def solve_bank_invst(self, X, model):

        ## will mostly be the same, but will ...
        # - use bank_invst instead of bank_start
        # - depending on alg iter, place in bank_start + alg_iter * bank_inves OR overwrite somewhere
        # - use model to warm start when alg_iter > 2
        # - no blocking catch at the end... new function for storing the alg data?

        # self.alg_iter += 1 maybe this doesnt happen since async

        pass
            
