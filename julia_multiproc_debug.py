
from tqdm import tqdm
import time
import numpy as np
import torch
import warnings
import contextlib

from multiprocessing import Pool, Lock
from multiprocessing.shared_memory import SharedMemory
import multiprocessing as mp

import sys
import os
import shutil

## Multi-Processing

class HopfJuliaPool(object):
    ## By design, only one instance should be defined for class attribute stability
    ## (instance attributes do not transfer to processes, hence, need class)

    lock = None
    jl = None
    log_loc = "runs/test_run/julia_multiproc_logs" ## FIXME: actual run name, not 'test_run'
    master_log = "master_log.log"
    worker_id = 0
    worker_log, worker_log_full = "", ""
    log_flush = None
    flush_julia_procs = None
    
    shm_states_id = None 
    shm_states_shape = None
    shm_algdat_id = None 
    shm_algdat_shape = None

    bank_total, bank_start, bank_invst, ts, tp = 0, 0, 0, 0., 0

    use_hopf = False
    solve_grad = False
    gt_metrics = False
    hopf_warm_start = False

    N = 0

    V_hopf = None
    V_hopf_ws = None
    V_hopf_grad = None
    V_hopf_grad_ws = None

    V_hopf_gt = None
    V_hopf_gt_grad = None

    def __init__(self):

        ## Parameters
        # self.num_hopf_workers = num_hopf_workers
        # self.jobs = []
        self.alg_iter = 0
        # self.use_hopf = use_hopf
        # self.solve_grad = solve_grad
        # self.gt_metrics = gt_metrics
        # self.hopf_warm_start = hopf_warm_start
        # self.hopf_opt_p = hopf_opt_p
        # self.ts = time_step
        # self.tp = int(1/self.ts)
        # self.N = dynamics_data["N"]
        # self.settings = (use_hopf, solve_grad, gt_metrics, hopf_warm_start)

        # ## Initialize Pool running Julia
        # print('\nInitializing pool...')
        # self.pool = Pool(num_hopf_workers, initializer=self.init_worker)
        # print("\nFinished initializing workers.")

        # ## Initialize HopfReachability.jl Solver
        # with tqdm(total=num_hopf_workers) as pbar:
            
        #     for jid in range(num_hopf_workers):
        #         job = self.pool.apply_async(self.init_solver, (dynamics_data, time_step, hopf_opt_p, settings))
        #         self.jobs.append(job)

        #     ## Block
        #     while self.jobs:
        #         for job in self.jobs:
        #             if job.ready():
        #                 print(pbar.get())
        #                 pbar.update(1)
        #                 self.jobs.remove(job)
        # print("Finished initializing solvers.")
        pass

    @classmethod
    def init_worker(cls):
        
        cls.worker_id = os.getpid()
        # cls.worker_log = f"worker_{worker_id}.log"
        # sys.stdout = open(cls.worker_log, 'w')
        cls.worker_log = f"worker_{cls.worker_id}.log"
        cls.worker_log_full = os.path.join(cls.log_loc, cls.worker_log)
        print(f'Initializing worker {cls.worker_id}...')

        cls.lock = Lock()

        warnings.filterwarnings("ignore", category=UserWarning, message="torch was imported before juliacall. This may cause a segfault. To avoid this, import juliacall before importing torch. For updates, see https://github.com/pytorch/pytorch/issues/78829.")
        from juliacall import Main as jl, convert as jlconvert
        cls.jl = jl

#         exec_print = f"""

# # Set Output

# global log_f = open("{cls.worker_log_full}", "a")
# redirect_stdout(log_f)
# redirect_stderr(log_f)""" ## at least see partial Hopf_BRS print (until it presumably errors)
#         cls.jl.seval(exec_print)

        exec_load = f"""

# Load

using Pkg
Pkg.activate()

using LinearAlgebra

include(pwd() * "/HopfReachability/src/HopfReachability.jl");
using .HopfReachability: Hopf_BRS, Hopf_cd, make_grid, make_target, make_set_params"""
        cls.jl.seval(exec_load)
        print(exec_load)

        ## Stdout Flusher, doesn't help/change juliacall muting
        # exec_flush = """
        # function log_flush()
        #     flush(log_f)
        #     close(log_f)
        # end
        # """
        # cls.log_flush = cls.jl.seval(exec_flush)
        # cls.flush_julia_procs = lambda: staticmethod(cls.log_flush()) # doesnt help

    @classmethod
    def init_solver(cls, dynamics_data, time_step, hopf_opt_p, settings):

        # print('Setting up solver...')
        
        ## Store Class Attributes
        use_hopf, solve_grad, gt_metrics, hopf_warm_start = settings
        cls.use_hopf = use_hopf
        cls.solve_grad = solve_grad
        cls.gt_metrics = gt_metrics
        cls.hopf_warm_start = hopf_warm_start
        cls.ts = time_step
        cls.tp = int(1/cls.ts)
        cls.N = dynamics_data["N"]

        ## Execute HopfReachability.jl's solve_BRS
        if cls.use_hopf:
#             hopf_setup_exec = f"""

# # Define the Problem


# ## System & Game
# A, B₁, B₂, = {dynamics_data["A"].numpy()}, {dynamics_data["B"].numpy()}, {dynamics_data["C"].numpy()}
# max_u, max_d, input_center, input_shapes = {dynamics_data["u_max"]}, {dynamics_data["d_max"]}, {dynamics_data["input_center"].numpy()}, "{dynamics_data["input_shape"]}"
# Q₁, c₁ = make_set_params(input_center, max_u; type=input_shapes) 
# Q₂, c₂ = make_set_params(input_center, max_d; type=input_shapes) # 𝒰 & 𝒟
# system, game = (A, B₁, B₂, Q₁, c₁, Q₂, c₂), "reach"

# ## Target
# N, r = {cls.N}, {dynamics_data["goalR"]}
# Q, center, radius = diagm(ones(N)), zeros(N), r
# radius_N, Q_N = sqrt(N-1) * radius, diagm(vcat(1/(N-1), inv(1) * ones(N-1)))
# target = make_target(center, radius_N; Q=Q_N, type="ellipse")

# ## Times
# Th = {cls.ts}
# times = collect(Th : Th : 1.);
# th = min(1e-2, Th)

# ## Optimization Parameters
# # vh, stepsz, tol, decay_stepsz, conv_runs_rqd, max_runs, max_its = 0.01, 1, 1e-3, 100, 1, 1, 100 # for N=100, gives MSE=0.85 & 27 min/60k
# # vh, stepsz, tol, decay_stepsz, conv_runs_rqd, max_runs, max_its = 0.01, 1, 1e-3, 300, 1, 1, 300 # for N=100, gives MSE=0.25 & 78 min/60k
# # vh, stepsz, tol, decay_stepsz, conv_runs_rqd, max_runs, max_its = 0.01, 1, 1e-3, 1000, 1, 1, 1000 # for N=100, gives MSE=0.13 & 253 min/60k
# opt_p_cd = ({hopf_opt_p["vh"]}, {hopf_opt_p["stepsz"]}, {hopf_opt_p["tol"]}, {hopf_opt_p["decay_stepsz"]}, {hopf_opt_p["conv_runs_rqd"]}, {hopf_opt_p["max_runs"]}, {hopf_opt_p["max_its"]})

# ## Grad Reshape Fn
# P_in_f(gradVX) = reshape(hcat(gradVX[2:end]...), size(gradVX[1])..., length(gradVX)-1)"""
#             print(hopf_setup_exec)
#             cls.jl.seval(hopf_setup_exec)

#             solve_Hopf_BRS_exec = f"""

# # Wrapper for Hopf Solver

# function solve_Hopf_BRS(X; P_in=nothing, return_grad=false)
#     println("About to solve hopf with")
#     println("VERSION", VERSION)
#     println(Sys.BINDIR)
#     println(Pkg.envdir())
#     println("\n X:")
#     print(X)
#     println("\n")
#     println("return_grad=", return_grad)
#     # flush(log_f)
#     (XsT, VXsT), run_stats, opt_data, gradVXsT = Hopf_BRS(system, target, times; X, th, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, P_in, warm=true, warm_pattern="temporal", printing=true)
#     println("solved hopf")
#     # flush(log_f)
#     if return_grad
#         return VXsT[1], VXsT[2:end], P_in_f(gradVXsT)
#     else
#         return VXsT[1], VXsT[2:end]
#     end
# end"""    
            hopf_setup_exec = """

# Define the Problem


## System & Game
A, B₁, B₂, = [[-0.5 -0.  -0.  -0.  -0.  -0.  -0. ]
 [-1.  -0.5 -0.  -0.  -0.  -0.  -0. ]
 [-1.  -0.  -0.5 -0.  -0.  -0.  -0. ]
 [-1.  -0.  -0.  -0.5 -0.  -0.  -0. ]
 [-1.  -0.  -0.  -0.  -0.5 -0.  -0. ]
 [-1.  -0.  -0.  -0.  -0.  -0.5 -0. ]
 [-1.  -0.  -0.  -0.  -0.  -0.  -0.5]], [[0.  0.  0.  0.  0.  0. ]
 [0.4 0.  0.  0.  0.  0. ]
 [0.  0.4 0.  0.  0.  0. ]
 [0.  0.  0.4 0.  0.  0. ]
 [0.  0.  0.  0.4 0.  0. ]
 [0.  0.  0.  0.  0.4 0. ]
 [0.  0.  0.  0.  0.  0.4]], [[0.  0.  0.  0.  0.  0. ]
 [0.1 0.  0.  0.  0.  0. ]
 [0.  0.1 0.  0.  0.  0. ]
 [0.  0.  0.1 0.  0.  0. ]
 [0.  0.  0.  0.1 0.  0. ]
 [0.  0.  0.  0.  0.1 0. ]
 [0.  0.  0.  0.  0.  0.1]]
max_u, max_d, input_center, input_shapes = 0.5, 0.3, [0. 0. 0. 0. 0. 0.], "box"
Q₁, c₁ = make_set_params(input_center, max_u; type=input_shapes) 
Q₂, c₂ = make_set_params(input_center, max_d; type=input_shapes) # 𝒰 & 𝒟
system, game = (A, B₁, B₂, Q₁, c₁, Q₂, c₂), "reach"

## Target
N, r = 7, 0.36742346141747667
Q, center, radius = diagm(ones(N)), zeros(N), r
radius_N, Q_N = sqrt(N-1) * radius, diagm(vcat(1/(N-1), inv(1) * ones(N-1)))
target = make_target(center, radius_N; Q=Q_N, type="ellipse")

## Times
# Th = 0.001
Th = 0.1
times = collect(Th : Th : 1.);
th = min(1e-2, Th)

## Optimization Parameters
# vh, stepsz, tol, decay_stepsz, conv_runs_rqd, max_runs, max_its = 0.01, 1, 1e-3, 100, 1, 1, 100 # for N=100, gives MSE=0.85 & 27 min/60k
# vh, stepsz, tol, decay_stepsz, conv_runs_rqd, max_runs, max_its = 0.01, 1, 1e-3, 300, 1, 1, 300 # for N=100, gives MSE=0.25 & 78 min/60k
# vh, stepsz, tol, decay_stepsz, conv_runs_rqd, max_runs, max_its = 0.01, 1, 1e-3, 1000, 1, 1, 1000 # for N=100, gives MSE=0.13 & 253 min/60k
opt_p_cd = (0.01, 1, 0.001, 100, 1, 1, 100)

## Grad Reshape Fn
P_in_f(gradVX) = reshape(hcat(gradVX[2:end]...), size(gradVX[1])..., length(gradVX)-1)"""
            print(hopf_setup_exec)
            cls.jl.seval(hopf_setup_exec)

            solve_Hopf_BRS_exec = """
# Wrapper for Hopf Solver

function solve_Hopf_BRS(X_in; P_in=nothing, return_grad=false)
    # X = Float32[-0.11893158 0.038724087 0.40518385 0.27271906 0.66952044; -0.91246986 0.6038015 0.6672152 0.49143624 0.75128675; 0.6455725 -0.22062263 0.839456 -0.74276614 -0.58126867; 0.7025011 -0.270896 0.4259202 0.077222735 -0.3687684; 0.29084364 -0.67864865 0.46917862 0.5430416 0.7472049; -0.0019951398 -0.890628 -0.8739691 0.4054919 -0.11211838; 0.7379937 0.010344575 0.63341236 0.03499771 -0.82848865]
    X = Float32[-0.31805927 0.23355128 0.37032932 0.09179542 -0.55636364; -0.36180115 -0.7545929 -0.76992947 0.7226987 -0.9838805; 0.37839848 0.22396313 0.2952317 0.09237017 -0.75982624; -0.5299584 -0.60631984 0.85858893 -0.36559156 0.72176003; -0.35700798 0.13337342 0.55256563 -0.5108615 -0.18235427; 0.75992125 0.91217756 -0.41531298 0.5474081 0.7569677; -0.8773644 0.69777286 -0.8319238 -0.61119413 -0.7741825];
    println("X == X_in", X == X_in)
    println("typeof(X_in)", typeof(X_in))
    println("typeof(Matrix(X_in))", typeof(Matrix(X_in)))
    println("typeof(X)", typeof(X))
    # P_in = nothing
    # return_grad=false
    println("About to solve hopf with")
    println("VERSION", VERSION)
    println(Sys.BINDIR)
    println(Pkg.envdir())
    println("X:")
    print(X)
    println("")
    println("return_grad=", return_grad)
    # flush(log_f)
    (XsT, VXsT), run_stats, opt_data, gradVXsT = Hopf_BRS(system, target, times; X=Matrix(X_in), th, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, P_in, warm=true, warm_pattern="temporal", printing=true)
    println("solved hopf")
    # flush(log_f)
    if return_grad
        return VXsT[1], vcat(VXsT[2:end]...), P_in_f(gradVXsT)
    else
        return VXsT[1], vcat(VXsT[2:end]...)
    end
end"""        

            cls.solve_hopf_BRS = cls.jl.seval(solve_Hopf_BRS_exec)
            print(solve_Hopf_BRS_exec)
            # cls.V_hopf = lambda tX: cls.solve_hopf_BRS() ## TODO: rmv, debugging
            cls.V_hopf = lambda tX: cls.solve_hopf_BRS(tX, return_grad=False)
            cls.V_hopf_ws = lambda tX, P_in: cls.solve_hopf_BRS(tX, P_in=P_in, return_grad=False)
            if solve_grad:
                cls.V_hopf_grad = lambda tX: cls.solve_hopf_BRS(tX, return_grad=True)
                cls.V_hopf_grad_ws = lambda tX, P_in: cls.solve_hopf_BRS(tX, P_in=P_in, return_grad=True)

        ## Interpolate the Dynamic Programming Solution
        if not cls.use_hopf or cls.gt_metrics:
            cls.jl.seval("using JLD, JLD2, Interpolations")

            llnd_path = "value_fns/LessLinear/"
            # V_itp = cls.jl.load(llnd_path + "interps/old/lin2d_hopf_interp_linear.jld")["V_itp"]
            V_itp = cls.jl.load(llnd_path + f"interps/LessLinear2D1i_interpolations_res1e-2_r{int(100 * dynamics_data['goalR_2d'])}e-2_c20.jld", "LessLinear2D_interpolations")["g0_m0_a0"]

            fast_interp_exec = """

# Warpper for Interpolation

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
end"""
            fast_interp = cls.jl.seval(fast_interp_exec)
            cls.V_hopf_gt = lambda tX: fast_interp(V_itp, tX)
            cls.V_hopf_gt_grad = lambda tX: fast_interp(V_itp, tX, compute_grad=True)

            # print(cls.V_hopf_gt())

        return 

    @classmethod
    def solve_hopf(cls, Xi, bix, shm_data, tX_grad=None):

        print("Xi (IN):", Xi)
        print("type(Xi) (IN):", type(Xi))
        # print("Xi (shape):", Xi.shape)

        split_size = int(Xi.shape[1] / cls.ts)
        job_id = int(bix/split_size)

        # print("split_size", split_size)

        ## Point to Shared Memory
        shm_states_id, shm_states_shape, shm_algdat_id, shm_algdat_shape = shm_data
        shm_states = SharedMemory(name=shm_states_id)
        shm_algdat = SharedMemory(name=shm_algdat_id)
        bank = np.ndarray(shm_states_shape, dtype=np.float32, buffer=shm_states.buf)
        alg_data = np.ndarray(shm_algdat_shape, dtype=np.float32, buffer=shm_algdat.buf)

        if not cls.use_hopf or cls.gt_metrics:
            tXi = np.vstack([np.hstack([(j+1) * cls.ts * torch.ones(Xi.shape[0],1), Xi]) for j in range(cls.tp)])

        ## Redirect Output to Log files
        # job_log = f"worker_{cls.worker_id}_job_{job_id}.log"
        # job_log_full = os.path.join(cls.log_loc, job_log)
        # print(f"Worker {cls.worker_id}, job {job_id}: Solving Hopf formula... writing to {job_log}")
        # sys.stdout = open(job_log_full, 'w')
        # job_log_file = open(job_log_full, 'a')
        # jl.seval(f"""
        #     redirect_stdout(unsafe_string(pointer_from_objref(PyCall.pyimport('builtins').open))("{job_log_full}", "a"))
        # """)
        # sys.stdout.flush()
        # redirect_stdout(open("{job_log_full}", "a"))
        # redirect_stderr(open("{job_log_full}", "a"))

        # cls.jl.seval(f"""

        # global log_f = open("{job_log_full}", "a")
        # redirect_stdout(log_f)
        # redirect_stderr(log_f)

        # """) ## cant see Hopf_BRS print this way?

        # with contextlib.redirect_stdout(job_log_file), contextlib.redirect_stderr(job_log_file):

        print(f"\n ########### Worker {cls.worker_id}, job {job_id} ###########\n")
        
        start_time = time.time()

        ## Compute Value (and Gradient) with Hopf, w/w/o Some Gradients for Warm-Starting Hopf
        if cls.use_hopf:
            if cls.solve_grad:
                if tX_grad is not None:
                    J, V, DxV = cls.V_hopf_grad_ws(Xi, tX_grad)
                else:
                    J, V, DxV = cls.V_hopf_grad(Xi)
            else:
                if tX_grad is not None:
                    J, V = cls.V_hopf_grad(Xi, tX_grad)
                else:
                    J, V = cls.V_hopf(Xi)

        ## Compute Value with Composed 2D Interpolation (for testing)
        else: 
            if cls.solve_grad:
                J, V, DxV = cls.V_hopf_gt_grad(tXi)
            else:
                J, V = cls.V_hopf_gt(tXi)
            print("interpolated!")
            

        mean_time = (time.time() - start_time) / split_size # number of pts solved

        if cls.use_hopf:
            J = np.repeat(J, int(1/cls.ts))
            # V = np.hstack(V)
            if cls.solve_grad:
                DxV = np.reshape(DxV, (cls.N, split_size)).T()

            ## Solve Ground Truth
            if cls.gt_metrics:

                if cls.solve_grad:
                    J_gt, V_gt, DxV_gt = cls.V_hopf_gt_grad(tXi)
                else:
                    J_gt, V_gt = cls.V_hopf_gt(tXi)

                SE = np.pow((V_gt - V), 2)
                if cls.solve_grad:
                    SE_grad = np.pow((DxV_gt - DxV), 2).mean(dim=1) # mse per pt

                MSE, MSE_grad = SE.mean(), SE_grad.mean()

        ## Store in Shared Memory
        with cls.lock:
            
            ## Store Solved Bank Data
            bank[bix:bix+split_size, cls.N+1] = J # boundary
            bank[bix:bix+split_size, cls.N+2] = V # value
            if cls.solve_grad:
                bank[bix:bix+split_size, cls.N+4:2*cls.N+2] = DxV # hopf-grad
            if cls.gt_metrics:
                bank[bix:bix+split_size, cls.N+3] = SE # error
                if cls.solve_grad:
                    bank[bix:bix+split_size, 2*cls.N+2] = SE_grad # grad error
            
            ## Store General Algorithm Data TODO: different ix (job_id+n) for bank_invst
            alg_data[job_id, 2] = mean_time
            if cls.gt_metrics:
                alg_data[job_id, 3] = MSE
                alg_data[job_id, 4] = MSE_grad
            
            # ## Write Job Log to Master Log
            # with open(job_log_full, 'r') as wlog, open(os.path.join(cls.log_loc, cls.master_log), 'a') as mlog:
            #     shutil.copyfileobj(wlog, mlog)
        
        ## Delete job log and restore Julia output
        # try:
        #     os.remove(job_log_full)
        # except OSError as e:
        #     print(f"Error deleting worker log {job_log}: {e}")
        # jl.seval(f"""
        #     redirect_stdout(stdout)
        # """)
        # job_log_file.close()

        return (job_id, mean_time)

    def solve_bank_start(self, dynamics_data, time_step, hopf_opt_p, bank_params, 
                solve_grad=False, use_hopf=False, hopf_warm_start=False, gt_metrics=False, 
                n_splits=1):
        
        self.ts = time_step
        self.tp = int(1/self.ts)
        self.N = dynamics_data["N"]
        num_hopf_workers = n_splits
        jobs = []
        settings = (use_hopf, solve_grad, gt_metrics, hopf_warm_start)

        print("\nMaster log at: ", os.path.join(self.log_loc, self.master_log))
        with open(os.path.join(self.log_loc, self.master_log), 'a') as mlog:
            mlog.write("\n########################### Bank Starter Logs ###########################\n")

        ## Define Shared Memory
        bank_total, bank_start, bank_invst, ts = bank_params["bank_total"], bank_params["bank_start"], bank_params["bank_invst"], time_step
        shm_states_shape = (bank_total, 1 + self.N + 3 + dynamics.N + 1) # bank_total x (time, state, bc, val, mse, state_grad, mse_grad)
        shm_algdat_shape = (1000, 5) # alg_log_max x (alg_iter, job_ix, avg_comp_time, avg_mse, avg_grad_mse)

        shm_states = SharedMemory(create=True, size=np.prod(shm_states_shape) * np.dtype(np.float32).itemsize)
        shm_algdat = SharedMemory(create=True, size=np.prod(shm_algdat_shape) * np.dtype(np.float32).itemsize)
        shm_states_id, shm_algdat_id = shm_states.name, shm_algdat.name
        shm_data = (shm_states_id, shm_states_shape, shm_algdat_id, shm_algdat_shape)

        bank = np.ndarray(shm_states_shape, dtype=np.float32, buffer=shm_states.buf)
        alg_data = np.ndarray(shm_algdat_shape, dtype=np.float32, buffer=shm_algdat.buf)

        ## split X into several jobs (X1, ... Xp)
        total_spatial_pts = int(bank_start / self.tp) # maybe this should be chosen instead of bank_total
        split_spatial_pts = int(total_spatial_pts / n_splits)
        split_size = int(bank_start / n_splits)

        ## Care
        print(f"\nSolving {bank_start} points to start the bank (in {n_splits} jobs for {num_hopf_workers} workers), composed of {total_spatial_pts} spatial x {self.tp} time pts ({split_size} per job).\n")
        if bank_start % split_spatial_pts != 0: raise AssertionError(f"Your bank isn't divided well ({n_splits} splits gives {bank_start % split_spatial_pts} pts/split); change your bank total or time-step") 

        ## Initialize Pool running Julia
        # print('\nInitializing pool...')
        # num_hopf_workers = n_splits
        # jobs = []
        # pool = Pool(num_hopf_workers, initializer=self.init_worker)
        # print("\nFinished initializing workers.")

        with Pool(num_hopf_workers, initializer=self.init_worker) as pool, tqdm(total=n_splits) as pbar:

            for jid in range(num_hopf_workers):
                job = pool.apply_async(self.init_solver, (dynamics_data, time_step, hopf_opt_p, settings))
                jobs.append(job)

            ## Block
            while jobs:
                for job in jobs:
                    if job.ready():
                        print(job.get())
                        # pbar.update(1)
                        jobs.remove(job)
            print("Finished initializing solvers.")
            
            ## Define Xi splits and store
            for i in range(0, bank_total, split_size):
                Xi = np.random.uniform(-1, 1, (split_spatial_pts, self.N)) 
                # TODO: try w/ fixed Xi to check BC for solve_hopf & w/o (to see alignment)

                for j in range(self.tp):
                    bank[i + j*split_spatial_pts: i + (j+1)*split_spatial_pts, 0:self.N+1] = np.hstack((self.ts * (j+1) * np.ones((Xi.shape[0],1)), Xi))
                    ## TODO: here is where solve_bank_invst will look up the grads if warmstarting
                
                alg_data[int(i/split_size), 0], alg_data[int(i/split_size), 1] = self.alg_iter, i/split_size
        
            ## Execute (blocking) on all workers 
            for i in range(n_splits):
                Xi = bank[i*split_size: i*split_size + split_spatial_pts, 1:self.N+1].T
                job = pool.apply_async(self.solve_hopf, (Xi, int(i*split_size), shm_data))
                jobs.append(job)
                if i == 0: print(Xi.shape); print(type(Xi))
                # print("jobs len:", len(self.jobs))

            # for job in self.jobs:
            #     job_id, mean_time = job.get()
            #     # TODO: in invst, will store alg_data[job_id, 0] = self.alg_iter (automatically)
            #     pbar.update(1)
            
            print("While jobs, jobs len:", len(jobs))
            while jobs:
                for job in jobs:
                    if job.ready():
                        # print("Job ready!")
                        print(job.get())
                        pbar.update(1)
                        jobs.remove(job)
            
            # while self.jobs:
            #     for job in self.jobs:
            #         try:
            #             if job.ready():
            #                 # print("Job ready!")
            #                 pbar.update(1)
            #                 self.jobs.remove(job)
            #         except Exception as e:
            #             print(f"JOB OBJECT BROKEN: {e}")
            #             self.jobs.remove(job)

        # ## Flush In Case of Error (doesnt help/change output)
        # for i in range(self.num_hopf_workers):
        #     job = self.pool.apply_async(self.flush_julia_procs)
        #     self.jobs.append(job)

        # while self.jobs:
        #     for job in self.jobs:
        #         if job.ready():
        #             self.jobs.remove(job)
        
        print("Finished solving bank starter.")

        with open(os.path.join(self.log_loc, self.master_log), 'a') as mlog:
            mlog.write("\n########################    End of Bank Starter Logs    ##################")

        print("\n\nBANK SAMPLE")
        print(np.around(bank[0:n_splits, :], decimals=2))

        print("\n\nALG DATA SAMPLE")
        print(alg_data[0:n_splits, :])

        self.alg_iter += 1
        return shm_data
    
    def solve_bank_invst(self, X, model):

        with open(os.path.join(self.log_loc, self.master_log), 'a') as mlog:
            mlog.write("\n######################## Start of Bank Invstestment Logs #################")

        ## will mostly be the same, but will ...
        # - use bank_invst instead of bank_start
        # - depending on alg iter, place in bank_start + alg_iter * bank_inves OR overwrite somewhere
        # - use model to warm start when alg_iter > 2
        # - no blocking catch at the end... new function for storing the alg data?
        # could this just be the same fn w/ some if's

        pass

from dynamics.dynamics import LessLinearND
# import logging

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    # mp.log_to_stderr(logging.DEBUG)

    dynamics = LessLinearND(7, 0., 0., 0.)
    dynamics_data = {param:getattr(dynamics, param) for param in dir(dynamics) if not (param.startswith('__') or param.startswith('_')) and not callable(getattr(dynamics, param))}
    dynamics_data = {key:val.cpu() if torch.is_tensor(val) else val for key,val in dynamics_data.items()}
    
    # time_step = 1e-3
    time_step = 1e-1
    hopf_opt_p = {"vh":0.01, "stepsz":1, "tol":1e-3, "decay_stepsz":100, "conv_runs_rqd":1, "max_runs":1, "max_its":100} 

    # hjpool = HopfJuliaPool(dynamics_data, time_step, hopf_opt_p,
                            # use_hopf=False, solve_grad=False, hopf_warm_start=False, gt_metrics=True, num_hopf_workers=1)
    hjpool = HopfJuliaPool()

    # bank_params = {"bank_total":200000, "bank_start":100000, "bank_invst":10000}
    bank_params = {"bank_total":100, "bank_start":50, "bank_invst":10}
    shm_data = hjpool.solve_bank_start(dynamics_data, time_step, hopf_opt_p, bank_params, 
                                        use_hopf=True, solve_grad=False, hopf_warm_start=False, gt_metrics=False, n_splits=1)
    
    # shm_states_id, shm_states_shape, shm_algdat_id, shm_algdat_shape = shm_data
    # shm_states, shm_algdat = SharedMemory(name=shm_states_id), SharedMemory(name=shm_algdat_id)
    # bank = np.ndarray(shm_states_shape, dtype=np.float32, buffer=shm_states.buf)
    # alg_data = np.ndarray(shm_algdat_shape, dtype=np.float32, buffer=shm_algdat.buf)

    # ## debugging serialization error...
    # hjpool.init_worker()
    # settings = (True, False, True, False)
    # hjpool.init_solver(dynamics_data, time_step, hopf_opt_p, settings)

    # X = np.random.uniform(-1, 1, (10, hjpool.N))
    # # result = hjpool.V_hopf(X)

    # tX = np.hstack(0.1 * np.ones(10,1), X)
    # result = hjpool.V_hopf_gt(X)

    # import pickle
    # pickle.dumps(result)

    print("He hecho\n")
            
