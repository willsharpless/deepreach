
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
    interp_file_name = ""
    write_output = True
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

    def __init__(self, dynamics_data, time_step, hopf_opt_p, 
                solve_grad=True, use_hopf=True, hopf_warm_start=True, gt_metrics=True, 
                num_hopf_workers=2):

        ## Parameters
        self.num_hopf_workers = num_hopf_workers
        self.jobs = []
        self.alg_iter = 0
        self.use_hopf = use_hopf
        self.solve_grad = solve_grad
        self.gt_metrics = gt_metrics
        self.hopf_warm_start = hopf_warm_start
        self.ts = time_step
        self.tp = int(1/self.ts)
        self.N = dynamics_data["N"]
        settings = (use_hopf, solve_grad, gt_metrics, hopf_warm_start)

        ## Initialize Pool running Julia
        print('\nInitializing pool...')
        self.pool = Pool(num_hopf_workers, initializer=self.init_worker)
        print("\nFinished initializing workers.")

        ## Initialize HopfReachability.jl Solver
        with tqdm(total=num_hopf_workers) as pbar:
            
            for jid in range(num_hopf_workers):
                job = self.pool.apply_async(self.init_solver, (dynamics_data, time_step, hopf_opt_p, settings))
                self.jobs.append(job)

            ## Block
            while self.jobs:
                for job in self.jobs:
                    if job.ready():
                        pbar.update(1)
                        self.jobs.remove(job)
        print("Finished initializing solvers.")

    @classmethod
    def init_worker(cls):
        
        cls.worker_id = os.getpid()
        cls.worker_log = f"worker_{cls.worker_id}.log"
        cls.worker_log_full = os.path.join(cls.log_loc, cls.worker_log)
        print(f'Initializing worker {cls.worker_id}...')

        cls.lock = Lock()

        warnings.filterwarnings("ignore", category=UserWarning, message="torch was imported before juliacall. This may cause a segfault. To avoid this, import juliacall before importing torch. For updates, see https://github.com/pytorch/pytorch/issues/78829.")
        from juliacall import Main as jl, convert as jlconvert
        cls.jl = jl

        if cls.write_output:
            exec_print = f"""\n
# Set Output

global log_f = open("{cls.worker_log_full}", "a")
redirect_stdout(log_f)
redirect_stderr(log_f)"""
            cls.jl.seval(exec_print)

            print(f"\n\n ################ Worker {cls.worker_id} log ################\n\n")

        exec_load = f"""\n
# Load

using LinearAlgebra

include(pwd() * "/HopfReachability/src/HopfReachability.jl");
using .HopfReachability: Hopf_BRS, Hopf_cd, make_grid, make_target, make_set_params"""
        cls.jl.seval(exec_load)
        print(exec_load)

    @classmethod
    def init_solver(cls, dynamics_data, time_step, hopf_opt_p, settings):
        
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
            hopf_setup_exec = f"""\n
# Define the Problem

## System & Game
A, B₁, B₂, = {dynamics_data["A"].numpy()}, {dynamics_data["B"].numpy()}, {dynamics_data["C"].numpy()}
max_u, max_d, input_center, input_shapes = {dynamics_data["u_max"]}, {dynamics_data["d_max"]}, {dynamics_data["input_center"].numpy()}, "{dynamics_data["input_shape"]}"
Q₁, c₁ = make_set_params(input_center, max_u; type=input_shapes) 
Q₂, c₂ = make_set_params(input_center, max_d; type=input_shapes) # 𝒰 & 𝒟
system, game = (A, B₁, B₂, Q₁, c₁, Q₂, c₂), "reach"

## Target
N, r = {cls.N}, {dynamics_data["goalR"]}
Q, center, radius = diagm(ones(N)), zeros(N), r
radius_N, Q_N = sqrt(N-1) * radius, diagm(vcat(1/(N-1), inv(1) * ones(N-1)))
target = make_target(center, radius_N; Q=Q_N, type="ellipse")

## Times
Th = {cls.ts}
times = collect(Th : Th : 1.);
th = min(1e-2, Th)

## Optimization Parameters
# vh, stepsz, tol, decay_stepsz, conv_runs_rqd, max_runs, max_its = 0.01, 1, 1e-3, 100, 1, 1, 100 # for N=100, gives MSE=0.85 & 27 min/60k
# vh, stepsz, tol, decay_stepsz, conv_runs_rqd, max_runs, max_its = 0.01, 1, 1e-3, 300, 1, 1, 300 # for N=100, gives MSE=0.25 & 78 min/60k
# vh, stepsz, tol, decay_stepsz, conv_runs_rqd, max_runs, max_its = 0.01, 1, 1e-3, 1000, 1, 1, 1000 # for N=100, gives MSE=0.13 & 253 min/60k
opt_p_cd = ({hopf_opt_p["vh"]}, {hopf_opt_p["stepsz"]}, {hopf_opt_p["tol"]}, {hopf_opt_p["decay_stepsz"]}, {hopf_opt_p["conv_runs_rqd"]}, {hopf_opt_p["max_runs"]}, {hopf_opt_p["max_its"]})

## Grad Reshape Fn
P_in_f(gradVX) = reshape(hcat(gradVX[2:end]...), size(gradVX[1])..., length(gradVX)-1)"""
            print(hopf_setup_exec)
            cls.jl.seval(hopf_setup_exec)

            solve_Hopf_BRS_exec = f"""\n
# Wrapper for Hopf Solver

function solve_Hopf_BRS(X_in; P_in=nothing, return_grad=false)
    println("Solving Hopf ... ")
    flush(log_f)
    (XsT, VXsT), run_stats, opt_data, gradVXsT = Hopf_BRS(system, target, times; X=Matrix(X_in), th, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, P_in, warm=true, warm_pattern="temporal", printing=false)
    println("Success!")
    flush(log_f)
    if return_grad
        return VXsT[1], vcat(VXsT[2:end]...), P_in_f(gradVXsT), run_stats[1]
    else
        return VXsT[1], vcat(VXsT[2:end]...), run_stats[1]
    end
end"""
            cls.solve_hopf_BRS = cls.jl.seval(solve_Hopf_BRS_exec)
            print(solve_Hopf_BRS_exec)
            cls.V_hopf = staticmethod(lambda tX: cls.solve_hopf_BRS(tX, return_grad=False))
            cls.V_hopf_ws = staticmethod(lambda tX, P_in: cls.solve_hopf_BRS(tX, P_in=P_in, return_grad=False))
            if solve_grad:
                cls.V_hopf_grad = staticmethod(lambda tX: cls.solve_hopf_BRS(tX, return_grad=True))
                cls.V_hopf_grad_ws = staticmethod(lambda tX, P_in: cls.solve_hopf_BRS(tX, P_in=P_in, return_grad=True))

        ## Interpolate the Dynamic Programming Solution
        if not cls.use_hopf or cls.gt_metrics:
            cls.jl.seval("using JLD, JLD2, Interpolations")

            llnd_path = "value_fns/LessLinear/"
            # V_itp = cls.jl.load(llnd_path + "interps/old/lin2d_hopf_interp_linear.jld")["V_itp"]
            cls.interp_file_name = f"LessLinear2D1i_interpolations_res1e-2_r{int(100 * dynamics_data['goalR_2d'])}e-2_c20.jld"
            V_DP_itp = cls.jl.load(llnd_path + f"interps/{cls.interp_file_name}", "LessLinear2D_interpolations")["g0_m0_a0"] # FIXME: flexible c param value

            fast_interp_exec = """\n
# Warpper for Interpolation

function fast_interp(_V_itp, tX_in; compute_grad=false)
    tX = Matrix(tX_in) # strange mp + torch + juliacall bug, only noticable here
    # print("Interpolating Xi across t... ")
    # flush(log_f)
    V = zeros(size(tX, 2))
    for i=1:length(V); V[i] = _V_itp(tX[:,i][end:-1:1]...); end # (assumes t in first row)
    J = zeros(size(tX, 2))
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

            def V_N_DP_linear_itp_combo(tXg):
                J = 0 * tXg[0,:]
                V = 0 * tXg[0,:]
                print("Interpolating DP solution ... ")
                for i in range(cls.N-1):
                    Ji, Vi = fast_interp(V_DP_itp, tXg[[0, 1, 2+i], :])
                    J, V = J+Ji, V+Vi
                print("Success.")
                return J, V

            def V_N_DP_itp_grad_combo(tXg):
                J = 0 * tXg[0,:]
                V = 0 * tXg[0,:]
                DV = 0 * tXg[1:,:].T
                print("Interpolating DP solution and gradients ... ")
                for i in range(self.N-1):
                    Ji, Vi, DVi = fast_interp(V_DP_itp, tXg[[0, 1, 2+i], :], compute_grad=True)
                    J, V = J+Ji, V+Vi
                    DV[:, [0, 1+i]] += DVi # assumes xN first
                print("Success.")
                return J, V, DV

            cls.V_hopf_gt = V_N_DP_linear_itp_combo
            cls.V_hopf_gt_grad = V_N_DP_itp_grad_combo

        return 

    @classmethod
    def solve_hopf(cls, Xi, bix, shm_data, tX_grad=None):

        split_size = int(Xi.shape[1] / cls.ts)
        job_id = int(bix/split_size)

        ## Point to Shared Memory
        shm_states_id, shm_states_shape, shm_algdat_id, shm_algdat_shape = shm_data
        shm_states = SharedMemory(name=shm_states_id)
        shm_algdat = SharedMemory(name=shm_algdat_id)
        bank = np.ndarray(shm_states_shape, dtype=np.float32, buffer=shm_states.buf)
        alg_data = np.ndarray(shm_algdat_shape, dtype=np.float32, buffer=shm_algdat.buf)

        if not cls.use_hopf or cls.gt_metrics:
            tXi = np.hstack([np.vstack(((j+1) * cls.ts * np.ones((1, Xi.shape[1])), Xi)) for j in range(cls.tp)])

        print(f"\n\n ########### Worker {cls.worker_id}, job {job_id} ###########\n\n")
        print(f"Solving {int(Xi.shape[1] * cls.tp)} points ({Xi.shape[1]} spatial, {cls.tp} time).")

        ## Compute Value (and Gradient) with Hopf, w/w/o Some Gradients for Warm-Starting Hopf
        if cls.use_hopf:
            if cls.solve_grad:
                if tX_grad is not None:
                    print("\nWarm-starting the hopf solve with gradient data.")
                    J, V, DxV, comp_time = cls.V_hopf_grad_ws(Xi, tX_grad)
                else:
                    J, V, DxV, comp_time = cls.V_hopf_grad(Xi)
            else:
                if tX_grad is not None:
                    print("\nWarm-starting the hopf solve with gradient data.")
                    J, V, comp_time = cls.V_hopf_grad(Xi, tX_grad)
                else:
                    J, V, comp_time = cls.V_hopf(Xi)

        ## Compute Value with Composed 2D Interpolation (for testing)
        else: 
            print(f"\nComputing value with DP-Interpolations (not solving hopf), from {cls.interp_file_name}")
            if cls.solve_grad:
                J, V, DxV = cls.V_hopf_gt_grad(tXi)
            else:
                J, V = cls.V_hopf_gt(tXi)

        mean_time = comp_time / split_size

        if cls.use_hopf:
            J = np.repeat(J, int(1/cls.ts))
            # V = np.hstack(V)
            if cls.solve_grad:
                DxV = np.reshape(DxV, (cls.N, split_size)).T()

            ## Solve Ground Truth
            if cls.gt_metrics:
                print(f"\nComputing value with DP-Interpolations for ground truth, from {cls.interp_file_name}")
                if cls.solve_grad:
                    J_gt, V_gt, DxV_gt = cls.V_hopf_gt_grad(tXi)
                else:
                    J_gt, V_gt = cls.V_hopf_gt(tXi)

                SE = np.power((V_gt - V), 2)
                MSE = SE.mean()

                if cls.solve_grad:
                    SE_grad = np.power((DxV_gt - DxV), 2).mean(dim=1) # mse per pt
                    MSE_grad = SE_grad.mean()
        
        print(f"\n Batch had accuracy of {MSE} and took {mean_time} s/pt")

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
                if cls.solve_grad:
                    alg_data[job_id, 4] = MSE_grad
            
            ## Write Job Log to Master Log
            with open(cls.worker_log_full, 'r') as wlog, open(os.path.join(cls.log_loc, cls.master_log), 'a') as mlog:
                shutil.copyfileobj(wlog, mlog)
        
        # Delete job log and restore Julia output
        # try:
        #     os.remove(cls.worker_log_full)
        # except OSError as e:
        #     print(f"Error deleting worker log {cls.worker_log_full}: {e}")
        # jl.seval(f"""
        #     redirect_stdout(stdout)
        # """)
        # cls.worker_log_full.close()

        return (job_id, mean_time)

    def solve_bank_start(self, bank_params, n_splits=10):
        
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
        print(f"\nSolving {bank_start} points to start the bank (in {n_splits} jobs for {self.num_hopf_workers} workers), composed of {total_spatial_pts} spatial x {self.tp} time pts ({split_size} per job).\n")
        if bank_start % split_spatial_pts != 0: raise AssertionError(f"Your bank isn't divided well ({n_splits} splits gives {bank_start % split_spatial_pts} pts/split); change your bank total or time-step") 

        with tqdm(total=n_splits) as pbar:
            
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
                job = self.pool.apply_async(self.solve_hopf, (Xi, i*split_size, shm_data))
                self.jobs.append(job)

            # for job in self.jobs:
            #     job_id, mean_time = job.get()
            #     # TODO: in invst, will store alg_data[job_id, 0] = self.alg_iter (automatically)
            #     pbar.update(1)
            
            while self.jobs:
                for job in self.jobs:
                    if job.ready():
                        pbar.update(1)
                        self.jobs.remove(job)
        
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

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    # mp.log_to_stderr(logging.DEBUG)

    dynamics = LessLinearND(7, 0., 0., 0.)
    dynamics_data = {param:getattr(dynamics, param) for param in dir(dynamics) if not (param.startswith('__') or param.startswith('_')) and not callable(getattr(dynamics, param))}
    dynamics_data = {key:val.cpu() if torch.is_tensor(val) else val for key,val in dynamics_data.items()}

    # time_step = 1e-3
    time_step = 1e-1
    hopf_opt_p = {"vh":0.01, "stepsz":1, "tol":1e-3, "decay_stepsz":100, "conv_runs_rqd":1, "max_runs":1, "max_its":100} 

    hjpool = HopfJuliaPool(dynamics_data, time_step, hopf_opt_p,
                            use_hopf=True, solve_grad=False, hopf_warm_start=False, gt_metrics=True, num_hopf_workers=4)

    # bank_params = {"bank_total":200000, "bank_start":100000, "bank_invst":10000}
    bank_params = {"bank_total":200, "bank_start":100, "bank_invst":10}
    shm_data = hjpool.solve_bank_start(bank_params, n_splits=10)
    
    shm_states_id, shm_states_shape, shm_algdat_id, shm_algdat_shape = shm_data
    shm_states, shm_algdat = SharedMemory(name=shm_states_id), SharedMemory(name=shm_algdat_id)
    bank = np.ndarray(shm_states_shape, dtype=np.float32, buffer=shm_states.buf)
    alg_data = np.ndarray(shm_algdat_shape, dtype=np.float32, buffer=shm_algdat.buf)

    print("He hecho\n")
            
