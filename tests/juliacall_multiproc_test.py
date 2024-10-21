
from multiprocessing import Pool, Lock
from multiprocessing.shared_memory import SharedMemory
import multiprocessing as mp

from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import numpy as np
import torch
import warnings

## Multi-Processing

class JuliaPool(object):

    jl = None
    V_linear = None
    solve_hopf = None
    # shared_memory_name = None
    # shared_shape = None
    lock = None

    def __init__(self):
        pass

    @classmethod
    def init_worker(cls):

        print('Initializing a process...')

        warnings.filterwarnings("ignore", category=UserWarning, message="torch was imported before juliacall. This may cause a segfault. To avoid this, import juliacall before importing torch. For updates, see https://github.com/pytorch/pytorch/issues/78829.")
        from juliacall import Main as jl, convert as jlconvert

        cls.jl = jl
        cls.jl.seval("using JLD, JLD2, Interpolations")

        llnd_path = "value_fns/LessLinear/"
        V_itp = cls.jl.load(llnd_path + "interps/old/lin2d_hopf_interp_linear.jld")["V_itp"]

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
        fast_interp = cls.jl.seval(fast_interp_exec)
        cls.V_linear = lambda tXg: torch.from_numpy(fast_interp(V_itp, tXg.numpy()).to_numpy())

        # cls.shared_name, cls.shm_shape, cls.lock = shared_data
        cls.lock = Lock()

    @classmethod
    def compute1(cls, jobid, tXg):
        # print(f'in main({jobid})...')

        # if not cls.V_linear:
        #     print(f'({jobid}): worker being initialized')
        #     cls.init_worker()
        # V_linear(tXg)

        cls.V_linear(tXg)
        start = time.time()
        for _ in range(3):
            cls.V_linear(tXg) # same as above
        end = time.time()
        print(f"({jobid}): JuliaCall 65k interp took {(end - start)/5}s")

        return
    
    def test1(self):

        n = 65000
        tXg = torch.zeros((3, n))
        njobs = 10
        workers = 2

        with Pool(workers, initializer=self.init_worker) as p, tqdm(total=njobs) as pbar:
            jobs = []
            for jid in range(njobs):
                jobs.append(p.apply_async(self.compute1, (jid, tXg)))
            # for job in jobs:
            #     job.get()
            #     pbar.update(1)

            steps = 0
            while jobs:
                steps += 1
                for job in jobs:
                    if job.ready():
                        pbar.update(1)
                        jobs.remove(job)

            print(f"Took {steps} steps while jobs were running")
    
    @classmethod
    def compute2(cls, jid, tXg, shared_data):

        shm_name, shm_shape = shared_data
        shm = SharedMemory(name=shm_name)
        Vr = np.ndarray(shm_shape, dtype=np.float32, buffer=shm.buf)

        tblen = tXg.shape[1]
        tmp = cls.V_linear(tXg).numpy()
        with cls.lock:
            Vr[jid * tblen : (jid + 1) * tblen] = tmp
        shm.close()

        return

    def test2(self):

        n = 65000
        
        Xr = np.random.uniform(low=-1., size=(2,n))
        t5 = [0., .25, .5, .75, 1.]
        ts = np.concatenate([t * np.ones(int(n/5)) for t in t5]).reshape((1, n))
        tXr = torch.from_numpy(np.concatenate((ts, Xr), axis=0))

        shm_shape = (n,)
        shm = SharedMemory(create=True, size=np.prod(shm_shape) * np.dtype(np.float32).itemsize)
        shared_data = (shm.name, shm_shape)

        Vr = np.ndarray(shm_shape, dtype=np.float32, buffer=shm.buf)
        Vr[:] = 1.

        njobs = len(t5)
        tblen = int(n/5)
        workers = 2

        with Pool(workers, initializer=self.init_worker) as p, tqdm(total=njobs) as pbar:
            jobs = []
            for jid in range(njobs):
                job = p.apply_async(self.compute2, (jid, tXr[:, jid * tblen : (jid + 1) * tblen], shared_data))
                jobs.append(job)

            while jobs:
                for job in jobs[:]:
                    if job.ready():
                        pbar.update(1)
                        jobs.remove(job)
        
        Xr_near = Xr[:, np.abs(Vr) < .005]
        plt.figure()
        plt.scatter(Xr_near[0,:], Xr_near[1,:])
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.savefig("interp_test_multiprocess.png")

        shm.close()
        shm.unlink()

    @classmethod
    def compute_hopf_BRS(cls):

        # warnings.filterwarnings("ignore", category=UserWarning, message="torch was imported before juliacall. This may cause a segfault. To avoid this, import juliacall before importing torch. For updates, see https://github.com/pytorch/pytorch/issues/78829.")
        # from juliacall import Main as jl, convert as jlconvert
        import os

        pid = os.getpid()
        print(pid, " job has been started")

        exec_print = f"""
# Set Output

global log_f = open("juliacall_hopfBRS_test_{pid}.log", "a")
redirect_stdout(log_f)
redirect_stderr(log_f)""" ## at least see partial Hopf_BRS print (until it presumably errors)
        cls.jl.seval(exec_print)

        exec_load = f"""

# Load

using Pkg
Pkg.activate()

using LinearAlgebra

include(pwd() * "/HopfReachability/src/HopfReachability.jl");
using .HopfReachability: Hopf_BRS, Hopf_cd, make_grid, make_target, make_set_params"""
        cls.jl.seval(exec_load)
        print(exec_load)

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
Th = 0.001
times = collect(Th : Th : 1.);
th = min(1e-2, Th)

## Optimization Parameters
# vh, stepsz, tol, decay_stepsz, conv_runs_rqd, max_runs, max_its = 0.01, 1, 1e-3, 100, 1, 1, 100 # for N=100, gives MSE=0.85 & 27 min/60k
# vh, stepsz, tol, decay_stepsz, conv_runs_rqd, max_runs, max_its = 0.01, 1, 1e-3, 300, 1, 1, 300 # for N=100, gives MSE=0.25 & 78 min/60k
# vh, stepsz, tol, decay_stepsz, conv_runs_rqd, max_runs, max_its = 0.01, 1, 1e-3, 1000, 1, 1, 1000 # for N=100, gives MSE=0.13 & 253 min/60k
opt_p_cd = (0.01, 1, 0.001, 100, 1, 1, 100)

## Grad Reshape Fn
P_in_f(gradVX) = reshape(hcat(gradVX[2:end]...), size(gradVX[1])..., length(gradVX)-1)"""
        cls.jl.seval(hopf_setup_exec)
        print(hopf_setup_exec)

        hopf_BRS_exec = """
# Wrapper for Hopf Solver

function solve_Hopf_BRS()
    X = Float32[-0.11893158 0.038724087 0.40518385 0.27271906 0.66952044; -0.91246986 0.6038015 0.6672152 0.49143624 0.75128675; 0.6455725 -0.22062263 0.839456 -0.74276614 -0.58126867; 0.7025011 -0.270896 0.4259202 0.077222735 -0.3687684; 0.29084364 -0.67864865 0.46917862 0.5430416 0.7472049; -0.0019951398 -0.890628 -0.8739691 0.4054919 -0.11211838; 0.7379937 0.010344575 0.63341236 0.03499771 -0.82848865]
    P_in = nothing
    return_grad=false
    println("About to solve hopf with")
    println("VERSION", VERSION)
    println(Sys.BINDIR)
    println(Pkg.envdir())
    println("X:")
    print(X)
    println("")
    println("return_grad=", return_grad)
    flush(log_f)
    (XsT, VXsT), run_stats, opt_data, gradVXsT = Hopf_BRS(system, target, times; X, th, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, P_in, warm=true, warm_pattern="temporal", printing=true)
    println("solved hopf")
    flush(log_f)
    if return_grad
        return VXsT[1], vcat(VXsT[2:end]...), P_in_f(gradVXsT)
    else
        return VXsT[1], vcat(VXsT[2:end]...)
    end
end"""        
        cls.solve_hopf = cls.jl.seval(hopf_BRS_exec)
        print(hopf_BRS_exec)

        result = cls.solve_hopf()
        return result

    def test_hopf_BRS(self):

        njobs = 1
        workers = 1

        with Pool(workers, initializer=self.init_worker) as p, tqdm(total=njobs) as pbar:
            jobs = []
            for jid in range(njobs):
                # job = p.apply_async(self.compute2, (jid, tXr[:, jid * tblen : (jid + 1) * tblen], shared_data))
                job = p.apply_async(self.compute_hopf_BRS)
                jobs.append(job)

            while jobs:
                for job in jobs[:]:
                    if job.ready():
                        print("\n")
                        print(job.get())
                        print("\n")
                        pbar.update(1)
                        jobs.remove(job)

    def solve_bank_start(self, dynamics_data, time_step, hopf_opt_p, bank_params, 
                solve_grad=False, use_hopf=False, hopf_warm_start=False, gt_metrics=False, 
                n_splits=1):

        # n = 65000
        
        # Xr = np.random.uniform(low=-1., size=(2,n))
        # t5 = [0., .25, .5, .75, 1.]
        # ts = np.concatenate([t * np.ones(int(n/5)) for t in t5]).reshape((1, n))
        # tXr = torch.from_numpy(np.concatenate((ts, Xr), axis=0))

        # shm_shape = (n,)
        # shm = SharedMemory(create=True, size=np.prod(shm_shape) * np.dtype(np.float32).itemsize)
        # shared_data = (shm.name, shm_shape)

        # Vr = np.ndarray(shm_shape, dtype=np.float32, buffer=shm.buf)
        # Vr[:] = 1.

        # njobs = len(t5)
        njobs = 1
        # tblen = int(n/5)
        # workers = 2
        workers = 1
        p = Pool(workers, initializer=self.init_worker)

        with tqdm(total=njobs) as pbar:
            jobs = []
            for jid in range(njobs):
                # job = p.apply_async(self.compute2, (jid, tXr[:, jid * tblen : (jid + 1) * tblen], shared_data))
                job = p.apply_async(self.compute_hopf_BRS)
                jobs.append(job)

            while jobs:
                for job in jobs[:]:
                    if job.ready():
                        print("\n")
                        print(job.get())
                        print("\n")
                        pbar.update(1)
                        jobs.remove(job)

from dynamics.dynamics import LessLinearND
# import logging

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    juliapool = JuliaPool()

    # juliapool.test1()

    # juliapool.test2()

    # result = juliapool.compute_hopf_BRS()
    # import pickle
    # pickle.dumps(result)

    juliapool.test_hopf_BRS()

    print("he hecho")

