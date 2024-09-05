
from multiprocessing import Pool, Lock
from multiprocessing.shared_memory import SharedMemory
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import numpy as np
import torch

## Multi-Processing

class JuliaPool(object):

    jl = None
    V_linear = None
    # shared_memory_name = None
    # shared_shape = None
    lock = None

    def __init__(self):
        pass

    @classmethod
    def init_worker(cls):

        print('Initializing a process...')

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
        fast_interp = jl.seval(fast_interp_exec)
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


juliapool = JuliaPool()

juliapool.test1()

juliapool.test2()

print("he hecho")

