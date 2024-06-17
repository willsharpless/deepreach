
import numpy as np
from juliacall import Main as jl, convert as jlconvert
import torch

# julia_path = jl.seval("Sys.BINDIR")
# print(f"Julia executable directory: {julia_path}")

# jl.seval("using Pkg")
# Pkg.add("JLD, JLD2, Interpolations")

jl.seval("using JLD2, Interpolations")

V_itp = jl.load("lin2d_hopf_interp_linear.jld")["V_itp"]

n = 65000
# tXg = np.zeros((3, n))
tXg = torch.zeros((3, n))

def py_interp(_V_itp, _tXg):
    _tXg = _tXg.numpy()
    Vg = np.zeros(_tXg.shape[1])
    for i in range(_tXg.shape[1]):
        Vg[i] = _V_itp(_tXg[2, i], _tXg[1, i], _tXg[0, i])
    return Vg

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

fast_interp = jl.seval(fast_interp_exec)

import time

# way faster
fast_interp(V_itp, tXg.numpy()) # JIT compile
start = time.time()
for _ in range(10):
    # fast_interp(V_itp, tXg)
    # fast_interp(V_itp, tXg).to_numpy() # adds 40ms smh
    torch.from_numpy(fast_interp(V_itp, tXg.numpy()).to_numpy()) # torch conv v fast atleast
    # Chat thinks I should use Torch.jl, which allows direct conv (will save ~70 ms/it)
end = time.time()
print("JuliaCall 65k interp call time:", (end - start)/5)

start = time.time()
for _ in range(5):
    py_interp(V_itp, tXg)
end = time.time()
print("Py-wrapper 65k interp call time:", (end - start)/5)

import matplotlib.pyplot as plt

Xr = np.random.uniform(low=-1., size=(2,n))
t5 = [0., .25, .5, .75, 1.]
ts = np.concatenate([t * np.ones(int(n/5)) for t in t5]).reshape((1, n))
tXr = np.concatenate((ts, Xr), axis=0)

Vr = fast_interp(V_itp, tXr).to_numpy()
Xr_near = Xr[:, np.abs(Vr) < .005]

plt.figure()
plt.scatter(Xr_near[0,:], Xr_near[1,:])
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.savefig("interp_test.png")

print("he hecho")

