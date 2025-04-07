#!/bin/bash

pip install -r requirements.txt

pip install juliacall

python3 -c '
from juliacall import Main as jl

jl.seval("import Pkg")
jl.seval("""Pkg.add(["LinearAlgebra", "Interpolations", "StatsBase", "TickTock", "Suppressor", "JLD", "JLD2", "Plots", "ScatteredInterpolation", "Contour"])""")
jl.seval("""Pkg.precompile()""")
'

git clone https://github.com/UCSD-SASLab/HopfReachability.git