#!/bin/bash

pip install -r requirements.txt

pip install juliacall

python3 -c '
from juliacall import Main

Main.eval("""
import Pkg
Pkg.add(["LinearAlgebra", "StatsBase", "TickTock", "Suppressor", 
         "Plots", "ScatteredInterpolation", "Contour"])
Pkg.precompile()
""")
'

git clone https://github.com/UCSD-SASLab/HopfReachability.git