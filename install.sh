#!/bin/bash

pip install -r requirements.txt

pip install hj_reachability

# Find the path to hj_reachability package
PKG_PATH=$(python -c "import hj_reachability, os; print(os.path.dirname(hj_reachability.__file__))")

# Replace grid.py with custom version
cp ./utils/hj_reachability_grid_custom.py "$PKG_PATH/grid.py"

# pip install juliacall

# python3 -c '
# from juliacall import Main

# Main.eval("""
# import Pkg
# Pkg.add(["LinearAlgebra", "StatsBase", "TickTock", "Suppressor", 
#          "Plots", "ScatteredInterpolation", "Contour"])
# Pkg.precompile()
# """)
# '

# git clone https://github.com/UCSD-SASLab/HopfReachability.git