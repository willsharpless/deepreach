
###### Linear Supervision for Nonlinear, High-Dimensional Neural Control and Differential Games
###### Sharpless (wsharpless@ucsd.edu), Feng, Bansal, Herbert - 2024, L4DC 
###### Benchmark Script

## Init

readonly N="3"
readonly base_args="--dynamics_class LessLinearND --N $N --goalR 0.25 --solve_grad"
readonly lin_sys_args="--gamma 0 --mu 0 --alpha 0"
readonly nlin_sys_args="--gamma 20 --mu 0 --alpha 0"

readonly use_wandb="--use_wandb" # "" if you dont want wandb
readonly wandb_project="deepreach_comparisons"

readonly dr_names="--experiment_name L${N}D_DR $use_wandb --wandb_project $wandb_project --wandb_name L${N}D_DR"
readonly fd_names="--experiment_name L${N}D_FD $use_wandb --wandb_project $wandb_project --wandb_name L${N}D_FD"

readonly seeds=3

## Solve Linear Model

# Finite Differencing

fd_as="--fd_as 2.5 2. 1.5 1."
fd_dxs="--fd_dxs 0.7 0.5 0.3 0.1"
fd_dts="--fd_dts 0.05 0.03 0.02 0.01"

for i in $(seq 1 $seeds); do
    echo run_experiment.py $base_args $lin_sys_args --fin_diff $fd_as $fd_dxs $fd_dts --num_epochs 3000 --numpoints 1000 --lr_std 1e-2 $fd_names --seed $i

    echo run_experiment.py $base_args $lin_sys_args --fin_diff $fd_as $fd_dxs $fd_dts --num_epochs 3000 --numpoints 1000 --lr_std 1e-3 $fd_names --seed $i

    echo run_experiment.py $base_args $lin_sys_args --fin_diff $fd_as $fd_dxs $fd_dts --num_epochs 30000 --numpoints 1000 --lr_std 1e-5 $fd_names --seed $i

    echo run_experiment.py $base_args $lin_sys_args --fin_diff $fd_as $fd_dxs $fd_dts --num_epochs 30000 --numpoints 1000 --lr_std 1e-6 $fd_names --seed $i

    echo run_experiment.py $base_args $lin_sys_args --fin_diff $fd_as $fd_dxs $fd_dts --num_epochs 30000 --numpoints 10000 --lr_std 1e-5 $fd_names --seed $i

    echo run_experiment.py $base_args $lin_sys_args --fin_diff $fd_as $fd_dxs $fd_dts --num_epochs 30000 --numpoints 10000 --lr_std 1e-6 $fd_names --seed $i
done

# DeepReach/PINN baseline

for i in $(seq 1 $seeds); do
    echo run_experiment.py $base_args $lin_sys_args --num_epochs 30000 --numpoints 65000 --lr_std 1e-5 --baseline --counter_end 20000 $dr_names --seed $i
done

# ## Solve Nonlinear Model 

# # DeepReach/PINN baseline
# echo run_experiment.py $base_args $nlin_sys_args --baseline --num_epochs 100000 --experiment_name LL${N}D_DR $use_wandb --wandb_project $wandb_project --wandb_name LL${N}D_DR

# # Finite Differencing
# echo run_experiment.py $base_args $nlin_sys_args --num_epochs 100000 --experiment_name LL${N}D_FD $use_wandb --wandb_project $wandb_project --wandb_name LL${N}D_FD
