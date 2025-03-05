
###### Linear Supervision for Nonlinear, High-Dimensional Neural Control and Differential Games
###### Sharpless (wsharpless@ucsd.edu), Feng, Bansal, Herbert - 2024, L4DC 
###### Benchmark Script

## Init

readonly N="10"
readonly base_args="--dynamics_class LessLinearND --N $N --goalR 0.25 --solve_grad --epochs_til_ckpt 500" # --deepreach_model vanilla (made worse)
readonly lin_sys_args="--gamma 0 --mu 0 --alpha 0"
readonly nlin_sys_args="--gamma 20 --mu 0 --alpha 0"

readonly use_wandb="--use_wandb" # "" if you dont want wandb
readonly wandb_project="deepreach_comparisons"

readonly dr_names="--experiment_name L${N}D_DR $use_wandb --wandb_project $wandb_project" # overwriting locally (not enough space)
readonly fd_names="--experiment_name L${N}D_FD $use_wandb --wandb_project $wandb_project" # overwriting locally (not enough space)

readonly seeds=1
# readonly act="sine" #relu was sig worse in 3D

## Finite Differencing

# base params (base)
fd_as="--fd_as 2. 1.5 1."
fd_dxs="--fd_dxs 0.025 0.01 0.001"
fd_dts="--fd_dts 0.03 0.02 0.01"

# aggresive params (agg)
fd_as="--fd_as 1.5 1. 0.5"
fd_dxs="--fd_dxs 0.01 0.005 0.001"
fd_dts="--fd_dts 0.01 0.005 0.001"

lr="--lr_std 1e-5"

for i in $(seq 1 $seeds); do

    ## Linear
    python run_experiment.py $base_args $lin_sys_args $lr --fin_diff $fd_as $fd_dxs $fd_dts --num_epochs 30000 --numpoints 1000 --no_curr $fd_names --wandb_name L${N}D_FD_base_bs1k --seed $i
    python run_experiment.py $base_args $lin_sys_args $lr --fin_diff $fd_as $fd_dxs $fd_dts --num_epochs 30000 --numpoints 1000 --no_curr $fd_names --wandb_name L${N}D_FD_agg_bs1k --seed $i

    python run_experiment.py $base_args $lin_sys_args $lr --fin_diff $fd_as $fd_dxs $fd_dts --num_epochs 30000 --numpoints 3000 --no_curr $fd_names --wandb_name L${N}D_FD_base_bs3k --seed $i
    python run_experiment.py $base_args $lin_sys_args $lr --fin_diff $fd_as $fd_dxs $fd_dts --num_epochs 30000 --numpoints 3000 --no_curr $fd_names --wandb_name L${N}D_FD_agg_bs3k --seed $i

    python run_experiment.py $base_args $lin_sys_args $lr --fin_diff $fd_as $fd_dxs $fd_dts --num_epochs 30000 --numpoints 5000 --no_curr $fd_names --wandb_name L${N}D_FD_base_bs5k --seed $i
    python run_experiment.py $base_args $lin_sys_args $lr --fin_diff $fd_as $fd_dxs $fd_dts --num_epochs 30000 --numpoints 5000 --no_curr $fd_names --wandb_name L${N}D_FD_agg_bs5k --seed $i

    python run_experiment.py $base_args $lin_sys_args $lr --fin_diff $fd_as $fd_dxs $fd_dts --num_epochs 30000 --numpoints 10000 --no_curr $fd_names --wandb_name L${N}D_FD_base_bs10k --seed $i
    python run_experiment.py $base_args $lin_sys_args $lr --fin_diff $fd_as $fd_dxs $fd_dts --num_epochs 30000 --numpoints 10000 --no_curr $fd_names --wandb_name L${N}D_FD_agg_bs10k --seed $i

done

# for i in $(seq 1 $seeds); do

#     ## Linear
#     python run_experiment.py $base_args $lin_sys_args $lr --fin_diff $fd_as $fd_dxs $fd_dts --num_epochs 30000 --numpoints 10000 --no_curr $fd_names --wandb_name L${N}D_FD_base --seed $i
#     python run_experiment.py $base_args $lin_sys_args $lr --fin_diff $fd_as $fd_dxs $fd_dts --num_epochs 30000 --numpoints 10000 --counter_end 20000 $fd_names --wandb_name L${N}D_FD_base_c --seed $i

#     python run_experiment.py $base_args $lin_sys_args $lr --fin_diff $fd_as $fd_dxs $fd_dts --num_epochs 30000 --numpoints 10000 --no_curr $fd_names --wandb_name L${N}D_FD_agg --seed $i
#     python run_experiment.py $base_args $lin_sys_args $lr --fin_diff $fd_as $fd_dxs $fd_dts --num_epochs 30000 --numpoints 10000 --counter_end 20000 $fd_names --wandb_name L${N}D_FD_agg_c --seed $i

#     ## Nonlinear
#     python run_experiment.py $base_args $nlin_sys_args $lr --fin_diff $fd_as $fd_dxs $fd_dts --num_epochs 100000 --numpoints 10000 --no_curr $fd_names --wandb_name NL${N}D_FD_base --seed $i
#     python run_experiment.py $base_args $nlin_sys_args $lr --fin_diff $fd_as $fd_dxs $fd_dts --num_epochs 100000 --numpoints 10000 --counter_end 80000 $fd_names --wandb_name NL${N}D_FD_base_c --seed $i

#     python run_experiment.py $base_args $nlin_sys_args $lr --fin_diff $fd_as $fd_dxs $fd_dts --num_epochs 100000 --numpoints 10000 --no_curr $fd_names --wandb_name NL${N}D_FD_agg --seed $i
#     python run_experiment.py $base_args $nlin_sys_args $lr --fin_diff $fd_as $fd_dxs $fd_dts --num_epochs 100000 --numpoints 10000 --counter_end 80000 $fd_names --wandb_name NL${N}D_FD_agg_c --seed $i

# done

## DeepReach (PINN) baseline

for i in $(seq 1 $seeds); do

    ## Linear
    python run_experiment.py $base_args $lin_sys_args $lr --baseline --num_epochs 30000 --numpoints 65000 --counter_end 20000 $dr_names --wandb_name L${N}D_DR --seed $i
    
    ## Nonlinear
    python run_experiment.py $base_args $nlin_sys_args $lr --baseline --num_epochs 100000 --numpoints 65000 --counter_end 80000 $dr_names --wandb_name NL${N}D_DR --seed $i

done

# ## Solve Nonlinear Model 

# # DeepReach/PINN baseline
# python run_experiment.py $base_args $nlin_sys_args --baseline --num_epochs 100000 --experiment_name LL${N}D_DR $use_wandb --wandb_project $wandb_project

# # Finite Differencing
# python run_experiment.py $base_args $nlin_sys_args --num_epochs 100000 --experiment_name LL${N}D_FD $use_wandb --wandb_project $wandb_project --wandb_name LL${N}D_FD
