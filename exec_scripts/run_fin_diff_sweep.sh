
###### Linear Supervision for Nonlinear, High-Dimensional Neural Control and Differential Games
###### Sharpless (wsharpless@ucsd.edu), Feng, Bansal, Herbert - 2024, L4DC 
###### Benchmark Script

## Init

readonly N="10"
readonly base_args="--dynamics_class LessLinearND --N $N --goalR 0.25 --solve_grad --epochs_til_ckpt 500" #--minWith none --deepreach_model vanilla (neither helps)
readonly lin_sys_args="--gamma 0 --mu 0 --alpha 0"
readonly nlin_sys_args="--gamma 20 --mu 0 --alpha 0"

readonly use_wandb="--use_wandb" # "" if you dont want wandb
readonly wandb_project="deepreach_comparisons"

readonly dr_names="--experiment_name L${N}D_DR $use_wandb --wandb_project $wandb_project" # overwriting locally (not enough space)
readonly fd_names="--experiment_name L${N}D_FD $use_wandb --wandb_project $wandb_project" # overwriting locally (not enough space)

readonly seeds=2

## Solve Linear Model

# Finite Differencing
for a in $(seq 2 2); do

    if [ $a == 1 ]; then
        arch="sine"
    elif [ $a == 2 ]; then
        arch="relu"
    else
        arch="tanh"
    fi

    for i in $(seq 1 $seeds); do

        # base params (base)
        fd_as="--fd_as 2. 1.5 1."
        fd_dxs="--fd_dxs 0.025 0.01 0.001"
        fd_dts="--fd_dts 0.03 0.02 0.01"
        lr="--lr_std 1e-5"

        # python run_experiment.py $base_args $lin_sys_args $lr --fin_diff $fd_as $fd_dxs $fd_dts --num_epochs 30500 --numpoints 10000 --counter_end 20000 $fd_names --wandb_name L${N}D_FD_${arch}_base_c2 --model $arch --seed $i
        python run_experiment.py $base_args $lin_sys_args $lr --fin_diff $fd_as $fd_dxs $fd_dts --num_epochs 30500 --numpoints 10000 --no_curr $fd_names --wandb_name L${N}D_FD_${arch}_base2 --model $arch --seed $i

        # decreased alpha (lila)
        fd_as="--fd_as 1.5 1. 0.5"
        fd_dxs="--fd_dxs 0.025 0.01 0.001"
        fd_dts="--fd_dts 0.03 0.02 0.01"
        lr="--lr_std 1e-5"

        # python run_experiment.py $base_args $lin_sys_args $lr --fin_diff $fd_as $fd_dxs $fd_dts --num_epochs 30500 --numpoints 10000 --counter_end 20000 $fd_names --wandb_name L${N}D_FD_${arch}_lila_c2 --model $arch --seed $i
        python run_experiment.py $base_args $lin_sys_args $lr --fin_diff $fd_as $fd_dxs $fd_dts --num_epochs 30500 --numpoints 10000 --no_curr $fd_names --wandb_name L${N}D_FD_${arch}_lila2 --model $arch --seed $i

        # decreased dx (lilx)
        fd_as="--fd_as 2. 1.5 1."
        fd_dxs="--fd_dxs 0.01 0.005 0.001"
        fd_dts="--fd_dts 0.03 0.02 0.01"
        lr="--lr_std 1e-5"

        # python run_experiment.py $base_args $lin_sys_args $lr --fin_diff $fd_as $fd_dxs $fd_dts --num_epochs 30500 --numpoints 10000 --counter_end 20000 $fd_names --wandb_name L${N}D_FD_${arch}_lilx_c2 --model $arch --seed $i
        python run_experiment.py $base_args $lin_sys_args $lr --fin_diff $fd_as $fd_dxs $fd_dts --num_epochs 30500 --numpoints 10000 --no_curr $fd_names --wandb_name L${N}D_FD_${arch}_lilx2 --model $arch --seed $i

        # decreased dt (lilt)
        fd_as="--fd_as 2. 1.5 1."
        fd_dxs="--fd_dxs 0.025 0.01 0.001"
        fd_dts="--fd_dts 0.01 0.005 0.001"
        lr="--lr_std 1e-5"

        # python run_experiment.py $base_args $lin_sys_args $lr --fin_diff $fd_as $fd_dxs $fd_dts --num_epochs 30500 --numpoints 10000 --counter_end 20000 $fd_names --wandb_name L${N}D_FD_${arch}_lilt_c2 --model $arch --seed $i
        python run_experiment.py $base_args $lin_sys_args $lr --fin_diff $fd_as $fd_dxs $fd_dts --num_epochs 30500 --numpoints 10000 --no_curr $fd_names --wandb_name L${N}D_FD_${arch}_lilt2 --model $arch --seed $i

    done
done

# DeepReach/PINN baseline

# for a in $(seq 1 1); do

#     if [ $a == 1 ]; then
#         arch="sine"
#     elif [ $a == 2 ]; then
#         arch="relu"
#     else
#         arch="tanh"
#     fi

#     for i in $(seq 1 $seeds); do
#         python run_experiment.py $base_args $lin_sys_args --num_epochs 30000 --numpoints 65000 --lr_std 1e-5 --baseline --counter_end 20000 $dr_names --wandb_name L${N}D_DR_${arch} --seed $i --model $arch
#     done
# done

# ## Solve Nonlinear Model 

# # DeepReach/PINN baseline
# python run_experiment.py $base_args $nlin_sys_args --baseline --num_epochs 100000 --experiment_name LL${N}D_DR $use_wandb --wandb_project $wandb_project

# # Finite Differencing
# python run_experiment.py $base_args $nlin_sys_args --num_epochs 100000 --experiment_name LL${N}D_FD $use_wandb --wandb_project $wandb_project --wandb_name LL${N}D_FD
