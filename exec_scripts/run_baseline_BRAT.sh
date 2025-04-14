#!/bin/bash

## Init

base_params="--tMax 2."
readonly convr_dyn="--dynamics_class ConveyorND $base_params"
readonly canoe_dyn="--dynamics_class CanoeND $base_params"
readonly convr_lam_dyn="--dynamics_class ConveyorNDlambda $base_params"
readonly canoe_lam_dyn="--dynamics_class CanoeNDlambda $base_params"

# readonly use_wandb=""
readonly use_wandb="--use_wandb"
readonly wandb_project="mulob_baseline_ND"
readonly wandb_parms="$use_wandb --wandb_project $wandb_project"

readonly exp_dir_test="--experiments_dir ./runs/mulob/test"
readonly exp_dir_convr="--experiments_dir ./runs/mulob/ConveyorND/decomposedND/"
readonly exp_dir_canoe="--experiments_dir ./runs/mulob/CanoeND/decomposedND"

## Conveyor

i_max="10"
dims=()
for ((i=1; i<=${i_max}; i++)); do
    dims+=($((2 ** i)))
done

# for j in $(seq 1 1); do
    # for i in $(seq 1 10); do

# for i in $(seq 2 7); do
for i in $(seq 1 10); do
    for j in $(seq 2 2); do

        if [ $j == 1 ]; then
            learn_params="--lr_std 2e-5 --pretrain_iters 500  --num_epochs 5000   --counter_end 500"
        elif [ $j == 2 ]; then
            learn_params="--lr_std 5e-6 --pretrain_iters 1000 --num_epochs 50000  --counter_end 1000"
        elif [ $j == 3 ]; then
            learn_params="--lr_std 2e-6 --pretrain_iters 2000 --num_epochs 100000 --counter_end 5000"
        elif [ $j == 4 ]; then
            learn_params="--lr_std 1e-6 --pretrain_iters 5000 --num_epochs 200000 --counter_end 10000" 
        elif [ $j == 5 ]; then
            learn_params="--lr_std 1e-6 --pretrain_iters 5000 --num_epochs 300000 --counter_end 20000"  
        fi

        dim="--N ${dims[i-1]}"
        
        ## Conveyor
        python run_experiment.py $convr_dyn $dim --old_brat True --deepreach_model exact $learn_params $exp_dir_convr --experiment_name BRAT_${dims[i-1]}D_p$j --wandb_name BRAT_${dims[i-1]}D_p$j $wandb_parms
        # python run_experiment.py $convr_dyn $dim --reach_only True $learn_params $exp_dir_convr --experiment_name reach_only_${dims[i-1]}D_p$j --wandb_name reach_only_${dims[i-1]}D_p${j} $wandb_parms
        # python run_experiment.py $convr_dyn $dim --avoid_only True $learn_params $exp_dir_convr --experiment_name avoid_only_${dims[i-1]}D_p$j --wandb_name avoid_only_${dims[i-1]}D_p${j} $wandb_parms

        ## Canoe
        # python run_experiment.py $canoe_dyn $dim --reach_1_only True $learn_params $exp_dir_canoe --experiment_name reach_1_only_${dims[i-1]}D_p$j --wandb_name reach_1_only_${dims[i-1]}D_p${j} $wandb_parms
        # python run_experiment.py $canoe_dyn $dim --reach_2_only True $learn_params $exp_dir_canoe --experiment_name reach_2_only_${dims[i-1]}D_p$j --wandb_name reach_2_only_${dims[i-1]}D_p${j} $wandb_parms

    done
done