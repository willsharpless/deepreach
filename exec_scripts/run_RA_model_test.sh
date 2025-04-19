#!/bin/bash

## Init

base_params="--tMax 2."
readonly convr_dyn="--dynamics_class ConveyorND $base_params"
readonly canoe_dyn="--dynamics_class CanoeND $base_params"
readonly convr_lam_dyn="--dynamics_class ConveyorNDlambda $base_params"
readonly canoe_lam_dyn="--dynamics_class CanoeNDlambda $base_params"

# readonly use_wandb=""
readonly use_wandb="--use_wandb"
readonly wandb_project="new_reach_avoid_models"
readonly wandb_parms="$use_wandb --wandb_project $wandb_project"

readonly exp_dir_convr="--experiments_dir ./runs/mulob/ConveyorND/new_RA_models/"
# readonly exp_dir_canoe="--experiments_dir ./runs/mulob/CanoeND/comparison_ND"

## Conveyor

i_max="7"
dims=()
for ((i=1; i<=${i_max}; i++)); do
    dims+=($((2 ** i)))
done

for i in $(seq 4 4); do

    if [ $i == 1 ]; then
        j="1"
        learn_params="--lr_std 2e-5 --pretrain_iters 500  --num_epochs 5000   --counter_end 500"
    elif [ $i == 2 ]; then
        j="2"
        learn_params="--lr_std 5e-6 --pretrain_iters 500 --num_epochs 10000  --counter_end 500"
    elif [ $i == 3 ]; then
        j="3"
        learn_params="--lr_std 5e-6 --pretrain_iters 1000 --num_epochs 50000  --counter_end 1000"
    elif [ $i == 4 ]; then
        j="4"
        learn_params="--lr_std 2e-6 --pretrain_iters 2000 --num_epochs 50000 --counter_end 5000"
    else
        j="5"
        learn_params="--lr_std 1e-6 --pretrain_iters 5000 --num_epochs 100000 --counter_end 10000" 
    fi

    # elif [ $i > 6 ]; then
    #     j="5"
    #     learn_params="--lr_std 1e-6 --pretrain_iters 5000 --num_epochs 300000 --counter_end 20000"  
    # fi

    dim="--N ${dims[i-1]}"

    # sampling_params="--spatial_sampling_type truncated_normal --spatial_sampling_std 0.3"
    # sampling_params="--boundary_sampling"
    # nc_loss_divisor="0.5"
    
    ## Conveyor

    # # # Baseline
    # python run_experiment.py $convr_dyn $dim --mulob_type BRAT --deepreach_model exact $learn_params $exp_dir_convr --experiment_name test --wandb_name BRAT_${dims[i-1]}D_exact_p$j $wandb_parms

    # # # R-A max bounding
    # # python run_experiment.py $convr_dyn $dim --mulob_type BRAT --deepreach_model exact_ra $learn_params $exp_dir_convr --experiment_name test --wandb_name BRAT_${dims[i-1]}D_exactRA_p$j $wandb_parms

    # # R-A sigmoid interpolation
    # python run_experiment.py $convr_dyn $dim --mulob_type BRAT --deepreach_model exact_ra_itp --sigmoid_slope 1.0 $learn_params $exp_dir_convr --experiment_name test --wandb_name BRAT_${dims[i-1]}D_exactRAitp_p$j $wandb_parms

    # # R-A dual-sigmoid interpolation
    # python run_experiment.py $convr_dyn $dim --mulob_type BRAT --deepreach_model exact_ra_itp_2sigma --sigmoid_slope 1.0 --sigmoid_slope_t 50.0 $learn_params $exp_dir_convr --experiment_name test --wandb_name BRAT_${dims[i-1]}D_exactRAitp_2sig_p$j $wandb_parms

    # Tuning

    python run_experiment.py $convr_dyn $dim --mulob_type BRAT --deepreach_model exact_ra_itp --sigmoid_slope 1.5 $learn_params $exp_dir_convr --experiment_name test --wandb_name BRAT_${dims[i-1]}D_exactRAitp_p$j $wandb_parms

    python run_experiment.py $convr_dyn $dim --mulob_type BRAT --deepreach_model exact_ra_itp_2sigma --sigmoid_slope 1.5 --sigmoid_slope_t 50.0 $learn_params $exp_dir_convr --experiment_name test --wandb_name BRAT_${dims[i-1]}D_exactRAitp_2sig_p$j $wandb_parms

    python run_experiment.py $convr_dyn $dim --mulob_type BRAT --deepreach_model exact_ra_itp_2sigma --sigmoid_slope 1.5 --sigmoid_slope_t 100.0 $learn_params $exp_dir_convr --experiment_name test --wandb_name BRAT_${dims[i-1]}D_exactRAitp_2sig_p$j $wandb_parms

    python run_experiment.py $convr_dyn $dim --mulob_type BRAT --deepreach_model exact_ra_itp_2sigma --sigmoid_slope 1.0 --sigmoid_slope_t 100.0 $learn_params $exp_dir_convr --experiment_name test --wandb_name BRAT_${dims[i-1]}D_exactRAitp_2sig_p$j $wandb_parms
    
    learn_params="--lr_std 5e-6 --pretrain_iters 2000 --num_epochs 50000 --counter_end 5000"

    python run_experiment.py $convr_dyn $dim --mulob_type BRAT --deepreach_model exact_ra_itp --sigmoid_slope 1.0 $learn_params $exp_dir_convr --experiment_name test --wandb_name BRAT_${dims[i-1]}D_exactRAitp_p$j $wandb_parms

    python run_experiment.py $convr_dyn $dim --mulob_type BRAT --deepreach_model exact_ra_itp_2sigma --sigmoid_slope 1.0 --sigmoid_slope_t 50.0 $learn_params $exp_dir_convr --experiment_name test --wandb_name BRAT_${dims[i-1]}D_exactRAitp_2sig_p$j $wandb_parms

done