#!/bin/bash

## Init

base_params="--tMax 2."
readonly convr_dyn="--dynamics_class ConveyorND $base_params"
readonly canoe_dyn="--dynamics_class CanoeND $base_params"
readonly convr_lam_dyn="--dynamics_class ConveyorNDlambda $base_params"
readonly canoe_lam_dyn="--dynamics_class CanoeNDlambda $base_params"

# readonly use_wandb=""
readonly use_wandb="--use_wandb"
readonly wandb_project="mulob_bc_sampling_test"
readonly wandb_parms="$use_wandb --wandb_project $wandb_project"

readonly exp_dir_convr="--experiments_dir ./runs/mulob/ConveyorND/BRAT_comparison_ND/"
# readonly exp_dir_canoe="--experiments_dir ./runs/mulob/CanoeND/comparison_ND"

## Conveyor

i_max="7"
dims=()
for ((i=1; i<=${i_max}; i++)); do
    dims+=($((2 ** i)))
done

for i in $(seq 3 3); do

    if [ $i == 1 ]; then
        j="1"
        learn_params="--lr_std 2e-5 --pretrain_iters 500  --num_epochs 5000   --counter_end 500"
    elif [ $i == 3 ]; then
        j="2"
        learn_params="--lr_std 5e-6 --pretrain_iters 1000 --num_epochs 50000  --counter_end 1000"
    elif [ $i == 4 ]; then
        j="3"
        learn_params="--lr_std 2e-6 --pretrain_iters 2000 --num_epochs 50000 --counter_end 5000"
    else
        j="4"
        learn_params="--lr_std 1e-6 --pretrain_iters 5000 --num_epochs 200000 --counter_end 10000" 
    fi
    # elif [ $i > 6 ]; then
    #     j="5"
    #     learn_params="--lr_std 1e-6 --pretrain_iters 5000 --num_epochs 300000 --counter_end 20000"  
    # fi

    dim="--N ${dims[i-1]}"

    # sampling_params="--spatial_sampling_type truncated_normal --spatial_sampling_std 0.3"
    sampling_params="--boundary_sampling"
    nc_loss_divisor="0.5"
    
    ## Conveyor

    # Test
    # python run_experiment.py $convr_dyn $dim --mulob_type BRAT --deepreach_model exact $learn_params $exp_dir_convr --experiment_name test --wandb_name BRAT_${dims[i-1]}D_baseline_p$j $wandb_parms

    # python run_experiment.py $convr_dyn $dim $sampling_params --boundary_sample_pts 1 --mulob_type BRAT --deepreach_model exact $learn_params $exp_dir_convr --experiment_name test --wandb_name BRAT_${dims[i-1]}D_bcsamp1_p$j $wandb_parms

    # Sweep
    # python run_experiment.py $convr_dyn $dim $sampling_params --boundary_sample_pts 1000 --mulob_type BRAT --deepreach_model exact $learn_params $exp_dir_convr --experiment_name test --wandb_name BRAT_${dims[i-1]}D_bcsamp100_p$j $wandb_parms

    # python run_experiment.py $convr_lam_dyn $dim $sampling_params --boundary_sample_pts 1000 --mulob_type BRAT --nc_lower_bound --ncss_value_loss_divisor $nc_loss_divisor --mulob_loss_type 'naive-combo self-supervision' --deepreach_model exact $learn_params $exp_dir_convr --experiment_name test --wandb_name BRAT_lam_${dims[i-1]}D_bcsamp100_ncss_decay_$nc_loss_divisor $wandb_parms

    python run_experiment.py $convr_dyn $dim $sampling_params --boundary_sample_pts 10000 --mulob_type BRAT --deepreach_model exact $learn_params $exp_dir_convr --experiment_name test --wandb_name BRAT_${dims[i-1]}D_bcsamp10k_p$j $wandb_parms

    python run_experiment.py $convr_lam_dyn $dim $sampling_params --boundary_sample_pts 10000 --mulob_type BRAT --nc_lower_bound --ncss_value_loss_divisor $nc_loss_divisor --mulob_loss_type 'naive-combo self-supervision' --deepreach_model exact $learn_params $exp_dir_convr --experiment_name test --wandb_name BRAT_lam_${dims[i-1]}D_bcsamp10k_ncss_decay_$nc_loss_divisor $wandb_parms

    python run_experiment.py $convr_dyn $dim $sampling_params --boundary_sample_pts 20000 --mulob_type BRAT --deepreach_model exact $learn_params $exp_dir_convr --experiment_name test --wandb_name BRAT_${dims[i-1]}D_bcsamp20k_p$j $wandb_parms

    python run_experiment.py $convr_lam_dyn $dim $sampling_params --boundary_sample_pts 20000 --mulob_type BRAT --nc_lower_bound --ncss_value_loss_divisor $nc_loss_divisor --mulob_loss_type 'naive-combo self-supervision' --deepreach_model exact $learn_params $exp_dir_convr --experiment_name test --wandb_name BRAT_lam_${dims[i-1]}D_bcsamp20k_ncss_decay_$nc_loss_divisor $wandb_parms

    python run_experiment.py $convr_dyn $dim $sampling_params --boundary_sample_pts 30000 --mulob_type BRAT --deepreach_model exact $learn_params $exp_dir_convr --experiment_name test --wandb_name BRAT_${dims[i-1]}D_bcsamp30k_p$j $wandb_parms

    python run_experiment.py $convr_lam_dyn $dim $sampling_params --boundary_sample_pts 30000 --mulob_type BRAT --nc_lower_bound --ncss_value_loss_divisor $nc_loss_divisor --mulob_loss_type 'naive-combo self-supervision' --deepreach_model exact $learn_params $exp_dir_convr --experiment_name test --wandb_name BRAT_lam_${dims[i-1]}D_bcsamp30k_ncss_decay_$nc_loss_divisor $wandb_parms

    learn_params="--lr_std 1e-5 --pretrain_iters 1000 --num_epochs 50000 --counter_end 1000 --numpoints 100000"

    python run_experiment.py $convr_dyn $dim $sampling_params --boundary_sample_pts 30000 --mulob_type BRAT --deepreach_model exact $learn_params $exp_dir_convr --experiment_name test --wandb_name BRAT_${dims[i-1]}D_bcsamp30k_ag_p$j $wandb_parms

    python run_experiment.py $convr_lam_dyn $dim $sampling_params --boundary_sample_pts 30000 --mulob_type BRAT --nc_lower_bound --ncss_value_loss_divisor $nc_loss_divisor --mulob_loss_type 'naive-combo self-supervision' --deepreach_model exact $learn_params $exp_dir_convr --experiment_name test --wandb_name BRAT_lam_${dims[i-1]}D_bcsamp30k_ag_ncss_decay_$nc_loss_divisor $wandb_parms

done